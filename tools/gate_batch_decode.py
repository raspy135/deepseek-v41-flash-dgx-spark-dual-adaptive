"""TP2 gate for batched multi-request decode (engine/batch_decode.py). Serving must be stopped.

    GATE_IMAGE=<image> GATE_LOG_DIR=results/batch-decode-<date> GATE_SOURCE_ROOT=<snapshot> \\
      GATE_ENV="DSV41_MAX_CONCURRENCY=4" GATE_TIMEOUT=1800s \\
      bash tools/run_two_node_gate.sh gate_batch_decode.py --out /app/results/batch-decode-<date>

The profile's expert layout and depth policy, swaps frozen, no prefix caches: every difference is
batching's.
Both ranks run this file identically; the runtime's actions are the scheduler's actions.

1. step      N lanes prefilled with different prompts and stopped at their first verify
             boundary; each lane's batched logits against that lane's own single-request
             FastDecoder.step at the same state. Reported bitwise, then as max |diff| and argmax.
2. requests  the N greedy prompts decoded together against each decoded alone by the engine's
             normal single-request path (fixed length, ignore_eos). Token equality.
3. speed     the same N fixed-length requests: one after another on the single-request path,
             one alone on the batched path, all N together. Wall time and aggregate tok/s.
4. isolation seeded sampling under interleaving, cancelling one lane, lane reuse.

Distinct prompts on purpose: identical ones route to identical experts and overstate the gain.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

sys.path[:0] = ['/app', '/app/tools']


def fail_fast(kind, value, tb):
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()
    # A container that exits this way can lose its last output; keep a copy on the results mount.
    try:
        with open(f"/app/results/gate-batch-decode-fail-rank{os.environ.get('RANK', '?')}.txt", 'w') as f:
            traceback.print_exception(kind, value, tb, file=f)
    except OSError:
        pass
    # A rank-local assertion must not leave the peer inside a collective until the gate timeout.
    os._exit(1)


sys.excepthook = fail_fast

# The profile's own layout and decode settings (dynamic experts, dynamic depth) stay as .env has
# them. Frozen here is only what would make the two arms differ for reasons other than batching:
# expert swaps (sensitivity off; probe prefill needs swaps, so it is off too) and prefix reuse.
# Greedy output is the same at any depth schedule, so the depth policy may choose freely.
for key, value in {'DSV41_ADAPT_SENSITIVITY': 'off', 'DSV41_PROBE_PREFILL': '0',
                   'DSV41_PREFIX_CACHE': '0', 'DSV41_PREFIX_DISK': '0', 'DSV41_PREFIX_RESPONSE': '0',
                   'DSV41_LOOKUP_DRAFT_ENABLED': '0', 'DSV41_DRAFT_BYPASS': '0',
                   'DSV41_GPU_TIMING': '0', 'DSV41_STEP_TIMING': '0', 'DSV41_ROUTE_STATS': '0'}.items():
    os.environ[key] = value
os.environ.setdefault('DSV41_MAX_CONCURRENCY', '4')
os.environ.setdefault('DSV41_LANE_MAX_SEQ', '32768')

import torch  # noqa: E402
import engine.v41_engine as V  # noqa: E402
from engine.decode_events import VerifyStep  # noqa: E402
from engine.expert_profiles import mask_digest  # noqa: E402
from server.app import Tok, load_encoding_module, build_chat_prompt  # noqa: E402

PROMPTS = [
    'Write a Python function that parses an ISO-8601 duration string into seconds, with tests.',
    'Write a clear explanation of how B-tree insertion and splitting preserve balance.',
    '日本語だけで、二分探索の仕組みと計算量を例を使って説明してください。',
    'Tell a short story about a lighthouse keeper who finds a message in a bottle.',
    'Compare TCP congestion control algorithms Reno, CUBIC and BBR in a few paragraphs.',
    'Write a small HTML page with CSS for a bakery menu with three sections.',
    'Explain the difference between a mutex and a semaphore with a concrete example in C.',
    'Summarize the causes and consequences of the 1929 stock market crash.',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=192)
    ap.add_argument('--max-seq', type=int, default=65536)
    ap.add_argument('--arena-gb', type=float, default=float(os.environ.get('GATE_ARENA_GB', '90.2')))
    ap.add_argument('--prune-keep', type=float, default=.61,
                    help='resident fraction; must fit the arena (0.61 needs ~88.4 GB, 0.57 ~82.4 GB)')
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=args.max_seq, arena_gb=args.arena_gb,
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=args.prune_keep, transient_slots=8, keep_free_gb=6,
                    expert_format='fp4')
    n = int(os.environ['DSV41_MAX_CONCURRENCY'])
    assert 2 <= n <= len(PROMPTS)
    rank = e.ep.rank
    tok, enc = Tok(root), load_encoding_module(root)
    ids = [build_chat_prompt({'messages': [{'role': 'user', 'content': p}]}, enc, tok, False, 75, e)[1]
           for p in PROMPTS[:n]]
    kw = dict(max_tokens=args.max_tokens, temperature=0.0, seed=42, ignore_eos=True)
    signature = hashlib.sha256(json.dumps([ids, kw, n]).encode()).hexdigest()
    assert len(set(e.ep.gather_objects(signature))) == 1, 'workload differs between ranks'
    maphash = mask_digest(e.model_prune_mask)
    report = {'lanes': n, 'max_tokens': args.max_tokens, 'arena_gb': args.arena_gb,
              'prune_keep': args.prune_keep,
              'max_seq': args.max_seq, 'lane_max_seq': os.environ['DSV41_LANE_MAX_SEQ'],
              'block': os.environ.get('DSV41_BLOCK', '5'),
              'block_dynamic': os.environ.get('DSV41_BLOCK_DYNAMIC', ''),
              'dynamic_experts': os.environ.get('DSV41_DYNAMIC_EXPERTS', '0'),
              'prompt_tokens': [len(x) for x in ids]}
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/rank{rank}.json'

    def emit(name, value):
        report[name] = value
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        if rank == 0:
            print('BATCH_GATE ' + json.dumps({name: value}), flush=True)

    def mark(phase):
        torch.cuda.synchronize()
        with open('/proc/meminfo') as f:
            avail = next(int(line.split()[1]) for line in f if line.startswith('MemAvailable')) / 1048576
        print(f'BATCH_GATE_MARK rank={rank} {phase}: allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, '
              f'reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB, host available {avail:.2f} GiB',
              flush=True)

    def agree(value, label):
        digest = hashlib.sha256(json.dumps(value).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1, f'ranks disagree: {label}'

    # ---- serial references on the normal single-request path (twice: the first captures graphs)
    def generate(prompt_ids, **kwargs):
        out = []
        for burst in e.generate(prompt_ids, **kwargs):
            out.extend(burst)
        return out

    references, serial_each = [], []
    for i in range(n):
        generate(ids[i], **kw)
        torch.cuda.synchronize()
        started = time.perf_counter()
        references.append(generate(ids[i], **kw))
        torch.cuda.synchronize()
        serial_each.append(time.perf_counter() - started)
    serial_s = sum(serial_each)
    agree(references, 'references')
    mark('references done')
    sampled_kw = dict(kw, temperature=0.7, seed=17)

    from engine.serving import DecodeRuntime

    def available_gb():
        with open('/proc/meminfo') as f:
            return next(int(line.split()[1]) for line in f if line.startswith('MemAvailable')) / 1048576

    torch.cuda.synchronize()
    before = (torch.cuda.memory_allocated(), available_gb())
    runtime = DecodeRuntime(e)
    torch.cuda.synchronize()
    assert runtime.n == n
    mark('runtime built')
    emit('lane_memory', {'extra_lanes': n - 1, 'lane_context': runtime.capacity,
                         'allocated_gb': round((torch.cuda.memory_allocated() - before[0]) / 1e9, 3),
                         'host_available_drop_gib': round(before[1] - available_gb(), 3)})

    def drive(jobs, cancel=None, on_ready=None):
        """jobs {lane: (ids, kwargs)}: the scheduler's loop without HTTP. Prefill is sequential."""
        out = {i: [] for i in jobs}
        for i, (prompt_ids, kwargs) in jobs.items():
            runtime.execute({'op': 'start', 'lane': i, 'prompt_ids': prompt_ids, 'kwargs': kwargs})
        active = set(jobs)
        while active:
            for i in sorted(active):
                event = runtime.events[i]
                if event is None:
                    active.discard(i)
                elif not isinstance(event, VerifyStep):
                    out[i].extend(event)
                    runtime.execute({'op': 'advance', 'lane': i})
            if cancel in active and isinstance(runtime.events[cancel], VerifyStep):
                runtime.execute({'op': 'close', 'lane': cancel})
                active.discard(cancel)
                cancel = None
            ready = sorted(i for i in active if isinstance(runtime.events[i], VerifyStep))
            if ready:
                if on_ready is not None:
                    on_ready(ready)
                    on_ready = None
                runtime.execute({'op': 'verify', 'lanes': ready})
                for i in ready:
                    runtime.execute({'op': 'advance', 'lane': i})
        return out

    jobs = {i: (ids[i], kw) for i in range(n)}
    # Capture the batched graphs and every lane's before anything is compared or timed.
    drive({i: (ids[i], dict(kw, max_tokens=8)) for i in range(n)})
    mark('batched graphs captured')

    # ---- 2. request equivalence
    got = drive(jobs)
    rows = []
    for i in range(n):
        mismatch = next((j for j, (x, y) in enumerate(zip(got[i], references[i])) if x != y), None)
        rows.append({'lane': i, 'tokens': len(got[i]), 'exact': got[i] == references[i],
                     'first_mismatch': mismatch})
    agree(got, 'concurrent outputs')
    emit('requests', rows)

    # ---- 1. step equivalence: every lane stopped at its first verify boundary takes its own
    # single-request step (reference), rolls back, takes the batched step, rolls back again; the
    # runtime's own verify then runs that step for real and the requests finish normally.
    step_rows = []

    def check(ready):
        assert ready == list(range(n)), ready
        refs = {}
        for i in ready:
            fd, step = runtime.engines[i].fast, runtime.events[i]
            logits, _ = fd.step(step.block, step.position, step.rows)
            refs[i] = logits.clone()
            fd.c.rollback(step.position)
        results = runtime.batch.step({i: (runtime.events[i].block, runtime.events[i].position,
                                          runtime.events[i].rows) for i in ready})
        torch.cuda.synchronize()
        for i in ready:
            ref, out = refs[i], results[i][0]
            step_rows.append({
                'lane': i, 'bitwise': bool(torch.equal(out, ref)),
                'max_abs_diff': float((out - ref).abs().max()),
                'argmax_agree': float((out.argmax(-1) == ref.argmax(-1)).float().mean())})
            runtime.engines[i].fast.c.rollback(runtime.events[i].position)

    drive({i: (ids[i], dict(kw, max_tokens=4)) for i in range(n)}, on_ready=check)
    agree(step_rows, 'step rows')
    emit('step', step_rows)

    # ---- 3. speed
    torch.cuda.synchronize()
    started = time.perf_counter()
    solo = drive({0: (ids[0], kw)})
    torch.cuda.synchronize()
    solo_s = time.perf_counter() - started
    advance_s = [0.0]
    execute = runtime.execute

    def timed_execute(action):
        t0 = time.perf_counter()
        try:
            return execute(action)
        finally:
            if action['op'] in ('advance', 'start'):
                advance_s[0] += time.perf_counter() - t0

    runtime.execute = timed_execute
    runtime.batch.profile = {}
    steps_before = runtime.batch.stats['steps']
    started = time.perf_counter()
    together = drive(jobs)
    torch.cuda.synchronize()
    together_s = time.perf_counter() - started
    verify_gpu = runtime.batch.profile_report() or {}
    runtime.batch.profile = None
    runtime.execute = execute
    rounds = runtime.batch.stats['steps'] - steps_before
    total = sum(len(x) for x in references)
    emit('speed', {
        'serial_single_request_path_s': round(serial_s, 3),
        'serial_tok_s': round(total / serial_s, 2),
        'runtime_one_request_s': round(solo_s, 3),
        'runtime_one_request_tok_s': round(len(solo[0]) / solo_s, 2),
        # the same prompt (0) both ways: the runtime runs a lone lane on its own graphs
        'single_request_path_one_request_s': round(serial_each[0], 3),
        'single_request_path_one_request_tok_s': round(len(references[0]) / serial_each[0], 2),
        'together_s': round(together_s, 3), 'together_tok_s': round(total / together_s, 2),
        'throughput_ratio': round(serial_s / together_s, 3),
        # The concurrent run split: batched verify on the GPU vs the wall time of the lanes'
        # advance actions (drafts, hash sync, accept loop -- these overlap the verify only partly).
        'together_verify_gpu_s': round(sum(verify_gpu.values()) / 1e3, 3),
        'together_verify_gpu_by_kind_s': {k: round(v / 1e3, 3) for k, v in verify_gpu.items()},
        'together_advance_wall_s': round(advance_s[0], 3),
        'together_rounds': rounds,
        'together_exact': all(together[i] == references[i] for i in range(n)),
        'batch': runtime.batch.memory_report(),
        'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3),
        'host_available_gib': round(available_gb(), 2)})

    # ---- 4. isolation
    # A sampled request's RNG draws depend on its verify depth, and the dynamic depth policy picks
    # depths from measured step times -- which interleaving changes by design. Pin the depth here,
    # so this compares the per-lane RNG streams and nothing else.
    policies = [getattr(engine, 'depth_policy', None) for engine in runtime.engines]
    for policy in policies:
        if policy is not None:
            policy.pinned = policy.hi
    sampled = {i: drive({i: (ids[i], sampled_kw)})[i] for i in range(n)}
    interleaved = drive({i: (ids[i], sampled_kw) for i in range(n)})
    for policy in policies:
        if policy is not None:
            policy.pinned = None
    cancelled = drive(jobs, cancel=0)
    reused = drive(jobs)
    isolation = {
        'sampling_interleaved_equal': all(interleaved[i] == sampled[i] for i in range(n)),
        'cancel_others_exact': all(cancelled[i] == references[i] for i in range(1, n)),
        'reuse_exact': all(reused[i] == references[i] for i in range(n)),
        'expert_map_unchanged': mask_digest(e.model_prune_mask) == maphash,
    }
    agree(isolation, 'isolation')
    emit('isolation', isolation)
    ok = (all(r['exact'] for r in rows) and all(isolation.values())
          and all(r['argmax_agree'] == 1.0 for r in step_rows))
    emit('pass', ok)
    assert all(e.ep.gather_objects(True))
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
