"""Where a batched verify step's time goes (engine/batch_decode.py). Serving must be stopped.

    GATE_IMAGE=<image> GATE_LOG_DIR=results/batch-bench-<date> GATE_SOURCE_ROOT=<snapshot> \\
      GATE_ENV="DSV41_MAX_CONCURRENCY=4" bash tools/run_two_node_gate.sh bench_batch_decode_tp.py \\
      --out /app/results/batch-bench-<date> --arena-gb 84 --prune-keep 0.57

Every lane is prefilled with a different prompt and stopped at its first verify boundary. At that
fixed state, for k = 1..N lanes: k single-request FastDecoder.step calls one after another against
one batched step of the same k lanes (CUDA-event time, warm graphs, rolled back after every rep),
then the batched step's GPU time per graph kind. Same frozen-map settings as gate_batch_decode.py.
Only the verify step is timed here: drafts, Engram reads and the host accept loop are per lane in
both arms and are measured by the gate's end-to-end numbers instead.
"""

import argparse
import json
import os
import sys
import traceback

sys.path[:0] = ['/app', '/app/tools']


def fail_fast(kind, value, tb):
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()
    try:
        with open(f"/app/results/bench-batch-decode-fail-rank{os.environ.get('RANK', '?')}.txt", 'w') as f:
            traceback.print_exception(kind, value, tb, file=f)
    except OSError:
        pass
    os._exit(1)


sys.excepthook = fail_fast
for key, value in {'DSV41_DYNAMIC_EXPERTS': '0', 'DSV41_PROBE_PREFILL': '0',
                   'DSV41_BLOCK_DYNAMIC': '0', 'DSV41_BLOCK_CONFIDENCE': '0',
                   'DSV41_PRUNE_SWAP': '0', 'DSV41_PRUNE_SWAP_PREFILL': '0',
                   'DSV41_PREFIX_CACHE': '0', 'DSV41_PREFIX_DISK': '0', 'DSV41_PREFIX_RESPONSE': '0',
                   'DSV41_ADAPT_DECODE_TOKENS': '0', 'DSV41_ADAPT_URGENT': '0',
                   'DSV41_LOOKUP_DRAFT_ENABLED': '0', 'DSV41_DRAFT_BYPASS': '0',
                   'DSV41_GPU_TIMING': '0', 'DSV41_STEP_TIMING': '0', 'DSV41_ROUTE_STATS': '0'}.items():
    os.environ[key] = value
os.environ.setdefault('DSV41_MAX_CONCURRENCY', '4')
os.environ.setdefault('DSV41_LANE_MAX_SEQ', '32768')

import torch  # noqa: E402
import engine.v41_engine as V  # noqa: E402
from engine.decode_events import VerifyStep  # noqa: E402
from server.app import Tok, load_encoding_module, build_chat_prompt  # noqa: E402
from gate_batch_decode import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--max-seq', type=int, default=65536)
    ap.add_argument('--arena-gb', type=float, default=90.2)
    ap.add_argument('--prune-keep', type=float, default=.61)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=args.max_seq, arena_gb=args.arena_gb,
                    trace_stats='/app/results/trace-union/stats/coverage.json', spec=True,
                    prune_keep=args.prune_keep, transient_slots=8, keep_free_gb=6, expert_format='fp4')
    n = int(os.environ['DSV41_MAX_CONCURRENCY'])
    tok, enc = Tok(root), load_encoding_module(root)
    ids = [build_chat_prompt({'messages': [{'role': 'user', 'content': p}]}, enc, tok, False, 75, e)[1]
           for p in PROMPTS[:n]]
    # What one extra lane costs, piece by piece (engine/serving.py:clone_lane's steps).
    import copy
    import gc
    from engine.fastdecode import FastDecoder
    from engine.model import Caches, Model

    def alloc():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()

    lane_seq = int(os.environ['DSV41_LANE_MAX_SEQ'])
    m = [alloc()]
    caches = Caches(e.args, lane_seq, e.device); m.append(alloc())
    model = Model(e.W, e.store, caches, e.fast.m.moe_fn, act_quant=e.act_quant); m.append(alloc())
    model.hash_state = copy.deepcopy(e.model.hash_state); m.append(alloc())
    torch.cuda.memory._record_memory_history(max_entries=200000)
    fd = FastDecoder(model, e, use_graphs=True); m.append(alloc())
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    biggest = []
    for seg in snap['segments']:
        for block in seg['blocks']:
            if block['state'] != 'active_allocated' or not block.get('frames'):
                continue
            where = next((f"{f['filename'].split('/app/')[-1]}:{f['line']} {f['name']}"
                          for f in block['frames'] if '/app/' in f['filename']), '?')
            biggest.append((block['size'], where))
    biggest.sort(reverse=True)
    if e.ep.rank == 0:
        for size, where in biggest[:12]:
            print(f'BATCH_BENCH_ALLOC {size / 1e6:8.1f} MB  {where}', flush=True)
    one_lane = {name: round((b - a) / 1e6, 1) for name, a, b in
                zip(('caches', 'model', 'hash_state', 'fast_decoder'), m, m[1:])}
    one_lane['lane_max_seq'] = lane_seq
    if e.ep.rank == 0:
        print('BATCH_BENCH ' + json.dumps({'one_lane_mb': one_lane}), flush=True)
    del fd, model, caches
    gc.collect()
    torch.cuda.empty_cache()

    from engine.serving import DecodeRuntime
    runtime = DecodeRuntime(e)
    batch = runtime.batch
    kw = dict(max_tokens=24, temperature=0.0, seed=42, ignore_eos=True)
    report = {'lanes': n, 'block': os.environ.get('DSV41_BLOCK', '5'), 'reps': args.reps,
              'arena_gb': args.arena_gb, 'prune_keep': args.prune_keep, 'one_lane_mb': one_lane}

    def timed(fn):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def measure(ready):
        steps = {i: (runtime.events[i].block, runtime.events[i].position, runtime.events[i].rows)
                 for i in ready}

        def rollback(lanes):
            for i in lanes:
                runtime.engines[i].fast.c.rollback(steps[i][1])

        def serial(k):
            for i in range(k):
                runtime.engines[i].fast.step(*steps[i])
            rollback(range(k))

        def batched(k):
            batch.step({i: steps[i] for i in range(k)})
            rollback(range(k))

        rows = []
        for k in range(1, n + 1):
            serial(k), batched(k)          # capture every graph either arm needs
            s = sorted(timed(lambda: serial(k)) for _ in range(args.reps))
            b = sorted(timed(lambda: batched(k)) for _ in range(args.reps))
            rows.append({'lanes': k, 'serial_ms': round(s[len(s) // 2], 2),
                         'batched_ms': round(b[len(b) // 2], 2),
                         'ratio': round(s[len(s) // 2] / b[len(b) // 2], 3)})
            if e.ep.rank == 0:
                print('BATCH_BENCH ' + json.dumps(rows[-1]), flush=True)
        report['steps'] = rows
        for k in (1, n):
            batch.profile = {}
            for _ in range(args.reps):
                batched(k)
            per = {kind: round(ms / args.reps, 3) for kind, ms in batch.profile_report().items()}
            batch.profile = None
            per['sum'] = round(sum(per.values()), 3)
            report[f'profile_{k}'] = per
            if e.ep.rank == 0:
                print('BATCH_BENCH ' + json.dumps({f'profile_{k}': per}), flush=True)

    for i in range(n):
        runtime.execute({'op': 'start', 'lane': i, 'prompt_ids': ids[i], 'kwargs': kw})
        runtime.execute({'op': 'advance', 'lane': i})
    assert all(isinstance(runtime.events[i], VerifyStep) for i in range(n))
    measure(list(range(n)))
    for i in range(n):
        runtime.execute({'op': 'close', 'lane': i})
    os.makedirs(args.out, exist_ok=True)
    with open(f'{args.out}/rank{e.ep.rank}.json', 'w') as f:
        json.dump(report, f, indent=2)
    assert all(e.ep.gather_objects(True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
