"""Two-node response-prefix gate: exact next-turn outputs and disk restoration."""
import hashlib
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
os.environ['DSV41_PREFIX_RESPONSE'] = '1'
os.environ['DSV41_PREFIX_CACHE'] = '1'
os.environ['DSV41_PREFIX_DISK'] = '1'
os.environ['DSV41_PREFIX_DISK_STRICT'] = '1'
for k in ('DSV41_PRUNE_ADAPT', 'DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL'):
    os.environ[k] = '0'
import torch
import engine.v41_engine as V
from server.app import Tok, load_encoding_module, build_chat_prompt


def main():
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=786432, arena_gb=92,
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.63, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    def chat(messages):
        return build_chat_prompt({'messages': messages}, enc, tok, False, 75, e)[1]
    def run(ids, n):
        out = []
        for burst in e.generate(ids, max_tokens=n, temperature=0, seed=42):
            out.extend(burst)
        return out[:out.index(e.eos_token_id)] if e.eos_token_id in out else out
    messages = [{'role': 'user', 'content': 'Explain binary search in a short paragraph.'}]
    prompt = chat(messages)
    output = run(prompt, 96)
    before = e.model._want_counts.clone()
    report = e.cache_response(prompt, output)
    assert report['status'] == 'saved', report
    assert torch.equal(before, e.model._want_counts), 'replay changed routing demand'
    next_ids = chat(messages + [{'role': 'assistant', 'content': tok.decode(output)},
                               {'role': 'user', 'content': 'What is its time complexity? Be brief.'}])
    end = len(prompt) + len(output)
    assert next_ids[:end] == prompt + output, 'chat serialization changed the cached token prefix'
    # Force restore from disk, not the surviving in-memory snapshot.
    e.prefix_disk.join()
    run(chat([{'role': 'user', 'content': 'Reply OK.'}]), 8)
    cached = run(next_ids, 32)
    stats = dict(e.last_stats)
    assert stats['prefix_cached_tokens'] >= end, stats
    source = dict(e.prefix_disk.stats)
    assert source['source'] == 'disk', source
    # Fresh next-turn prefill, same expert mask, no prefix reuse.
    e._prefix_cache, e._prefix_snapshots = None, {}
    disk, e.prefix_disk = e.prefix_disk, None
    fresh = run(next_ids, 32)
    assert e.last_stats['prefix_cached_tokens'] == 0
    assert cached == fresh, 'cached next-turn output differs from fresh prefill'
    e.prefix_disk = disk
    # Exercise the scheduler's second lane and shared disk writer on both ranks.
    from engine.serving import DecodeRuntime
    from engine.decode_events import VerifyStep
    runtime = DecodeRuntime(e, lanes=2)
    def run_lane(ids, limit):
        output = []
        event = runtime.execute({'op': 'start', 'lane': 1, 'prompt_ids': ids,
                                 'kwargs': {'max_tokens': limit, 'temperature': 0, 'seed': 42}})
        while event is not None:
            if isinstance(event, VerifyStep):
                runtime.execute({'op': 'verify', 'lanes': [1]})
            else:
                output.extend(event)
            event = runtime.execute({'op': 'advance', 'lane': 1})
        return output[:output.index(e.eos_token_id)] if e.eos_token_id in output else output
    lane_prompt = chat([{'role': 'user', 'content': 'Name three primary colors.'}])
    lane_output = run_lane(lane_prompt, 16)
    runtime.execute({'op': 'cache_response', 'lane': 1,
                     'prompt_ids': lane_prompt, 'response_ids': lane_output})
    lane_end = len(lane_prompt) + len(lane_output)
    assert len(runtime.engines[1]._prefix_cache['ids']) == lane_end
    run_lane(lane_prompt + lane_output + tok.encode('\nContinue.'), 8)
    assert runtime.stats[1]['prefix_cached_tokens'] >= lane_end
    assert all(e.ep.gather_objects(True))
    if e.ep.rank == 0:
        print('RESPONSE_PREFIX_GATE_PASS ' + json.dumps({
            'extension': report, 'next_prompt_tokens': len(next_ids),
            'cached_tokens': stats['prefix_cached_tokens'], 'source': source,
            'cached_prefill_s': stats['prefill_s'], 'fresh_prefill_s': e.last_stats['prefill_s'],
            'output_hash': hashlib.sha256(json.dumps(cached).encode()).hexdigest()}), flush=True)
    disk.join()
    torch.cuda.synchronize()
    os._exit(0)


if __name__ == '__main__':
    main()
