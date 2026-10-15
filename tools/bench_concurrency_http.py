"""Small synthetic serving probe; no private captures or API credentials.

Run against a local server with DSV41_MAX_CONCURRENCY > 1. Times the same N requests one after
another and then all at once, each with a fixed output length (ignore_eos), and prints aggregate
output tok/s for both. Distinct prompts on purpose: identical ones route to the same experts and
overstate what batching saves. Results describe the server's current adaptation/cache settings,
not a frozen-weight quality A/B.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from urllib.request import Request, urlopen

PROMPTS = [
    'Write a clear explanation of how B-tree insertion and splitting preserve balance.',
    '日本語だけで、二分探索の仕組みと計算量を例を使って説明してください。',
    'Write a Python function that parses an ISO-8601 duration string into seconds, with tests.',
    'Tell a short story about a lighthouse keeper who finds a message in a bottle.',
    'Compare TCP congestion control algorithms Reno, CUBIC and BBR in a few paragraphs.',
    'Write a small HTML page with CSS for a bakery menu with three sections.',
    'Explain the difference between a mutex and a semaphore with a concrete example in C.',
    'Summarize the causes and consequences of the 1929 stock market crash.',
    'Write a Python script that reads a CSV file and prints summary statistics for every column, '
    'with tests.',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--clients', type=int, default=0,
                        help='concurrent requests (default: the server\'s lane count)')
    parser.add_argument('--rounds', type=int, default=2, help='concurrent rounds after one serial pass')
    parser.add_argument('--prompts', default='',
                        help='comma-separated PROMPTS indices, one per client (default: the first N); '
                             'e.g. 2,8 = two Python requests, 2,3 = Python + a story')
    args = parser.parse_args()
    if args.prompts:
        indices = [int(i) for i in args.prompts.split(',')]
        if any(not 0 <= i < len(PROMPTS) for i in indices):
            raise SystemExit(f'--prompts indices must be 0..{len(PROMPTS) - 1}')
        clients = len(indices)
    else:
        clients = args.clients
        if clients <= 0:
            with urlopen(args.url + '/health', timeout=30) as response:
                clients = int((json.load(response).get('concurrency') or {}).get('lanes', 2))
        if not 1 <= clients <= len(PROMPTS):
            raise SystemExit(f'--clients must be 1..{len(PROMPTS)}')
        indices = list(range(clients))

    def request(index):
        body = {'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': PROMPTS[index]}],
                'temperature': 0, 'max_tokens': args.tokens, 'stream': False,
                'reasoning_effort': 'none', 'ignore_eos': True}
        started = time.perf_counter()
        req = Request(args.url + '/v1/chat/completions', json.dumps(body).encode(),
                      {'Content-Type': 'application/json'})
        with urlopen(req, timeout=900) as response:
            result = json.load(response)
        assert result['choices'][0]['message']['content']
        stats = result.get('x_engine_stats', {})
        return {'prompt': index, 'wall_s': round(time.perf_counter() - started, 3),
                'tokens': result['usage']['completion_tokens'],
                'prefill_s': stats.get('prefill_s'), 'decode_tok_s': stats.get('decode_tok_s'),
                'accept': stats.get('accept_len_mean'), 'scheduler': stats.get('scheduler'),
                'prefix_cached_tokens': stats.get('prefix_cached_tokens')}

    # Warm every prompt (prefix cache, graphs for each lane's buckets), then time.
    for index in indices:
        print(json.dumps({'warmup': request(index)}), flush=True)
    for parallel in [False] + [True] * args.rounds:
        started = time.perf_counter()
        if parallel:
            with ThreadPoolExecutor(max_workers=clients) as pool:
                results = list(pool.map(request, indices))
        else:
            results = [request(i) for i in indices]
        elapsed = time.perf_counter() - started
        print(json.dumps({'parallel': parallel, 'clients': clients, 'prompts': indices,
                          'wall_s': round(elapsed, 3),
                          'aggregate_tok_s': round(sum(r['tokens'] for r in results) / elapsed, 2),
                          'requests': results}), flush=True)


if __name__ == '__main__':
    main()
