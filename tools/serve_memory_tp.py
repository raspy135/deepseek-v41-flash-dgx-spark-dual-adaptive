"""Host-memory trace of the real server through a mixed concurrent / long-context session.

The scenario that ran rank 1 into the memory watchdog on 2026-10-14 (DSV41_MAX_CONCURRENCY=2 on the
512K / 98.4 GB profile): two clients for a while, then one client whose conversation keeps growing.
Both ranks sample /proc/meminfo every 2 s ("MEM rank=R t=... avail_gib=..."), and rank 0 marks
the phases, so the low-water mark of each phase can be read off either log. Run it at
concurrency 1 and >1 with the same profile to see what concurrency costs.

    GATE_IMAGE=<image> GATE_LOG_DIR=results/serve-memory-<date> GATE_SOURCE_ROOT=<snapshot> \\
      GATE_ENV="DSV41_MAX_CONCURRENCY=2" GATE_TIMEOUT=3600s \\
      bash tools/run_two_node_gate.sh serve_memory_tp.py --arena-gb 97.4 --max-seq 524288

The long conversation is built from this repository's own docs/*.md (no private data); each turn
adds text and asks for a short answer, so the prefix cache serves the earlier turns.
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def sample_memory(rank, stop):
    start = time.time()
    while not stop.wait(2.0):
        with open('/proc/meminfo') as f:
            avail = next(int(line.split()[1]) for line in f if line.startswith('MemAvailable'))
        print(f'MEM rank={rank} t={time.time() - start:.0f} avail_gib={avail / 1048576:.2f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8011)
    ap.add_argument('--max-seq', type=int, default=524288)
    ap.add_argument('--arena-gb', type=float, default=97.4)
    ap.add_argument('--prune-keep', type=float, default=.61)
    ap.add_argument('--turns', type=int, default=5)
    ap.add_argument('--chars-per-turn', type=int, default=60000)    # ~15K tokens a turn
    ap.add_argument('--filler', default='/app/docs/*.md',
                    help='glob of plain-text files the long conversation is built from')
    args = ap.parse_args()
    rank = int(os.environ.get('RANK', '0'))
    stop = threading.Event()
    threading.Thread(target=sample_memory, args=(rank, stop), daemon=True).start()
    cmd = [sys.executable, '-m', 'server.app', '--model-dir', os.environ['MODEL_DIR'],
           '--host', '127.0.0.1', '--port', str(args.port), '--engine', 'v41',
           '--max-seq', str(args.max_seq), '--arena-gb', str(args.arena_gb),
           '--trace-stats', '/app/results/trace-union/stats/coverage.json',
           '--engine-kwargs', json.dumps({'prune_keep': args.prune_keep, 'expert_format': 'fp4',
                                          'transient_slots': 8, 'keep_free_gb': 6})]
    if rank != 0:
        worker = subprocess.Popen(cmd + ['--headless'], cwd='/app', stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in worker.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if 'EP2 worker: shutdown from rank 0' in line:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                sys.exit(0)
        sys.exit(worker.wait())

    server = subprocess.Popen(cmd, cwd='/app')
    url = f'http://127.0.0.1:{args.port}'

    def post(messages, tokens):
        body = {'model': 'deepseek-v4.1-flash', 'messages': messages, 'temperature': 0,
                'max_tokens': tokens, 'stream': False, 'reasoning_effort': 'none'}
        req = Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                      {'Content-Type': 'application/json'})
        with urlopen(req, timeout=1800) as response:
            result = json.load(response)
        return result['choices'][0]['message']['content'], result['usage']

    def mark(phase):
        with urlopen(url + '/health', timeout=30) as response:
            health = json.load(response)
        print('PHASE ' + json.dumps({'phase': phase, 'status': health.get('status'),
                                     'concurrency': health.get('concurrency'),
                                     'decode_memory': (health.get('engine_config') or {}).get('decode_memory')}),
              flush=True)

    try:
        deadline = time.time() + 900
        while True:
            if server.poll() is not None:
                raise SystemExit(f'server exited during startup: {server.returncode}')
            try:
                with urlopen(url + '/health', timeout=5) as response:
                    if json.load(response).get('status') == 'ok':
                        break
            except OSError:
                pass
            if time.time() > deadline:
                raise SystemExit('server did not become healthy')
            time.sleep(5)
        mark('ready')
        short = ['Explain how a B-tree stays balanced, briefly.',
                 'Write a Python function that merges two sorted lists, with a test.']
        for _ in range(2):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda p: post([{'role': 'user', 'content': p}], 200), short))
        mark('two clients done')
        text = ''.join(open(path).read() for path in sorted(glob.glob(args.filler)))
        if len(text) < args.turns * args.chars_per_turn:
            raise SystemExit(f'filler {args.filler!r} has {len(text)} characters, need '
                             f'{args.turns * args.chars_per_turn}')
        messages = []
        for turn in range(args.turns):
            chunk = text[turn * args.chars_per_turn:(turn + 1) * args.chars_per_turn]
            messages.append({'role': 'user', 'content': f'Document part {turn + 1}:\n\n{chunk}\n\n'
                             'Name the three most important points of this part in one sentence each.'})
            answer, usage = post(messages, 160)
            messages.append({'role': 'assistant', 'content': answer})
            print('TURN ' + json.dumps({'turn': turn + 1, 'prompt_tokens': usage['prompt_tokens'],
                                        'completion_tokens': usage['completion_tokens']}), flush=True)
            mark(f'long turn {turn + 1}')
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda m: post(m, 200),
                          [messages + [{'role': 'user', 'content': 'Summarize everything so far.'}],
                           [{'role': 'user', 'content': short[0]}]]))
        mark('long + short together')
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()
        stop.set()


if __name__ == '__main__':
    main()
