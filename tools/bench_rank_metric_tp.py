"""Held-out NLL of resident sets ranked by each candidate demand metric, at one expert budget.

The demand database ranks experts for residency. DSV41_PRUNE_METRIC=score adds each unpruned
top-6 pick's raw router score (sqrt softplus), so a pick carrying 3% of a token's routing weight
votes nearly as much as one carrying 40% -- and since DSV41_EXPERT_TOPP=0.90 drops the low-share
picks in decode, some of what buys residency is never read. Candidates, each per (layer, expert)
over a token's unpruned top-6, folded per document as one request-unit vote like serving:

  frequency   1 a pick                                   (DSV41_PRUNE_METRIC=frequency)
  score       the pick's raw score                       (DSV41_PRUNE_METRIC=score, serving)
  share       its share of the token's routing weight    (each token-layer contributes 1)
  kept        1 if top-p 0.90 keeps the pick, else 0
  share_kept  share if top-p 0.90 keeps the pick, else 0

The engine runs the EXL3 all-weight profile (every routed expert resident), so a resident set is
emulated by the router mask alone: the masked router substitutes the next resident expert exactly
as a pruned arena does, and nothing is loaded. Demand is recorded over the trace corpora with the
full model (no mask, no top-p); each metric's resident set is built by the engine's own
global_residency.initial_selection at the same budget; each set scores corpus/heldout_eval.jsonl
teacher-forced with decode's top-p rule on every position (and, for `--full6` metrics, without
it, which is what a prefill computes). Reported: token-weighted NLL per category, its paired
difference from the reference metric (first in --metrics) with a per-sequence clustered SE, the
miss rate of all picks / of the picks top-p keeps / of routing share, and how many experts each
set differs from the reference's by. The reference arm is re-scored at the end on its first
sequences and must reproduce bit for bit. Use the disposable two-node gate with
DSV41_BENCH_EXPERT_FORMAT=exl3 DSV41_EXL3_EXACT_SLOTS=1, not serving.
"""
import argparse
import json
import os
import sys
import time

sys.path[:0] = ['/app', '/app/tools']
import bench_accept_ab_tp as AB  # noqa: E402  (sets the frozen-map environment first)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from bench_miss_policy_tp import clustered  # noqa: E402
from engine import expert_topp as ETP  # noqa: E402
from engine.expert_profiles import mask_digest  # noqa: E402
from engine.global_residency import initial_selection  # noqa: E402

V = AB.V
METRICS = ('frequency', 'score', 'share', 'kept', 'share_kept')
TOPP, MIN_K = 0.90, 3


class Demand:
    """Stands in for Model._record_prune_miss, which moe() calls with the UNMASKED logits and the
    raw scores of every routed layer. Collecting: per-document accumulators per metric. Scoring:
    device miss counters against the arm's mask. Never touches the engine's demand database."""

    def __init__(self, n_layers, n_experts, device):
        z = lambda: torch.zeros(n_layers, n_experts, dtype=torch.float64, device=device)  # noqa: E731
        self.doc = {m: z() for m in METRICS}
        self.total = {m: z() for m in METRICS}
        self.stats = torch.zeros(6, dtype=torch.float64, device=device)
        self.collecting = True

    def __call__(self, logits, scores, keep_mask, L, k, decode=False):
        want = logits.topk(k, dim=-1)[1]
        s = scores.gather(1, want).double()
        share = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        kept = ETP.keep_mask(s, TOPP, MIN_K).double()
        if self.collecting:
            flat = want.reshape(-1)
            for m, v in (('frequency', torch.ones_like(s)), ('score', s), ('share', share),
                         ('kept', kept), ('share_kept', share * kept)):
                self.doc[m][L].scatter_add_(0, flat, v.reshape(-1))
            return
        miss = (~keep_mask[want]).double()
        self.stats += torch.stack([miss.sum(), miss.new_tensor(float(miss.numel())),
                                   (miss * kept).sum(), kept.sum(), (miss * share).sum(), share.sum()])

    def end_document(self):
        """One request-unit vote per layer, as Model.flush_request_demand folds a request."""
        for m in METRICS:
            d = self.doc[m]
            self.total[m] += d / d.sum(dim=1, keepdim=True).clamp_min(1e-30)
            d.zero_()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train', default='/app/corpus/trace_corpus_v3.jsonl,/app/corpus/trace_corpus_v2.jsonl')
    ap.add_argument('--corpus', default='/app/corpus/heldout_eval.jsonl')
    ap.add_argument('--budget', type=int, default=9793, help='resident experts (FP4 98.4 GB: 9,793)')
    ap.add_argument('--metrics', default='score,frequency,share,kept,share_kept')
    ap.add_argument('--full6', default='score,share_kept', help='metrics also scored without top-p')
    ap.add_argument('--segment', type=int, default=512, help='training documents run as segments of this many tokens')
    ap.add_argument('--limit', type=int, default=0, help='first N held-out sequences (0 = all)')
    ap.add_argument('--repeat', type=int, default=6)
    args = ap.parse_args()
    metrics = args.metrics.split(',')
    full6 = [x for x in args.full6.split(',') if x]
    assert set(metrics) <= set(METRICS) and set(full6) <= set(metrics)
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=32768, arena_gb=float(os.environ.get('DSV41_BENCH_ARENA_GB', '98.4')),
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6,
                    expert_format=os.environ.get('DSV41_BENCH_EXPERT_FORMAT', 'exl3'))
    m = e.model
    nL, nE, k = e.args.n_layers, e.args.n_routed_experts, e.args.n_activated_experts
    full = e.model_prune_mask
    assert full and all(bool(full[L].all()) for L in full) and len(full) == nL, \
        'needs every routed expert resident (EXL3 all-weight profile): a mask is emulated, never loaded'
    maphash = mask_digest(full)
    rec = Demand(nL, nE, e.device)
    m._record_prune_miss = rec
    m.miss_policy, m.expert_topp = 'substitute', None

    def forward_nll(t):
        e._reset()
        m.begin_prompt()
        logits, _ = m.forward(t, 0, prefill=True, need_logits=True)
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        return -lp.gather(1, t[1:, None]).squeeze(1).cpu()

    # -- demand over the training corpora, full model
    seen, train = set(), []
    for path in args.train.split(','):
        for line in open(path):
            if line.strip():
                d = json.loads(line)
                if d['text'] not in seen:
                    seen.add(d['text'])
                    train.append(d)
    ttoks = [e.tokenizer.encode(d['text'], add_special_tokens=False) for d in train]
    seqs = [json.loads(l) for l in open(args.corpus) if l.strip()]
    if args.limit:
        seqs = seqs[:args.limit]
    toks = [e.tokenizer.encode(d['text'], add_special_tokens=False) for d in seqs]
    sig = json.dumps([len(t) for t in ttoks] + [len(t) for t in toks] + metrics + full6 + [args.budget])
    assert len(set(e.ep.gather_objects(sig))) == 1, 'workload differs between ranks'
    t0 = time.time()
    rec.collecting = True
    for t in ttoks:   # a document is one vote; a segment longer than one held-out sequence is not needed
        for i in range(0, len(t), args.segment):
            forward_nll(torch.tensor(t[i:i + args.segment], dtype=torch.long, device=e.device))
        rec.end_document()
    rec.collecting = False
    collect_s = round(time.time() - t0, 1)
    totals = {mm: rec.total[mm].cpu().numpy() for mm in METRICS}
    print('RANK_METRIC_COLLECT ' + json.dumps({'documents': len(ttoks), 'tokens': sum(map(len, ttoks)),
                                               'seconds': collect_s}), flush=True)

    # -- one resident set per metric, chosen on rank 0 (as the engine broadcasts its ranking)
    sel = None
    if e.ep.rank == 0:
        sel = {}
        for mm in metrics:
            sc = {L: totals[mm][L] / (totals[mm][L].sum() or 1.0) for L in range(nL)}
            sel[mm] = {L: sorted(int(x) for x in ids)
                       for L, ids in initial_selection(sc, args.budget, k).items()}
    sel = e.ep.broadcast_obj(sel)
    masks = {}
    for mm, keep in sel.items():
        masks[mm] = {L: torch.zeros(nE, dtype=torch.bool, device=e.device) for L in range(nL)}
        for L, ids in keep.items():
            masks[mm][L][ids] = True
        assert sum(int(x.sum()) for x in masks[mm].values()) == args.budget
    ref = metrics[0]
    os.makedirs(args.out, exist_ok=True)
    if e.ep.rank == 0:
        np.savez_compressed(f'{args.out}/demand.npz', **totals,
                            **{f'mask_{mm}': np.stack([masks[mm][L].cpu().numpy() for L in range(nL)])
                               for mm in metrics})
    cfg = e.config()
    path = f'{args.out}/rank{e.ep.rank}.json'
    report = {'arms': {}, 'corpus': args.corpus, 'sequences': len(seqs), 'train': args.train,
              'train_documents': len(ttoks), 'train_tokens': sum(map(len, ttoks)), 'collect_seconds': collect_s,
              'budget': args.budget, 'topp': [TOPP, MIN_K], 'reference': ref,
              'config': {kk: cfg.get(kk) for kk in ('expert_format', 'arena_slots', 'tp_experts')},
              'layer_counts': {mm: [len(sel[mm][L]) for L in range(nL)] for mm in metrics}}

    def score(mask, topp, which):
        m.prune_mask = mask
        m.expert_topp = (TOPP, MIN_K, 'orig') if topp else None
        rec.stats.zero_()
        out = [forward_nll(torch.tensor(toks[i], dtype=torch.long, device=e.device)) for i in which]
        m.prune_mask, m.expert_topp = full, None
        return out, rec.stats.tolist()

    arms = ([('all', None, True)] + [(mm, mm, True) for mm in metrics]
            + [('all/full6', None, False)] + [(f'{mm}/full6', mm, False) for mm in full6])
    base = {}
    for name, mm, topp in arms:
        assert len(set(e.ep.gather_objects(name))) == 1, 'arm differs between ranks'
        t0 = time.time()
        nll, st = score(full if mm is None else masks[mm], topp, range(len(seqs)))
        total = round(float(sum(x.double().sum() for x in nll)), 6)
        assert len(set(e.ep.gather_objects(total))) == 1, f'{name}: ranks scored different NLL'
        assert mask_digest(e.model_prune_mask) == maphash, 'engine map changed'
        group = 'topp' if topp else 'full6'
        if mm == ref:
            base[group] = nll
        if mm is None:
            base['all-' + group] = nll
        row = {'seconds': round(time.time() - t0, 1), 'metric': mm, 'topp': topp,
               'miss_rate': st[0] / max(1, st[1]), 'kept_miss_rate': st[2] / max(1, st[3]),
               'share_miss': st[4] / max(1e-30, st[5]), 'categories': {}}
        if mm is not None:
            row['experts_differing_from_ref'] = int(sum(int((masks[mm][L] & ~masks[ref][L]).sum())
                                                        for L in range(nL)))
        for cat in sorted({d['category'] for d in seqs}):
            idx = [i for i, d in enumerate(seqs) if d['category'] == cat]
            ns = [nll[i].numel() for i in idx]
            c = {'tokens': sum(ns), 'mean_nll': sum(float(nll[i].double().sum()) for i in idx) / sum(ns)}
            for tag, b in (('ref', base.get(group)), ('all', base.get('all-' + group))):
                if b is not None:
                    dm, se = clustered([float((nll[i].double() - b[i].double()).mean()) for i in idx], ns)
                    c[f'd_vs_{tag}'], c[f'se_vs_{tag}'] = dm, se
            row['categories'][cat] = c
        row['per_seq_mean_nll'] = [round(float(x.double().mean()), 6) for x in nll]
        report['arms'][name] = row
        print('RANK_METRIC_ARM ' + json.dumps({'arm': name, **{kk: v for kk, v in row.items()
                                                                 if kk != 'per_seq_mean_nll'}}), flush=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    rep = list(range(min(args.repeat, len(seqs))))
    again = score(masks[ref], True, rep)[0]
    same = all(torch.equal(a, base['topp'][i]) for a, i in zip(again, rep))
    report['reference_repeat_identical'] = same
    print('RANK_METRIC_REPEAT ' + json.dumps({'sequences': len(rep), 'identical': same}), flush=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    for p in (path, f'{args.out}/demand.npz'):
        if os.path.exists(p):
            os.chown(p, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
