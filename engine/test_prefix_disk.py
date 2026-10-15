"""CPU persistence and fail-closed tests; no private prompts or checkpoint needed."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import torch

from engine.prefix_disk import PrefixDisk
from engine.prefix_persistence import PersistentPrefixes, namespace


def payload(tokens=(1, 2, 3, 4, 5, 6)):
    snapshots = {}
    for n in (2, 4, 6):
        w = min(n, 4)
        snapshots[n] = {
            'ids': tuple(tokens[:n]), 'route': 'route-a',
            'slots': torch.arange(n-w, n) % 8,
            'win': {0: torch.ones(w, 2, dtype=torch.bfloat16)},
            'pending': {0: None},
            'rep': {'h': torch.ones(w, 2, 3, dtype=torch.bfloat16),
                    'pre_mix': torch.ones(w, 2), 'topk': torch.zeros(w, 1, dtype=torch.int32),
                    'cand': None}}
    return {'ids': tuple(tokens), 'snapshots': snapshots,
            'ckv': {0: torch.arange(6).view(3, 2).to(torch.bfloat16)},
            'ik': {0: torch.arange(3).view(3, 1).to(torch.bfloat16)}}


class PrefixDiskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.disk = PrefixDisk(self.tmp.name, 'model-a', 1000000)

    def save(self, value=None):
        bid = uuid.uuid4().hex
        self.assertTrue(self.disk.save(bid, value or payload())['saved'])
        return bid

    def test_longest_boundary_and_restart(self):
        bid = self.save()
        disk = PrefixDisk(self.tmp.name, 'model-a', 1000000)
        self.assertEqual(disk.candidates([1, 2, 3, 4, 99]), [(bid, 4), (bid, 2)])
        restored = disk.load((bid, 4), [1, 2, 3, 4, 99])
        self.assertTrue(torch.equal(restored['ckv'][0], payload()['ckv'][0]))
        self.assertEqual(os.stat(Path(self.tmp.name) / (bid + '.pt')).st_mode & 0o777, 0o600)

    def test_multiple_prompts_and_exact_tokens(self):
        a = self.save()
        b = self.save(payload((7, 8, 9, 10, 11, 12)))
        self.assertEqual(self.disk.candidates([1, 2, 3, 4, 5, 6])[0], (a, 6))
        self.assertEqual(self.disk.candidates([7, 8, 9, 10, 11, 12])[0], (b, 6))
        self.assertIsNone(self.disk.load((a, 6), [7, 8, 9, 10, 11, 12]))

    def test_namespace_and_strict_routing(self):
        self.save()
        self.assertEqual(PrefixDisk(self.tmp.name, 'model-b', 1000000).candidates([1, 2, 3, 4]), [])
        self.assertEqual(self.disk.candidates([1, 2, 3, 4], 'route-b'), [])
        self.assertEqual(len(self.disk.candidates([1, 2, 3, 4], 'route-a')), 2)

    def test_attention_width_changes_persistent_namespace(self):
        model = Path(self.tmp.name) / 'model'
        (model / 'inference').mkdir(parents=True)
        for name in ('model.safetensors.index.json', 'inference/config.json', 'tokenizer.json'):
            (model / name).write_text('{}')
        engine = SimpleNamespace(model_dir=model, ep=SimpleNamespace(world=2, rank=0),
                                 prune_keep=0.6, act_quant=False, kernel='fp4',
                                 max_context=32768, swa_replay=True,
                                 args=SimpleNamespace(index_topk=512))
        old = namespace(engine)
        engine.args.index_topk = 1024
        self.assertNotEqual(namespace(engine), old)
        engine.args.index_topk = 512
        self.assertEqual(namespace(engine), old)

    def test_corrupt_and_missing_blob_are_misses(self):
        bid = self.save()
        path = Path(self.tmp.name) / (bid + '.pt')
        with path.open('r+b') as f:
            f.seek(100)
            f.write(b'bad')
        self.assertIsNone(self.disk.load((bid, 4), [1, 2, 3, 4]))
        path.unlink()
        self.assertIsNone(self.disk.load((bid, 4), [1, 2, 3, 4]))

    def test_restart_removes_only_unpublished_cache_artifacts(self):
        saved = self.save()
        orphan = Path(self.tmp.name) / (uuid.uuid4().hex + '.pt')
        temporary = Path(self.tmp.name) / '.writing-interrupted'
        unrelated = Path(self.tmp.name) / 'notes.txt'
        for path in (orphan, temporary, unrelated):
            path.write_bytes(b'test')
        PrefixDisk(self.tmp.name, 'model-a', 1000000)
        self.assertFalse(orphan.exists())
        self.assertFalse(temporary.exists())
        self.assertTrue(unrelated.exists())
        self.assertTrue((Path(self.tmp.name) / (saved + '.pt')).exists())

    def test_budget_eviction_and_oversized_rejection(self):
        a = self.save()
        size = (Path(self.tmp.name) / (a + '.pt')).stat().st_size
        self.disk.budget = size + 100
        b = self.save()
        self.assertFalse((Path(self.tmp.name) / (a + '.pt')).exists())
        self.assertTrue((Path(self.tmp.name) / (b + '.pt')).exists())
        self.disk.budget = 10
        self.assertFalse(self.disk.save(uuid.uuid4().hex, payload())['saved'])

    def test_serving_lists_only_live_matching_blobs(self):
        from engine.prefix_disk import token_hashes
        bid = self.save()
        value = payload()
        hashes = token_hashes(value['ids'], value['snapshots'])
        self.assertEqual(self.disk.serving({2: hashes[2], 4: hashes[4]}), {2: [bid], 4: [bid]})
        self.assertEqual(self.disk.serving({2: hashes[2], 5: 'missing'}), {2: [bid], 5: []})
        self.assertEqual(PrefixDisk(self.tmp.name, 'model-b', 1000000).serving({2: hashes[2]}),
                         {2: []})
        self.assertEqual(self.disk.serving({2: hashes[2]}, route='route-b'), {2: []})
        # The index row alone is not coverage: the blob file is what serves a restore.
        (Path(self.tmp.name) / (bid + '.pt')).unlink()
        self.assertEqual(self.disk.serving({2: hashes[2]}), {2: []})

    def adapter(self, peer_options=None, peer_load_ok=True):
        from engine.v41_engine import V41Engine
        p = object.__new__(PersistentPrefixes)
        p.disk, p.strict, p.pending, p.stats, p.log = self.disk, False, None, {}, lambda _: None
        p.dedup = False
        e = object.__new__(V41Engine)
        e.args = SimpleNamespace(compress_ratios={0: 2}, window_size=4,
                                 candidate_source_layer=0, head_dim=2, hc_mult=2, dim=3)
        e.caches = SimpleNamespace(ckv={0: torch.zeros(10, 2, dtype=torch.bfloat16)},
                                   ik={0: torch.zeros(10, 1, dtype=torch.bfloat16)},
                                   win=[torch.zeros(8, 2, dtype=torch.bfloat16)], pending={0: None})
        e.model = SimpleNamespace()
        e.device, e._prefix_route, e.swa_replay, e.replica_slots = 'cpu', 'route-a', True, 0
        e._prefix_cache, e._prefix_snapshots, e.prefix_disk = None, {}, p
        def gather(x):
            if isinstance(x, bool):
                return [x, peer_load_ok]
            return [x, peer_options if peer_options is not None else x]
        e.ep = SimpleNamespace(gather_objects=gather)
        p.engine = e
        return p

    def test_restore_into_existing_allocations(self):
        self.save()
        p = self.adapter()
        before = p.engine.caches.ckv[0].data_ptr()
        self.assertEqual(p.restore([1, 2, 3, 4, 99], 0), 4)
        self.assertEqual(p.engine.caches.ckv[0].data_ptr(), before)
        self.assertTrue(torch.equal(p.engine.caches.ckv[0][:2], payload()['ckv'][0][:2]))
        self.assertEqual(p.engine.caches.len, 4)

    def test_rank_missing_entry_or_failed_load_forces_shared_miss(self):
        self.save()
        self.assertEqual(self.adapter(peer_options=[]).restore([1, 2, 3, 4], 0), 0)
        p = self.adapter(peer_load_ok=False)
        self.assertEqual(p.restore([1, 2, 3, 4], 0), 0)
        self.assertEqual(p.engine.caches.ckv[0].count_nonzero(), 0)

    def test_packed_words_round_trip_and_bf16_rejected(self):
        value = payload()
        value['ckv'][0] = torch.randint(-(2**62), 2**62, (3, 36), dtype=torch.int64)
        self.save(value)
        p = self.adapter()
        self.assertEqual(p.restore([1, 2, 3, 4, 99], 0), 0)
        p.engine.caches.ckv[0] = torch.zeros(10, 36, dtype=torch.int64)
        before = p.engine.caches.ckv[0].data_ptr()
        self.assertEqual(p.restore([1, 2, 3, 4, 99], 0), 4)
        self.assertEqual(p.engine.caches.ckv[0].data_ptr(), before)
        self.assertTrue(torch.equal(p.engine.caches.ckv[0][:2], value['ckv'][0][:2]))

    def test_invalid_shape_rejected_before_gpu_state_changes(self):
        value = payload()
        value['ckv'][0] = torch.ones(3, 3, dtype=torch.bfloat16)
        self.save(value)
        p = self.adapter()
        self.assertEqual(p.restore([1, 2, 3, 4], 0), 0)
        self.assertEqual(p.engine.caches.ckv[0].count_nonzero(), 0)

    def saving_adapter(self, rank=0, dedup=False, peer_options=None):
        p = self.adapter(peer_options=peer_options)
        value = payload()
        p.engine._prefix_cache = value['snapshots'][6]
        p.engine._prefix_snapshots = dict(value['snapshots'])
        p.engine.caches.ckv = value['ckv']
        p.engine.caches.ik = value['ik']
        p.engine.ep.rank = rank
        p.engine.ep.broadcast_obj = Mock(side_effect=lambda value: value)
        p.min_interval_s, p._last_save_attempt = 300, None
        p.dedup = dedup
        from concurrent.futures import ThreadPoolExecutor
        p.pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(p.pool.shutdown)
        return p

    def test_save_interval_skips_staging_but_keeps_ram_and_disk_prefixes(self):
        p = self.saving_adapter()
        ids = list(payload()['ids'])
        with patch('engine.prefix_persistence.time.monotonic', return_value=100):
            p.save(ids)
        p.join()
        original = self.disk.candidates(ids)
        ram = p.engine._prefix_cache
        with patch('engine.prefix_persistence.time.monotonic', return_value=399), \
                patch('engine.prefix_persistence.cpu_tree') as copy, \
                patch.object(p, 'join') as join:
            p.save(ids)
            copy.assert_not_called()
            join.assert_not_called()
        self.assertIs(p.engine._prefix_cache, ram)
        self.assertEqual(self.disk.candidates(ids), original)
        with patch('engine.prefix_persistence.time.monotonic', return_value=400):
            p.save(ids)
        p.join()
        self.assertGreater(len(self.disk.candidates(ids)), len(original))
        self.assertEqual(p.engine.ep.broadcast_obj.call_count, 3)

    def test_peer_uses_rank_zero_decision_despite_local_clock(self):
        p = self.saving_adapter(rank=1)
        # A first save would be locally due, but rank zero says skip.
        with patch('engine.prefix_persistence.cpu_tree') as copy:
            p.save(list(payload()['ids']))
            copy.assert_not_called()
        # Conversely, a recent local attempt must not veto rank zero's save.
        p._last_save_attempt = 999999
        bid = uuid.uuid4().hex
        p.engine.ep.broadcast_obj.side_effect = lambda _: bid
        with patch('engine.prefix_persistence.time.monotonic', return_value=1):
            p.save(list(payload()['ids']))
        p.join()
        self.assertIn((bid, 6), self.disk.candidates(list(payload()['ids'])))
        self.assertEqual(p.engine.ep.broadcast_obj.call_count, 2)

    def test_zero_interval_saves_every_boundary(self):
        p = self.saving_adapter()
        p.min_interval_s = 0
        with patch('engine.prefix_persistence.time.monotonic', return_value=100):
            p.save(list(payload()['ids']))
            p.save(list(payload()['ids']))
        p.join()
        self.assertEqual(len(self.disk.candidates(list(payload()['ids']))), 6)

    def test_failed_staging_is_also_rate_limited(self):
        p = self.saving_adapter()
        with patch('engine.prefix_persistence.time.monotonic', return_value=100), \
                patch('engine.prefix_persistence.cpu_tree', side_effect=RuntimeError('copy failed')) as copy:
            p.save(list(payload()['ids']))
            p.save(list(payload()['ids']))
            self.assertEqual(copy.call_count, 1)
        self.assertIsNone(p.pending)

    def test_duplicate_boundaries_skip_staging_and_write(self):
        p = self.saving_adapter(dedup=True)
        ids = list(payload()['ids'])
        with patch('engine.prefix_persistence.time.monotonic', return_value=100):
            p.save(ids)
        p.join()
        original = self.disk.candidates(ids)
        self.assertEqual(len(original), 3)
        # A restart resets the interval; the same boundaries are already published.
        p._last_save_attempt = None
        with patch('engine.prefix_persistence.time.monotonic', return_value=1000), \
                patch('engine.prefix_persistence.cpu_tree') as copy:
            p.save(ids)
            copy.assert_not_called()
        self.assertIsNone(p.pending)
        self.assertEqual(self.disk.candidates(ids), original)
        # A skip is not a write, so the next new suffix may still be persisted at once.
        self.assertIsNone(p._last_save_attempt)
        self.assertEqual(p.stats['source'], 'disk-duplicate')

    def test_new_boundary_still_writes_after_a_duplicate_skip(self):
        p = self.saving_adapter(dedup=True)
        ids = list(payload()['ids'])
        with patch('engine.prefix_persistence.time.monotonic', return_value=100):
            p.save(ids)
        p.join()
        # Now the prompt grows by two tokens: boundary 8 has no stored hash, so this must
        # reach disk even though boundaries 2/4/6 are already there.
        grown = ids + [7, 8]
        snap8 = dict(payload()['snapshots'][6])
        snap8['ids'] = tuple(grown)
        p.engine._prefix_cache = snap8
        p.engine._prefix_snapshots = {6: payload()['snapshots'][6]}
        p.engine.caches.ckv = {0: torch.zeros(4, 2, dtype=torch.bfloat16)}
        p.engine.caches.ik = {0: torch.zeros(4, 1, dtype=torch.bfloat16)}
        p._last_save_attempt = None
        with patch('engine.prefix_persistence.time.monotonic', return_value=101):
            p.save(grown)
        p.join()
        self.assertTrue(any(n == 8 for _, n in self.disk.candidates(grown)))

    def first_save_then_restart(self, **kwargs):
        p = self.saving_adapter(dedup=True, **kwargs)
        ids = list(payload()['ids'])
        with patch('engine.prefix_persistence.time.monotonic', return_value=100):
            p.save(ids)
        p.join()
        p._last_save_attempt = None
        return p, ids

    def test_duplicate_needs_the_same_bundle_on_every_rank(self):
        # Rank 1 never wrote (or has evicted) the bundle rank 0 holds. Restore only uses
        # bundle ids common to both ranks, so skipping would make every restart miss.
        p, ids = self.first_save_then_restart(peer_options={2: [], 4: [], 6: []})
        before = len(self.disk.candidates(ids))
        with patch('engine.prefix_persistence.time.monotonic', return_value=1000):
            p.save(ids)
        p.join()
        self.assertEqual(len(self.disk.candidates(ids)), 2 * before)
        self.assertEqual(p._last_save_attempt, 1000)

    def test_duplicate_with_a_different_bundle_on_the_peer_still_writes(self):
        p, ids = self.first_save_then_restart(
            peer_options={n: [uuid.uuid4().hex] for n in (2, 4, 6)})
        before = len(self.disk.candidates(ids))
        with patch('engine.prefix_persistence.time.monotonic', return_value=1000):
            p.save(ids)
        p.join()
        self.assertEqual(len(self.disk.candidates(ids)), 2 * before)

    def test_failed_peer_lookup_forces_a_write(self):
        p, ids = self.first_save_then_restart()
        def gather(x):
            return [x, None]
        p.engine.ep.gather_objects = gather
        before = len(self.disk.candidates(ids))
        with patch('engine.prefix_persistence.time.monotonic', return_value=1000):
            p.save(ids)
        p.join()
        self.assertEqual(len(self.disk.candidates(ids)), 2 * before)

    def test_duplicate_touches_the_agreed_bundle_on_this_rank(self):
        p, ids = self.first_save_then_restart()
        (bid, _), = {(b, n) for b, n in self.disk.candidates(ids) if n == 6}
        with self.disk._db() as db:
            db.execute('UPDATE bundles SET used=0 WHERE id=?', (bid,))
        with patch('engine.prefix_persistence.time.monotonic', return_value=1000):
            p.save(ids)
        self.assertEqual(p.stats['source'], 'disk-duplicate')
        with self.disk._db() as db:
            used, = db.execute('SELECT used FROM bundles WHERE id=?', (bid,)).fetchone()
        self.assertGreater(used, 0)

    def test_duplicate_check_disabled_saves_every_due_boundary(self):
        p = self.saving_adapter(dedup=False)
        ids = list(payload()['ids'])
        with patch('engine.prefix_persistence.time.monotonic', return_value=100):
            p.save(ids)
            p.join()
            p._last_save_attempt = None
            p.save(ids)
            p.join()
        self.assertEqual(len(self.disk.candidates(ids)), 6)


if __name__ == '__main__':
    unittest.main()
