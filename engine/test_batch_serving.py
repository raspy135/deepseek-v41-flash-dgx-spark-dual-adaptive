"""DecodeRuntime's memory policy and lane reserve, without weights (engine/serving.py).

The releases are the part that can wedge the pair if they diverge: a graph one rank dropped is
recaptured with an eager warm-up that runs the step's collectives an extra time. So they must be
driven only by scheduler state both ranks share -- which these tests pin down.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from engine import serving
from engine.decode_events import VerifyStep


def runtime(n=2):
    rt = object.__new__(serving.DecodeRuntime)
    rt.n, rt.capacity = n, [524288] + [65536] * (n - 1)
    rt.batch = Mock()
    rt.batch.stats = {'steps': 0, 'lane_steps': 0}
    rt.batch.step.side_effect = lambda steps: {i: ('logits', 'mh') for i in steps}
    rt.engines = []
    for i in range(n):
        fast = Mock()
        fast.step.return_value = ('logits', 'mh')
        fast.route_idx.numel.return_value = 36
        lane = SimpleNamespace(fast=fast, args=SimpleNamespace(n_layers=40),
                               stats=lambda: {}, prefix_disk=None, expert_generation=0,
                               _prefix_route=None, _requests_since_plan=0,
                               store=SimpleNamespace(stats={}), device='cpu')
        rt.engines.append(lane)
    rt.generators, rt.events, rt.answers, rt.rng, rt.stats = {}, {}, {}, {}, {}
    rt.store_stats = [{'hits': 0} for _ in range(n)]
    rt.prefix_stats = [{} for _ in range(n)]
    rt.solo_steps = rt._solo_run = 0
    return rt


def ready(rt, lanes):
    for i in lanes:
        rt.events[i] = VerifyStep(block=None, position=0, rows=None)


class ReleasePolicyTests(unittest.TestCase):
    def test_finished_lanes_keep_their_graphs(self):
        # Releasing them at every request end cost each concurrent pair a recapture.
        rt = runtime(3)
        for i in (0, 2):
            rt.generators[i] = (_ for _ in ())   # a generator whose next send ends the request
            rt._advance(i)
        rt.batch.release_lane.assert_not_called()
        rt.batch.release_shared.assert_not_called()
        for lane in rt.engines:
            lane.fast.release_graphs.assert_not_called()

    def test_batched_graphs_kept_through_solo_steps_by_default(self):
        rt = runtime(2)
        ready(rt, [0, 1])
        rt.execute({'op': 'verify', 'lanes': [0, 1]})
        for _ in range(300):
            ready(rt, [0])
            rt.execute({'op': 'verify', 'lanes': [0]})
        rt.batch.release_shared.assert_not_called()

    def test_opt_in_release_after_a_run_of_solo_steps(self):
        rt = runtime(2)
        with patch.object(serving, 'SOLO_IDLE_STEPS', 4):
            for _ in range(3):
                ready(rt, [0])
                rt.execute({'op': 'verify', 'lanes': [0]})
            ready(rt, [0, 1])
            rt.execute({'op': 'verify', 'lanes': [0, 1]})          # a batched step resets the run
            for _ in range(3):
                ready(rt, [0])
                rt.execute({'op': 'verify', 'lanes': [0]})
            rt.batch.release_shared.assert_not_called()
            ready(rt, [0])
            rt.execute({'op': 'verify', 'lanes': [0]})
            rt.batch.release_shared.assert_called_once()
            for _ in range(5):                                    # once per run, not every step
                ready(rt, [0])
                rt.execute({'op': 'verify', 'lanes': [0]})
            rt.batch.release_shared.assert_called_once()


class StaleCoreTests(unittest.TestCase):
    """BatchDecoder keeps one context bucket of attention-core graphs a lane."""

    def decoder(self, held, limit=8):
        from engine.batch_decode import BatchDecoder
        bd = object.__new__(BatchDecoder)
        bd.core = {0: dict.fromkeys(held, 'graphs')}
        bd.core_limit = limit
        bd._release_core = Mock()
        return bd

    def test_same_bucket_other_parity_is_kept(self):
        bd = self.decoder([(0, 16384)])
        bd._drop_stale_core(0, (1, 16384))
        bd._release_core.assert_not_called()

    def test_held_key_is_kept(self):
        bd = self.decoder([(0, 16384), (1, 16384)])
        bd._drop_stale_core(0, (1, 16384))
        bd._release_core.assert_not_called()

    def test_new_bucket_drops_the_old_one(self):
        bd = self.decoder([(0, 16384), (1, 16384)])
        bd._drop_stale_core(0, (0, 32768))
        bd._release_core.assert_called_once_with(0)

    def test_limit_still_applies(self):
        bd = self.decoder([(0, 16384)], limit=1)
        bd._drop_stale_core(0, (1, 16384))
        bd._release_core.assert_called_once_with(0)


class BatchedTimingTests(unittest.TestCase):
    """A batched round's wall time is the round's: the depth policy must not learn from it."""

    def test_flag_follows_the_round(self):
        rt = runtime(2)
        ready(rt, [0, 1])
        rt.execute({'op': 'verify', 'lanes': [0, 1]})
        self.assertTrue(rt.engines[0]._last_verify_batched)
        self.assertTrue(rt.engines[1]._last_verify_batched)
        ready(rt, [0])
        rt.execute({'op': 'verify', 'lanes': [0]})
        self.assertFalse(rt.engines[0]._last_verify_batched)

    def test_batched_rounds_do_not_train_the_policy(self):
        from engine.v41_engine import V41Engine
        e = object.__new__(V41Engine)
        e.fast = SimpleNamespace(graph_captures=0)
        e._last_verify_batched = True
        self.assertIsNone(e._policy_step_s(0.0, 0))
        e._last_verify_batched = False
        self.assertIsNotNone(e._policy_step_s(0.0, 0))


class LaneReserveTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(head_dim=512, index_head_dim=128, rope_head_dim=64, n_layers=40,
                               index_n_heads=32, kv_source_layers=(2, 8, 14, 20),
                               compress_ratios=tuple([0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]))

    def test_nothing_reserved_at_concurrency_one(self):
        self.assertEqual(serving.lane_reserve_bytes(self.args(), 524288, 1), 0)

    def test_reserve_grows_with_lanes_and_lane_context(self):
        a = self.args()
        with patch.dict('os.environ', {'DSV41_LANE_MAX_SEQ': '65536', 'DSV41_LANE_RESERVE_GB': ''}):
            two = serving.lane_reserve_bytes(a, 524288, 2)
            four = serving.lane_reserve_bytes(a, 524288, 4)
        with patch.dict('os.environ', {'DSV41_LANE_MAX_SEQ': '32768', 'DSV41_LANE_RESERVE_GB': ''}):
            short = serving.lane_reserve_bytes(a, 524288, 2)
        self.assertGreater(four, two)
        self.assertGreater(two, short)
        # the measured fixed part alone: a 180 MB window ring a lane (RING 4096 x 43 x 512 x bf16)
        self.assertGreater(four - two, 2 * 180e6)

    def test_measured_allowance_overrides_the_estimate(self):
        a = self.args()
        with patch.dict('os.environ', {'DSV41_LANE_MAX_SEQ': '65536', 'DSV41_LANE_RESERVE_GB': '1.0'}):
            one = serving.lane_reserve_bytes(a, 524288, 2)
        with patch.dict('os.environ', {'DSV41_LANE_MAX_SEQ': '65536', 'DSV41_LANE_RESERVE_GB': '2.0'}):
            two = serving.lane_reserve_bytes(a, 524288, 2)
        self.assertAlmostEqual(two - one, 1e9, delta=1)


if __name__ == '__main__':
    unittest.main()
