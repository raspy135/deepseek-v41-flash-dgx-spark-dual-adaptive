"""Scheduler lifecycle tests, without loading CUDA or model weights."""

import threading
import unittest
from types import SimpleNamespace

from engine.decode_events import VerifyStep
from server.concurrency import Scheduler, RequestStream


class FakeRuntime:
    def __init__(self):
        self.events, self.generators, self.stats, self.actions = {}, {}, {}, []
        self.verified = set()

    def execute(self, action):
        self.actions.append(action)
        op, lane = action['op'], action.get('lane')
        if op == 'start':
            self.generators[lane] = iter([VerifyStep(None, 0, None), [action['prompt_ids'][0] + 1]])
            event = action['prompt_ids']
        elif op == 'verify':
            self.verified.update(action['lanes'])
            return None
        elif op == 'close':
            event = None
        else:
            if isinstance(self.events[lane], VerifyStep):
                assert lane in self.verified
                self.verified.remove(lane)
            event = next(self.generators[lane], None)
        self.events[lane] = event
        if event is None:
            del self.generators[lane]
            self.stats[lane] = {'lane': lane}
        return event


def make_scheduler(n=None, capacity=None):
    runtime = FakeRuntime()
    if n is not None:
        runtime.n = n
    if capacity is not None:
        runtime.capacity = capacity
    state = SimpleNamespace(lock=threading.Lock(), ep_fault=None,
                            ep=SimpleNamespace(broadcast_request=lambda x: None,
                                               gather_objects=lambda x: [x, x]),
                            maintain_experts=lambda: None)
    # Hold the GPU lock so tests can enqueue a pair deterministically.
    state.lock.acquire()
    return Scheduler(state, runtime), state, runtime


class TestScheduler(unittest.TestCase):
    def test_pair_and_lane_reuse(self):
        scheduler, state, runtime = make_scheduler()
        a = scheduler.submit({'prompt_ids': [10], 'kwargs': {}})
        b = scheduler.submit({'prompt_ids': [20], 'kwargs': {}})
        state.lock.release()
        self.assertTrue(a.finished.wait(2))
        self.assertTrue(b.finished.wait(2))
        self.assertEqual(list(a), [[10], [11]])
        self.assertEqual(list(b), [[20], [21]])
        self.assertIn({'op': 'verify', 'lanes': [0, 1]}, runtime.actions)
        c = scheduler.submit({'prompt_ids': [30], 'kwargs': {}})
        self.assertTrue(c.finished.wait(2))
        self.assertEqual(list(c), [[30], [31]])
        self.assertIsNone(state.ep_fault)

    def test_cancel_before_admission(self):
        scheduler, state, runtime = make_scheduler()
        job = scheduler.submit({'prompt_ids': [10], 'kwargs': {}})
        job.cancelled.set()
        state.lock.release()
        self.assertTrue(job.finished.wait(2))
        self.assertEqual(list(job), [])
        self.assertEqual(runtime.actions, [])

    def test_failure_unblocks_consumers(self):
        scheduler, state, runtime = make_scheduler()
        runtime.execute = lambda _: (_ for _ in ()).throw(ValueError('test failure'))
        jobs = [scheduler.submit({'prompt_ids': [10], 'kwargs': {}}) for _ in range(3)]
        state.lock.release()
        for job in jobs:
            self.assertTrue(job.finished.wait(2))
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                next(job)
            job.close()
        self.assertIn('test failure', state.ep_fault)

    def test_cancel_active_lane_keeps_peer_and_queue_running(self):
        scheduler, state, runtime = make_scheduler()
        jobs = [scheduler.submit({'prompt_ids': [10 * i], 'kwargs': {}}) for i in (1, 2, 3)]
        execute = runtime.execute

        def cancelling(action):
            if action['op'] == 'verify' and action['lanes'] == [0, 1]:
                jobs[0].cancelled.set()
            return execute(action)

        runtime.execute = cancelling
        state.lock.release()
        for job in jobs:
            self.assertTrue(job.finished.wait(2))
        self.assertEqual(list(jobs[1]), [[20], [21]])
        self.assertEqual(list(jobs[2]), [[30], [31]])
        self.assertIn({'op': 'close', 'lane': 0}, runtime.actions)
        self.assertIsNone(state.ep_fault)

    def test_scheduled_peer_control_carries_rank_zero_decisions(self):
        """A lane's step gate must be the real broadcast: rank 1 decides nothing itself (its
        depth is a placeholder 0), so keeping its own value made it verify a root-only block
        against rank 0's full one. Here rank 1 passes 0 and must read back rank 0's 3 and 2."""
        from engine.serving import ScheduledPeer
        sent = []

        class Peer:
            rank, active = 1, True

            def control(self, keep_going, value=0, flag=0):
                sent.append((keep_going, value, flag))
                self.control_value, self.control_flag = 3, 2   # what rank 0 broadcast
                return True

        peer = ScheduledPeer(Peer())
        self.assertTrue(peer.control(True, 0, 0))
        self.assertEqual((peer.control_value, peer.control_flag), (3, 2))
        self.assertEqual(sent, [(True, 0, 0)])
        self.assertEqual(peer.rank, 1)          # __getattr__ still reaches the wrapped peer
        peer.release_peer()                     # termination stays a scheduler action
        self.assertEqual(len(sent), 1)

    def test_four_lanes_verify_together(self):
        scheduler, state, runtime = make_scheduler(n=4)
        jobs = [scheduler.submit({'prompt_ids': [10 * i], 'kwargs': {}}) for i in (1, 2, 3, 4)]
        state.lock.release()
        for i, job in enumerate(jobs, 1):
            self.assertTrue(job.finished.wait(2))
            self.assertEqual(list(job), [[10 * i], [10 * i + 1]])
        # Prefill is sequential (one start per lane), verification is one batched action.
        starts = [a['lane'] for a in runtime.actions if a['op'] == 'start']
        self.assertEqual(starts, [0, 1, 2, 3])
        self.assertIn({'op': 'verify', 'lanes': [0, 1, 2, 3]}, runtime.actions)
        self.assertEqual(sum(a['op'] == 'verify' for a in runtime.actions), 1)

    def test_long_request_takes_lane_zero_without_blocking_short_ones(self):
        # Lane 0 holds 1000 positions, the short lanes 100. need = prompt + max_tokens + 8.
        scheduler, state, runtime = make_scheduler(n=3, capacity=[1000, 100, 100])
        hold = scheduler.submit({'prompt_ids': [1], 'kwargs': {'max_tokens': 500}})
        long2 = scheduler.submit({'prompt_ids': [2], 'kwargs': {'max_tokens': 500}})
        short = scheduler.submit({'prompt_ids': [3], 'kwargs': {'max_tokens': 50}})
        state.lock.release()
        for job in (hold, long2, short):
            self.assertTrue(job.finished.wait(2))
        self.assertEqual((hold.lane, long2.lane, short.lane), (0, 0, 1))
        starts = [(a['lane'], a['prompt_ids'][0]) for a in runtime.actions if a['op'] == 'start']
        # The short request is admitted while the second long one still waits for lane 0.
        self.assertLess(starts.index((1, 3)), starts.index((0, 2)))
        self.assertIsNone(state.ep_fault)

    def test_short_request_prefers_a_short_lane(self):
        scheduler, state, runtime = make_scheduler(n=3, capacity=[1000, 100, 100])
        job = scheduler.submit({'prompt_ids': [3], 'kwargs': {'max_tokens': 50}})
        state.lock.release()
        self.assertTrue(job.finished.wait(2))
        self.assertEqual(job.lane, 1)

    def test_returning_conversation_goes_back_to_its_lane(self):
        scheduler, state, runtime = make_scheduler(n=3, capacity=[1000, 100, 100])
        runtime.engines = [SimpleNamespace(_prefix_cache=None, _prefix_snapshots={}),
                           SimpleNamespace(_prefix_cache=None, _prefix_snapshots={}),
                           SimpleNamespace(_prefix_cache={'ids': (7, 8, 9)}, _prefix_snapshots={})]
        job = scheduler.submit({'prompt_ids': [7, 8, 9, 10], 'kwargs': {'max_tokens': 20}})
        other = scheduler.submit({'prompt_ids': [1, 2], 'kwargs': {'max_tokens': 20}})
        state.lock.release()
        self.assertTrue(job.finished.wait(2) and other.finished.wait(2))
        self.assertEqual(job.lane, 2)       # its prefix lives on lane 2, not the default lane 1
        self.assertEqual(other.lane, 1)

    def test_request_larger_than_every_lane_fails_before_dispatch(self):
        scheduler, state, runtime = make_scheduler(n=2, capacity=[100, 50])
        job = scheduler.submit({'prompt_ids': [1], 'kwargs': {'max_tokens': 200}})
        state.lock.release()
        self.assertTrue(job.finished.wait(2))
        with self.assertRaisesRegex(ValueError, 'every lane'):
            next(job)
        self.assertEqual(runtime.actions, [])
        self.assertIsNone(state.ep_fault)

    def test_lane_capacities(self):
        from unittest.mock import patch
        from engine.serving import lane_capacities, concurrency
        with patch.dict('os.environ', {}, clear=False) as env:
            env.pop('DSV41_LANE_MAX_SEQ', None)
            self.assertEqual(lane_capacities(524288, 1), [524288])
            self.assertEqual(lane_capacities(524288, 3), [524288, 65536, 65536])
            self.assertEqual(lane_capacities(32768, 2), [32768, 32768])
            env['DSV41_LANE_MAX_SEQ'] = '16384'
            self.assertEqual(lane_capacities(524288, 2), [524288, 16384])
            env['DSV41_LANE_MAX_SEQ'] = '100'
            with self.assertRaises(ValueError):
                lane_capacities(524288, 2)
            env['DSV41_MAX_CONCURRENCY'] = '9'
            with self.assertRaises(ValueError):
                concurrency()
            env['DSV41_MAX_CONCURRENCY'] = '4'
            self.assertEqual(concurrency(), 4)

    def test_stream_eof_is_stable(self):
        job = RequestStream({})
        job.finish({'done': True})
        self.assertEqual(list(job), [])
        self.assertIsNone(next(job, None))
        job.close()

    def test_shutdown_cancels_queued_work(self):
        scheduler, state, runtime = make_scheduler()
        jobs = [scheduler.submit({'prompt_ids': [i], 'kwargs': {}}) for i in range(3)]
        scheduler.stop()
        state.lock.release()
        scheduler.thread.join(2)
        self.assertFalse(scheduler.thread.is_alive())
        self.assertTrue(all(job.finished.is_set() for job in jobs))
        self.assertEqual(runtime.actions, [])
        with self.assertRaisesRegex(RuntimeError, 'shutting down'):
            scheduler.submit({})


if __name__ == '__main__':
    unittest.main()
