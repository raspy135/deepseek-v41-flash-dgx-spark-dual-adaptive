import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
from server.app import State, Handler, GenerationResult
from server.concurrency import Scheduler
import queue
import threading


@patch.dict(os.environ, {'DSV41_PREFIX_RESPONSE': '1'})
class ResponseDeliveryTests(unittest.TestCase):
    def test_idle_queue_prioritizes_pending_requests_and_is_bounded(self):
        scheduler = object.__new__(Scheduler)
        scheduler.pending, scheduler.cache_jobs = queue.Queue(), queue.Queue(maxsize=2)
        scheduler.failure, scheduler.stopping = None, threading.Event()
        scheduler.n = 2
        scheduler.state = SimpleNamespace(lock=threading.Lock())
        scheduler._dispatch = Mock()
        for _ in range(4):
            scheduler.cache_response([1], [2], 0)
        self.assertEqual(scheduler.cache_jobs.qsize(), 2)
        scheduler.pending.put(object())
        scheduler._idle_cache()
        scheduler._dispatch.assert_not_called()
        scheduler.pending.get()
        scheduler._idle_cache()
        scheduler._dispatch.assert_called_once_with(
            {'op': 'cache_response', 'lane': 0, 'prompt_ids': [1], 'response_ids': [2]})

    def test_admission_and_peer_command(self):
        st = object.__new__(State)
        st.scheduler, st.ep_fault, st.ep_active = None, None, True
        st.ep, st.engine = Mock(), Mock()
        result = GenerationResult()
        result.gen_ids, result.router = [4, 5], SimpleNamespace(stopped=False)
        st.cache_response([1, 2], result)
        st.ep.broadcast_request.assert_called_once_with(
            {'cmd': 'cache_response', 'prompt_ids': [1, 2], 'response_ids': [4, 5]})
        for kwargs in ({'thinking': True}, {'vision': True}):
            st.engine.reset_mock()
            st.cache_response([1, 2], result, **kwargs)
            st.engine.cache_response.assert_not_called()
        st.scheduler = Mock()
        st.cache_response([1, 2], result)
        st.engine.cache_response.assert_not_called()
        st.scheduler.cache_response.assert_called_once_with([1, 2], [4, 5], None)

    def test_default_enabled_and_explicit_opt_out(self):
        st = object.__new__(State)
        st.scheduler, st.ep_fault, st.ep_active = None, None, False
        st.engine = Mock()
        result = GenerationResult()
        result.gen_ids, result.router = [3], SimpleNamespace(stopped=False)
        with patch.dict(os.environ):
            os.environ.pop('DSV41_PREFIX_RESPONSE', None)
            st.cache_response([1, 2], result)
        st.engine.cache_response.assert_called_once_with([1, 2], [3])
        st.engine.reset_mock()
        with patch.dict(os.environ, {'DSV41_PREFIX_RESPONSE': '0'}):
            st.cache_response([1, 2], result)
        st.engine.cache_response.assert_not_called()

    def test_response_flush_precedes_cache_work_stream_and_nonstream(self):
        for streaming in (False, True):
            events = []
            def generate(prompt, sampling, **kw):
                r = kw['result']
                r.gen_ids, r.router = [3], SimpleNamespace(content='answer')
                yield 'content', 'answer'
            h = object.__new__(Handler)
            h.server = SimpleNamespace(state=SimpleNamespace(
                model_name='test', request_context=nullcontext, generate=generate,
                cache_response=lambda *a: events.append('cache')))
            h.wfile = SimpleNamespace(flush=lambda: events.append('flush'))
            h._send_json = lambda *a: events.append('json')
            h._start_sse = lambda: None
            h._sse = lambda obj: events.append('done' if obj == b'[DONE]' else 'chunk')
            h._end_sse = lambda: events.append('end-flushed')
            h._completions({'prompt': [1, 2], 'max_tokens': 1, 'stream': streaming})
            self.assertEqual(events[-1], 'cache')
            self.assertEqual(events[-2], 'end-flushed' if streaming else 'flush')


if __name__ == '__main__':
    unittest.main()
