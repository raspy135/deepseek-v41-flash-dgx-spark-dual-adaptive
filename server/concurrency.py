"""HTTP-to-GPU queue. All GPU work and peer messages have one owner thread.

Requests wait in a FIFO. A request is admitted to the smallest free lane whose context holds its
prompt + max_tokens; admission runs its whole prefill at once (prefill is sequential: the other
lanes pause at their verify boundary). Every round, each lane that reached a verify boundary is
verified in ONE batched step (engine/batch_decode.py), so the weights are read once per round for
all of them. A request blocked on the long-context lane does not hold back later requests that fit
a free short lane.
"""

import queue
import threading

from engine.decode_events import VerifyStep, event_signature


class RequestStream:
    def __init__(self, payload):
        self.payload = payload
        self.output = queue.Queue()
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.stats = {}
        self.exhausted = False
        self.lane = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        value = self.output.get()
        if isinstance(value, BaseException):
            self.exhausted = True
            raise value
        if value is None:
            self.exhausted = True
            raise StopIteration
        return value

    def finish(self, stats=None, error=None):
        self.stats = stats or {}
        self.output.put(error if error is not None else None)
        self.finished.set()

    def close(self):
        self.cancelled.set()
        # Do not return a request's stats or reuse its lane until BOTH ranks have
        # closed its generator. No GPU operations occur on this HTTP thread.
        self.finished.wait()


class Scheduler:
    def __init__(self, state, runtime):
        self.state, self.runtime = state, runtime
        self.n = int(getattr(runtime, 'n', 2))
        self.capacity = list(getattr(runtime, 'capacity', None) or [float('inf')] * self.n)
        self.pending = queue.Queue(maxsize=32)
        self.cache_jobs = queue.Queue(maxsize=2)
        self.failure = None
        self.stopping = threading.Event()
        self.submission_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='decode-scheduler', daemon=True)
        self.thread.start()

    def submit(self, payload):
        with self.submission_lock:
            if self.failure or self.stopping.is_set():
                raise RuntimeError(self.failure or 'server is shutting down')
            job = RequestStream(payload)
            self.pending.put_nowait(job)
        return job

    def stop(self):
        self.stopping.set()

    def cache_response(self, prompt, response, lane):
        if lane is None or not 0 <= lane < self.n or self.failure or self.stopping.is_set():
            return
        try:
            self.cache_jobs.put_nowait({'op': 'cache_response', 'lane': lane,
                                       'prompt_ids': list(prompt), 'response_ids': list(response)})
        except queue.Full:
            pass  # Optional work must never back-pressure response delivery.

    def _idle_cache(self):
        # Called only outside the active-request loop. Recheck priority under the
        # GPU lock; another HTTP thread can enqueue a request while we acquire it.
        with self.state.lock:
            if not self.pending.empty() or self.stopping.is_set():
                return
            try:
                action = self.cache_jobs.get_nowait()
            except queue.Empty:
                return
            self._dispatch(action)

    def _dispatch(self, action):
        self.state.ep.broadcast_request({'cmd': 'schedule', 'action': action})
        event = self.runtime.execute(action)
        # Catch completion/step-count divergence before the next action can put
        # one rank into a model collective and the other into the command queue.
        tag = event_signature(event)
        tags = self.state.ep.gather_objects(tag)
        if any(t != tag for t in tags):
            raise RuntimeError(f'concurrent decode ranks diverged: {tags}')
        return event

    @staticmethod
    def _need(job):
        """Positions the request can occupy: prompt, answer and the engine's 8-token margin."""
        payload = job.payload
        return (len(payload.get('prompt_ids') or ()) + int((payload.get('kwargs') or {}).get('max_tokens', 0))
                + 8)

    def _cached_prefix(self, lane, prompt):
        """Tokens of `prompt` the lane's RAM prefix state already holds. That state points into the
        lane's own KV cache, so it only ever covers the lane's last conversation: a returning
        conversation sent to another lane prefills again (or restores from disk, if enabled)."""
        engines = getattr(self.runtime, 'engines', None)
        if not engines:
            return 0
        engine = engines[lane]
        best = 0
        snaps = [getattr(engine, '_prefix_cache', None),
                 *(getattr(engine, '_prefix_snapshots', None) or {}).values()]
        for snap in snaps:
            ids = (snap or {}).get('ids') or ()
            if best < len(ids) <= len(prompt) and tuple(prompt[:len(ids)]) == tuple(ids):
                best = len(ids)
        return best

    def _place(self, job, free):
        """The free lane that holds `job`: the one whose RAM prefix covers most of its prompt,
        else the smallest. None until one fits; the server rejects a request larger than lane 0
        before it is queued, so an idle runtime always places it."""
        need = self._need(job)
        fits = [i for i in free if self.capacity[i] >= need]
        if not fits:
            return None
        prompt = job.payload.get('prompt_ids') or ()
        return min(fits, key=lambda i: (-self._cached_prefix(i, prompt), self.capacity[i], i))

    def _run(self):
        active, waiting = {}, []
        try:
            while True:
                if not active and not waiting:
                    try:
                        waiting.append(self.pending.get(timeout=0.1))
                    except queue.Empty:
                        if self.stopping.is_set():
                            return
                        self._idle_cache()
                        continue
                with self.state.lock:
                    while waiting or active:
                        if self.stopping.is_set():
                            for job in waiting + list(active.values()):
                                job.cancelled.set()
                        # Look ahead up to one request per lane, so a long request waiting for
                        # lane 0 does not keep a short one out of a free short lane.
                        while len(waiting) < self.n:
                            try:
                                waiting.append(self.pending.get_nowait())
                            except queue.Empty:
                                break
                        for job in list(waiting):
                            if job.cancelled.is_set() or self.stopping.is_set():
                                waiting.remove(job)
                                job.finish()
                                continue
                            if self._need(job) > max(self.capacity):
                                # Never placeable; fail it here, before anything reaches the peer.
                                waiting.remove(job)
                                job.finish(error=ValueError('request exceeds every lane context'))
                                continue
                            lane = self._place(job, [i for i in range(self.n) if i not in active])
                            if lane is None:
                                continue
                            waiting.remove(job)
                            job.lane = lane
                            active[lane] = job
                            # The whole prefill runs here; every other lane waits at its boundary.
                            event = self._dispatch({'op': 'start', 'lane': lane, **job.payload})
                            if event is not None and not isinstance(event, VerifyStep):
                                job.output.put(event)
                        for lane, job in list(active.items()):
                            event = self.runtime.events[lane]
                            if job.cancelled.is_set() and lane in self.runtime.generators:
                                event = self._dispatch({'op': 'close', 'lane': lane})
                            elif event is not None and not isinstance(event, VerifyStep):
                                event = self._dispatch({'op': 'advance', 'lane': lane})
                                if event is not None and not isinstance(event, VerifyStep):
                                    job.output.put(event)
                            if event is None:
                                job.finish(self.runtime.stats.get(lane))
                                del active[lane]
                                # Both ranks are now waiting for an action. All
                                # other requests are frozen, so adaptation is safe.
                                self.state.maintain_experts()
                        ready = sorted(i for i in active if isinstance(self.runtime.events[i], VerifyStep))
                        if ready:
                            self._dispatch({'op': 'verify', 'lanes': ready})
                            for lane in ready:
                                event = self._dispatch({'op': 'advance', 'lane': lane})
                                if event is not None and not isinstance(event, VerifyStep):
                                    active[lane].output.put(event)
        except BaseException as exc:
            with self.submission_lock:
                self.failure = f'{type(exc).__name__}: {exc}'
            self.state.ep_fault = self.failure
            for job in list(active.values()) + waiting:
                if not job.finished.is_set():
                    job.finish(error=RuntimeError(self.failure))
            while True:
                try:
                    self.pending.get_nowait().finish(error=RuntimeError(self.failure))
                except queue.Empty:
                    break
