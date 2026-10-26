"""Own native GUI workers and their engine children for exactly one split.

This module deliberately has no model, prompt, image, or scoring policy.
"""
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import traceback


def _worker(index, group, tasks, events, initialize, infer, close, args):
    # vLLM engine children inherit this private session/process group. The
    # parent can therefore clean up this worker's children even after it dies.
    os.setsid()
    events.put(('started', index, os.getpid()))
    try:
        initialize(group, args)
        events.put(('ready', index, None))
        while True:
            task = tasks.get()
            if task is None:
                break
            task_id, sample = task
            events.put(('working', index, task_id))
            events.put(('result', index, (task_id, infer(sample))))
    except BaseException:
        events.put(('error', index, traceback.format_exc()))
    finally:
        try:
            close()
        except BaseException:
            events.put(('error', index, 'Model cleanup failed:\n' + traceback.format_exc()))
        events.put(('stopped', index, None))


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def _cleanup(processes, groups, queues, shutdown_timeout):
    # Request normal shutdown first. A failed initializer is never replaced,
    # and there is no consumable GPU-token queue to deadlock on.
    for tasks in queues:
        tasks.put(None)
    deadline = time.monotonic() + shutdown_timeout
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    for process in processes:
        # If a process died before its started message was consumed, a live
        # process is still ours, and its session identity can be checked here.
        if process.is_alive():
            try:
                if os.getpgid(process.pid) == process.pid:
                    groups.add(process.pid)
            except ProcessLookupError:
                pass
            if process.pid not in groups:
                process.terminate()
    remaining = {pgid for pgid in groups if _signal_group(pgid, signal.SIGTERM)}
    deadline = time.monotonic() + min(5, shutdown_timeout)
    while remaining and time.monotonic() < deadline:
        for process in processes:
            process.join(timeout=.01)
        remaining = {pgid for pgid in remaining if _signal_group(pgid, 0)}
        if remaining:
            time.sleep(.05)
    for pgid in remaining:
        _signal_group(pgid, signal.SIGKILL)
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)


def run_replicas(samples, gpu_groups, initialize, infer, close, args,
                 *, startup_timeout=1800, task_timeout=1800, shutdown_timeout=60):
    """Yield one result per sample, or raise on any startup/task/cleanup failure.

    GPU assignments are explicit and stable. Each worker loads its model once,
    serves one task at a time, then closes it before this generator finishes.
    No worker or failed sample is automatically replaced/re-sampled.
    """
    if not gpu_groups or any(not group for group in gpu_groups):
        raise ValueError('At least one nonempty GPU group is required')
    if min(startup_timeout, task_timeout, shutdown_timeout) <= 0:
        raise ValueError('Worker timeouts must be positive')
    if not samples:
        return
    context = mp.get_context('spawn')
    events = context.Queue()
    queues = [context.Queue() for _ in gpu_groups]
    processes, groups = [], set()
    old_sigterm = None

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    if threading.current_thread() is threading.main_thread():
        old_sigterm = signal.signal(signal.SIGTERM, interrupted)

    def receive(deadline, phase):
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'GUI workers timed out during {phase}')
            try:
                event = events.get(timeout=min(.2, max(.001, deadline - time.monotonic())))
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(processes) if p.exitcode is not None]
                if dead:
                    raise RuntimeError(f'GUI worker exited before {phase} completed: {dead}')
                continue
            kind, index, payload = event
            if kind == 'started':
                if payload != processes[index].pid:
                    raise RuntimeError('Unexpected GUI worker identity')
                groups.add(payload)
                continue
            if kind == 'error':
                raise RuntimeError(f'GUI worker {index} failed:\n{payload}')
            return event

    try:
        for index, group in enumerate(gpu_groups):
            process = context.Process(target=_worker,
                args=(index, group, queues[index], events, initialize, infer, close, args),
                daemon=False)
            process.start()
            processes.append(process)
        ready = set()
        deadline = time.monotonic() + startup_timeout
        while len(ready) < len(processes):
            kind, index, _ = receive(deadline, 'model initialization')
            if kind != 'ready' or index in ready:
                raise RuntimeError(f'Unexpected GUI startup event: {kind}, worker {index}')
            ready.add(index)

        pending, completed = {}, set()
        source = iter(enumerate(samples))

        def dispatch(index):
            item = next(source, None)
            if item is not None:
                pending[index] = (item[0], time.monotonic() + task_timeout)
                queues[index].put(item)

        for index in range(len(processes)):
            dispatch(index)
        while pending:
            kind, index, payload = receive(min(v[1] for v in pending.values()), 'inference')
            if kind == 'working' and index in pending and payload == pending[index][0]:
                continue
            if kind != 'result' or index not in pending or payload[0] != pending[index][0]:
                raise RuntimeError(f'Unexpected GUI inference event: {kind}, worker {index}')
            task_id, result = payload
            if task_id in completed:
                raise RuntimeError(f'Duplicate GUI sample result: {task_id}')
            completed.add(task_id)
            del pending[index]
            dispatch(index)
            yield result
        if len(completed) != len(samples):
            raise RuntimeError('GUI inference returned an incomplete sample set')
        for tasks in queues:
            tasks.put(None)
        stopped = set()
        deadline = time.monotonic() + shutdown_timeout
        while len(stopped) < len(processes):
            # A previously acknowledged worker can exit normally while others
            # still close their engines; do not treat that as an inference error.
            try:
                kind, index, payload = events.get(timeout=min(.2, max(.001, deadline-time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError('GUI workers timed out during model shutdown')
                dead = [(i, p.exitcode) for i, p in enumerate(processes)
                        if i not in stopped and p.exitcode is not None]
                if dead:
                    raise RuntimeError(f'GUI worker exited without cleanup confirmation: {dead}')
                continue
            if kind == 'error':
                raise RuntimeError(f'GUI worker {index} failed:\n{payload}')
            if kind != 'stopped' or index in stopped:
                raise RuntimeError(f'Unexpected GUI shutdown event: {kind}, worker {index}')
            stopped.add(index)
    finally:
        try:
            _cleanup(processes, groups, queues, shutdown_timeout)
        finally:
            for channel in [*queues, events]:
                channel.cancel_join_thread()
                channel.close()
            if old_sigterm is not None:
                signal.signal(signal.SIGTERM, old_sigterm)
