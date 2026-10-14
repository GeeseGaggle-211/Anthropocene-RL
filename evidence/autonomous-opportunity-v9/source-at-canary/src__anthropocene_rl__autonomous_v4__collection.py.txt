"""Bounded, schedule-independent native episode collection with IPC deadlines."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from anthropocene_rl.autonomous_v4.env import AutonomousTransport
from anthropocene_rl.autonomous_v4.training import episode


def episode_seed(seed, absolute_episode):
    key = f"autonomous-canary-v1:{seed}:{absolute_episode}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)


class DeadlineTransport(AutonomousTransport):
    def __init__(self, command, *, deadline, request_timeout=60):
        self.deadline, self.request_timeout = deadline, request_timeout
        super().__init__(command)

    def _request(self, payload):
        timeout = min(self.request_timeout, self.deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("Native campaign deadline exceeded")
        expired = threading.Event()

        def expire():
            expired.set()
            self.kill()

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            response = super()._request(payload)
            if expired.is_set():
                raise TimeoutError("Native IPC deadline exceeded")
            return response
        except Exception as error:
            if expired.is_set():
                raise TimeoutError("Native IPC deadline exceeded") from error
            raise
        finally:
            timer.cancel()

    def kill(self):
        with contextlib.suppress(ProcessLookupError):
            self._process.kill()

    def close(self):
        # No unbounded close RPC, including after campaign deadline.
        self.kill()
        self._process.wait(timeout=5)
        self._stderr_thread.join(timeout=1)


class Collector:
    """Native worlds per worker; immutable model shared only during collection."""

    transport_class = DeadlineTransport
    episode_runner = staticmethod(episode)

    def __init__(self, command, *, workers=4, deadline, request_timeout=60):
        if workers not in (1, 2, 4):
            raise ValueError("Only 1, 2, or 4 native workers are allowed")
        self.transports = [
            self.transport_class(command, deadline=deadline, request_timeout=request_timeout)
            for _ in range(workers)
        ]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.workers = workers

    def collect(self, model, bank, jobs, config, *, collect=True, verify=False):
        # Fixed lane assignment avoids overlapping use of any native process.
        # Order of completion is ignored; episode ordering is canonical.
        def lane(index):
            result = []
            for job_index in range(index, len(jobs), self.workers):
                job = jobs[job_index]
                batch, summary = self.episode_runner(
                    model,
                    self.transports[index],
                    bank,
                    job["entry"],
                    torch.Generator().manual_seed(job["sampler_seed"]),
                    dict(config, episode_limit=job.get("limit", config["episode_limit"])),
                    collect=collect,
                    verify=verify,
                )
                result.append((job_index, batch, summary))
            return result

        futures = [self.pool.submit(lane, i) for i in range(self.workers)]
        try:
            results = sorted(
                [item for future in futures for item in future.result()], key=lambda item: item[0]
            )
        except BaseException:
            for transport in self.transports:
                transport.kill()
            raise
        return [row for _, batch, _ in results for row in batch], [
            summary for _, _, summary in results
        ]

    def rollout(self, trainer, bank, entries, seed, remaining, *, verify=False):
        jobs = []
        for i in range(trainer.config["episodes_per_rollout"]):
            limit = min(trainer.config["episode_limit"], remaining)
            if limit <= 0:
                break
            index = trainer.episodes + i
            jobs.append(
                dict(
                    entry=entries[index % len(entries)],
                    sampler_seed=episode_seed(seed, index),
                    limit=limit,
                )
            )
            remaining -= limit
        rows, summaries = self.collect(trainer.model, bank, jobs, trainer.config, verify=verify)
        trainer.episodes += len(jobs)
        return rows, summaries

    def close(self):
        for transport in self.transports:
            transport.close()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
