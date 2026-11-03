"""Small opt-in Linux memory receipts for agentic control robot Native runs."""
import json
import os
from pathlib import Path
import resource
import time


def _read(path):
    try:
        return Path(path).read_text()
    except OSError as error:
        return f'UNAVAILABLE: {type(error).__name__}: {error}'


def record_memory(path, stage, **details):
    """Observe memory without imposing capacity waits or changing OOM priority."""
    if not path:
        return
    status = {}
    for line in _read('/proc/self/status').splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'VmSwap', 'RssAnon', 'RssFile', 'RssShmem', 'Threads'):
            status[key] = value.strip()
    meminfo = {}
    for line in _read('/proc/meminfo').splitlines():
        key, _, value = line.partition(':')
        if key in ('MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'SwapTotal', 'SwapFree'):
            meminfo[key] = value.strip()
    cgroup = {}
    for line in _read('/proc/self/cgroup').splitlines():
        if line.startswith('0::'):
            directory = Path('/sys/fs/cgroup')/line[3:].lstrip('/')
            cgroup = {'path': str(directory), **{name: _read(directory/name).strip()
                for name in ('memory.current', 'memory.peak', 'memory.max',
                             'memory.swap.current', 'memory.events')}}
            break
    receipt = dict(stage=stage, unix_s=time.time(), monotonic_s=time.monotonic(),
        pid=os.getpid(), process=status, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        host_meminfo=meminfo, cgroup=cgroup, details=details)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a') as stream:
        stream.write(json.dumps(receipt, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
