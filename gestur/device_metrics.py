"""Cheap, read-only runtime observations; missing hardware reports None."""

import os
import time
from pathlib import Path


def _read_number(path, divisor=1):
    try:
        return float(Path(path).read_text().strip()) / divisor
    except (OSError, ValueError):
        return None


class DeviceMetrics:
    def __init__(self, clock=time.monotonic, cpu_clock=time.process_time):
        self.clock, self.cpu_clock = clock, cpu_clock
        self._previous = None
        self._system_previous = None

    def sample(self):
        now, cpu = self.clock(), self.cpu_clock()
        percent = None
        if self._previous and now > self._previous[0]:
            percent = 100 * (cpu - self._previous[1]) / (now - self._previous[0])
        self._previous = now, cpu
        system_percent = None
        try:
            # user,nice,system,idle,iowait,irq,softirq,steal; guest is already
            # counted in user/nice and must not be included a second time.
            with Path("/proc/stat").open() as stream:
                counters = [int(value) for value in stream.readline().split()[1:9]]
            total, idle = sum(counters), sum(counters[3:5])
            if self._system_previous:
                elapsed, idle_elapsed = (
                    total - self._system_previous[0],
                    idle - self._system_previous[1],
                )
                if elapsed > 0:
                    system_percent = 100 * (elapsed - idle_elapsed) / elapsed
            self._system_previous = total, idle
        except (OSError, ValueError):
            pass
        temperature = _read_number("/sys/class/thermal/thermal_zone0/temp", 1000)
        return {
            "process_cpu_percent_one_core": round(percent, 2)
            if percent is not None
            else None,
            "system_cpu_percent": round(system_percent, 2)
            if system_percent is not None
            else None,
            "logical_cpus": os.cpu_count(),
            "cpu_temperature_c": temperature,
            "cpu_frequency_mhz": _read_number(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000
            ),
            "thermal_limit_near": temperature >= 78
            if temperature is not None
            else None,
        }
