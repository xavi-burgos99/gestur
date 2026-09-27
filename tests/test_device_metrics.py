import io
from pathlib import Path

from device_metrics import DeviceMetrics


def test_cpu_is_reported_per_core_and_linux_totals_do_not_double_count_guests(monkeypatch):
    now, cpu = [10.0], [2.0]
    counters = ['cpu 100 20 30 400 50 0 0 0 90 10\n']
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: io.StringIO(counters[0]))
    def read(path):
        return '79500' if str(path).endswith('/temp') else '2400000'
    monkeypatch.setattr(Path, 'read_text', read)
    monitor = DeviceMetrics(clock=lambda: now[0], cpu_clock=lambda: cpu[0])
    assert monitor.sample()['process_cpu_percent_one_core'] is None
    now[0], cpu[0] = 12.0, 5.0  # Native multithreading can legitimately exceed 100%.
    counters[0] = 'cpu 140 20 50 440 50 0 0 0 100 10\n'
    result = monitor.sample()
    assert result['process_cpu_percent_one_core'] == 150
    assert result['system_cpu_percent'] == 60
    assert result['cpu_temperature_c'] == 79.5
    assert result['cpu_frequency_mhz'] == 2400
    assert result['thermal_limit_near'] is True


def test_non_linux_hardware_is_unavailable_not_reported_as_zero(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(Path, 'open', unavailable)
    monkeypatch.setattr(Path, 'read_text', unavailable)
    result = DeviceMetrics().sample()
    for key in ('system_cpu_percent', 'cpu_temperature_c', 'cpu_frequency_mhz', 'thermal_limit_near'):
        assert result[key] is None
