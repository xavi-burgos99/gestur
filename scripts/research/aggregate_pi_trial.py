#!/usr/bin/env python3
"""Aggregate an already finished GESTUR trial into a new output directory.

Copies manifest.json and summary.json byte-for-byte; does not copy telemetry.
Produces telemetry-summary.json, windows-60s.csv and windows-60s.md.
No device access, model execution, conclusions or modifications to source data.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import mean, median


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def stats(values):
    values = [value for value in values if number(value)]
    return {
        "samples": len(values),
        "mean": mean(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
    }


def values(rows, key):
    return [(row.get("hardware") or {}).get(key) for row in rows]


def fps_between(points):
    if len(points) < 2 or any(b[1] < a[1] for a, b in zip(points, points[1:])):
        return {"samples": len(points), "fps": None}
    start, first = points[0]
    end, last = points[-1]
    return {
        "samples": len(points),
        "fps": (last - first) / (end - start),
        "first_runtime_seconds": start,
        "last_runtime_seconds": end,
        "duration_seconds": end - start,
        "drawn_frames": last - first,
    }


def spanish(value, digits=2):
    return "—" if not number(value) else f"{value:.{digits}f}".replace(".", ",")


def aggregate(source, output):
    # Validate the source manifest before copying it unchanged.
    json.loads((source / "manifest.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    telemetry = source / "telemetry.jsonl"
    rows, malformed = [], 0
    for line in telemetry.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(row, dict) and number(row.get("elapsed_seconds")):
            rows.append(row)
        else:
            malformed += 1
    rows.sort(key=lambda row: row["elapsed_seconds"])
    if not rows:
        raise ValueError("No valid telemetry samples")
    cpu = [row for row in rows if row["elapsed_seconds"] >= 5]
    duration = rows[-1]["elapsed_seconds"]
    intervals = [
        b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in zip(rows, rows[1:])
    ]
    offsets = [
        row["timestamp"] - row["elapsed_seconds"]
        for row in rows
        if number(row.get("timestamp"))
    ]
    origin = median(offsets) if offsets else None
    runtimes, conflicts, repeats = {}, set(), 0
    for row in rows:
        runtime = row.get("runtime") or {}
        stamp = runtime.get("updated_at")
        count = (runtime.get("render") or {}).get("frames")
        if (
            origin is None
            or not number(stamp)
            or not number(count)
            or count < 0
            or stamp < origin
        ):
            continue
        if stamp in runtimes:
            repeats += 1
            if runtimes[stamp] != count:
                conflicts.add(stamp)
        runtimes[stamp] = count
    points = [
        (stamp - origin, count)
        for stamp, count in sorted(runtimes.items())
        if stamp not in conflicts
    ]
    windows = []
    for index in range(math.floor(duration / 60) + 1):
        start, end = index * 60, (index + 1) * 60
        samples = [row for row in rows if start <= row["elapsed_seconds"] < end]
        if not samples:
            continue
        cpu_samples = [row for row in samples if row["elapsed_seconds"] >= 5]
        draws = fps_between([point for point in points if start <= point[0] < end])
        temp = stats(values(samples, "cpu_temperature_c"))
        rss = stats(values(samples, "process_rss_mib"))
        windows.append(
            {
                "start_seconds": start,
                "end_seconds_exclusive": end,
                "first_telemetry_seconds": samples[0]["elapsed_seconds"],
                "last_telemetry_seconds": samples[-1]["elapsed_seconds"],
                "telemetry_samples": len(samples),
                "draw": draws,
                "system_cpu_percent": stats(values(cpu_samples, "system_cpu_percent")),
                "process_cpu_percent_one_core": stats(
                    values(cpu_samples, "process_cpu_percent_one_core")
                ),
                "temperature_c": temp,
                "rss_mib": rss,
            }
        )
    throttles = [(row.get("hardware") or {}).get("throttling") for row in rows]
    valid_throttles = [
        value
        for value in throttles
        if isinstance(value, dict) and isinstance(value.get("raw"), str)
    ]
    flag_names = (
        "undervoltage",
        "frequency_capped",
        "throttled",
        "soft_temperature_limit",
    )
    runtime_errors = [row for row in rows if (row.get("runtime") or {}).get("error")]
    capture_counts = [
        (((row.get("runtime") or {}).get("tracking") or {}).get("metrics") or {}).get(
            "capture_failures"
        )
        for row in rows
    ]
    final_tracking = (
        summary.get("final_metrics", {}).get("tracking", {}).get("metrics", {})
    )
    tail_start = max(5, duration - 300)
    tail = [row for row in rows if row["elapsed_seconds"] >= tail_start]
    source_label = str(source)
    try:
        source_label = str(source.relative_to(Path.cwd()))
    except ValueError:
        pass
    result = {
        "source_telemetry": {
            "path": source_label + "/telemetry.jsonl",
            **fingerprint(telemetry),
            "committed": False,
        },
        "source_manifest": fingerprint(source / "manifest.json"),
        "source_summary": fingerprint(source / "summary.json"),
        "aggregation_script": {
            "filename": Path(__file__).name,
            **fingerprint(Path(__file__)),
        },
        "method": {
            "cpu": "Arithmetic mean of non-null outer hardware samples at elapsed_seconds >= 5; approximately 1 Hz, not a time-weighted integral.",
            "cpu_process_unit": "Percent of one logical core; 100 means one core.",
            "cpu_system_unit": "Percent of total system capacity across all logical cores; independently measured, not process CPU divided by cores.",
            "temperature_rss_throttling": "All telemetry rows including startup and sampled cleanup. RSS is current resident memory in MiB, not model size or ru_maxrss.",
            "fps_windows": "Fixed [n*60,(n+1)*60) second windows; unique runtime.updated_at stamps and render.frames counters. FPS=(last counter-first counter)/(last runtime timestamp-first runtime timestamp). Actual endpoints reported; no interpolation or invented counter at zero. Windows with <2 states or counter reset have null FPS.",
            "runtime_time_origin": "Median of outer timestamp-elapsed_seconds; used only to align runtime timestamps with telemetry windows.",
            "tail": "Last 300 seconds of observed telemetry (or from second 5 for shorter trials); includes sampled shutdown and is not labelled steady state.",
            "null": "Missing/nonfinite values excluded, never interpreted as zero.",
            "errors": "Counts are telemetry rows containing errors, not counts of independent incidents.",
            "final_frame_percentiles": "Final controller percentiles use at most the last 18000 draw/control intervals, not necessarily the whole endurance trial.",
        },
        "outcome": summary.get("outcome"),
        "exit_code": summary.get("exit_code"),
        "total_samples": len(rows),
        "malformed_rows": malformed,
        "observed_window": {
            "first_seconds": rows[0]["elapsed_seconds"],
            "last_seconds": duration,
        },
        "sample_spacing_seconds": {
            "median": median(intervals) if intervals else None,
            "maximum": max(intervals) if intervals else None,
            "gaps_over_2_5_seconds": sum(gap > 2.5 for gap in intervals),
        },
        "cpu_window": {
            "start_at_or_after_seconds": 5,
            "first_sample_seconds": cpu[0]["elapsed_seconds"] if cpu else None,
            "last_sample_seconds": cpu[-1]["elapsed_seconds"] if cpu else None,
            "samples": len(cpu),
        },
        "process_cpu_percent_one_core": stats(
            values(cpu, "process_cpu_percent_one_core")
        ),
        "system_cpu_percent": stats(values(cpu, "system_cpu_percent")),
        "temperature_c": stats(values(rows, "cpu_temperature_c")),
        "rss_mib": stats(values(rows, "process_rss_mib")),
        "process_peak_rss_mib": stats(values(rows, "process_peak_rss_mib")),
        "cpu_frequency_mhz": stats(values(rows, "cpu_frequency_mhz")),
        "throttling": {
            "samples": len(valid_throttles),
            "missing_samples": len(rows) - len(valid_throttles),
            "raw_values": sorted({value["raw"] for value in valid_throttles}),
            "now_true_samples": {
                name: sum(
                    value.get("now", {}).get(name) is True for value in valid_throttles
                )
                for name in flag_names
            },
            "since_boot_first": valid_throttles[0].get("since_boot")
            if valid_throttles
            else None,
            "since_boot_last": valid_throttles[-1].get("since_boot")
            if valid_throttles
            else None,
        },
        "sampler_errors": sum(bool(row.get("sampler_error")) for row in rows),
        "runtime_errors": len(runtime_errors),
        "tracking_error_samples": sum(
            bool(((row.get("runtime") or {}).get("tracking") or {}).get("error"))
            for row in rows
        ),
        "abort_reason_samples": sum(bool(row.get("abort_reason")) for row in rows),
        "capture_failure_counters": {
            "maximum_during_telemetry": max(
                (v for v in capture_counts if number(v)), default=None
            ),
            "after_cleanup": final_tracking.get("capture_failures"),
        },
        "runtime_snapshots": {
            "unique_valid": len(points),
            "repeated": repeats,
            "conflicting_timestamps_omitted": len(conflicts),
        },
        "final_300_seconds": {
            "requested_start_seconds": tail_start,
            "first_sample_seconds": tail[0]["elapsed_seconds"] if tail else None,
            "last_sample_seconds": tail[-1]["elapsed_seconds"] if tail else None,
            "system_cpu_percent": stats(values(tail, "system_cpu_percent")),
            "temperature_c": stats(values(tail, "cpu_temperature_c")),
            "rss_mib": stats(values(tail, "process_rss_mib")),
            "draw": fps_between([point for point in points if point[0] >= tail_start]),
        },
        "windows_60_seconds": windows,
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("manifest.json", "summary.json"):
        shutil.copyfile(source / filename, output / filename)
    (output / "telemetry-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    fields = [
        "interval_start_s",
        "interval_end_s_exclusive",
        "runtime_first_s",
        "runtime_last_s",
        "runtime_states",
        "draw_fps",
        "cpu_system_mean_percent",
        "cpu_process_mean_percent_one_core",
        "temperature_mean_c",
        "temperature_max_c",
        "rss_mean_mib",
        "rss_max_mib",
        "telemetry_samples",
    ]
    with (output / "windows-60s.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for window in windows:
            writer.writerow(
                dict(
                    zip(
                        fields,
                        [
                            window["start_seconds"],
                            window["end_seconds_exclusive"],
                            window["draw"].get("first_runtime_seconds"),
                            window["draw"].get("last_runtime_seconds"),
                            window["draw"]["samples"],
                            window["draw"]["fps"],
                            window["system_cpu_percent"]["mean"],
                            window["process_cpu_percent_one_core"]["mean"],
                            window["temperature_c"]["mean"],
                            window["temperature_c"]["maximum"],
                            window["rss_mib"]["mean"],
                            window["rss_mib"]["maximum"],
                            window["telemetry_samples"],
                        ],
                    )
                )
            )
    table = [
        "# Telemetría por ventanas de 60 segundos",
        "",
        "| Ventana nominal, s | FPS de dibujo | CPU sistema media, % | Temperatura media / máx., °C | RSS medio / máx., MiB | Muestras hardware / estados FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in windows:
        table.append(
            f"| {window['start_seconds']}–{window['end_seconds_exclusive']} | {spanish(window['draw']['fps'])} | "
            f"{spanish(window['system_cpu_percent']['mean'])} | {spanish(window['temperature_c']['mean'])} / {spanish(window['temperature_c']['maximum'])} | "
            f"{spanish(window['rss_mib']['mean'])} / {spanish(window['rss_mib']['maximum'])} | {window['telemetry_samples']} / {window['draw']['samples']} |"
        )
    table += [
        "",
        "Los intervalos son [inicio, fin). FPS usa las diferencias entre los estados reales primero y último de cada ventana; los extremos exactos están en el CSV y el JSON. No interpola bordes ni añade un contador inicial. Una ventana final parcial se conserva como tal.",
        "",
        "CPU: media aritmética de muestras válidas desde el segundo 5, porcentaje del sistema completo. Temperatura y RSS incluyen todas las muestras. Valores ausentes se omiten; «—» significa sin datos suficientes. El RSS es memoria residente del proceso en MiB. No son medidas de precisión ni latencia óptica.",
        "",
    ]
    (output / "windows-60s.md").write_text("\n".join(table))
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": len(rows),
                "windows": len(windows),
                "telemetry_sha256": result["source_telemetry"]["sha256"],
                "outcome": summary.get("outcome"),
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("new_output_directory", type=Path)
    args = parser.parse_args()
    aggregate(args.source_directory.resolve(), args.new_output_directory.resolve())


if __name__ == "__main__":
    main()
