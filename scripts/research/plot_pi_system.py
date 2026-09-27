#!/usr/bin/env python3
"""Plot an existing GESTUR trial; never collect data or modify the input.

Usage: python scripts/research/plot_pi_system.py INPUT_DIRECTORY OUTPUT.png
FPS: differences of real draw counters and runtime updated_at timestamps within
60-second windows (10 seconds for trials shorter than two minutes). Repeated
runtime snapshots are deduplicated. No interpolation, invented starting value,
or cumulative render_fps field is used. Incomplete windows use actual endpoints.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    rows = []
    skipped = 0
    with (directory / 'telemetry.jsonl').open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(row, dict) and number(row.get('elapsed_seconds')):
                rows.append(row)
    rows.sort(key=lambda row: row['elapsed_seconds'])
    if not rows:
        raise ValueError('No hay muestras de telemetría válidas.')
    return manifest, summary, rows, skipped


def window_fps(rows, window):
    # Use the sampler's observed wall-clock/monotonic correspondence, without
    # inventing an initial draw counter. Then use only runtime timestamps.
    offsets = [row['timestamp'] - row['elapsed_seconds'] for row in rows if number(row.get('timestamp'))]
    if not offsets:
        return []
    origin = median(offsets)
    unique = {}
    for row in rows:
        runtime = row.get('runtime')
        if not isinstance(runtime, dict):
            continue
        stamp = runtime.get('updated_at')
        render = runtime.get('render') or {}
        frames = render.get('frames')
        if number(stamp) and number(frames) and frames >= 0 and stamp >= origin:
            unique[stamp] = frames
    groups = {}
    for stamp, count in sorted(unique.items()):
        elapsed = stamp - origin
        groups.setdefault(math.floor(elapsed / window), []).append((elapsed, count))
    points = []
    for values in groups.values():
        if len(values) < 2 or any(right[1] < left[1] for left, right in zip(values, values[1:])):
            continue
        start, first = values[0]
        end, last = values[-1]
        if end > start:
            points.append({'start_seconds': start, 'end_seconds': end,
                           'fps': (last - first) / (end - start),
                           'drawn_frames': last - first, 'runtime_snapshots': len(values)})
    return points


def spanish(value, digits=1):
    return f'{value:,.{digits}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def hardware_series(rows, key, divisor):
    # NaNs leave genuine gaps in the line; no forward-fill or fabricated zeros.
    x, y = [], []
    for row in rows:
        x.append(row['elapsed_seconds'] / divisor)
        value = (row.get('hardware') or {}).get(key)
        y.append(value if number(value) else float('nan'))
    return x, y


def describe_motion(manifest):
    source = 'cámara USB' if manifest.get('source') == 'camera' else 'imágenes locales repetidas'
    motion = 'movimiento por gestos' if manifest.get('motion') == 'controls' else 'rotación continua impuesta'
    return f'{source} · {motion}'


def plot(directory, output):
    manifest, summary, rows, skipped = load(directory)
    duration = max(row['elapsed_seconds'] for row in rows)
    window = 10 if duration < 120 else 60
    fps = window_fps(rows, window)
    divisor = 60 if duration >= 120 else 1
    unit = 'min' if divisor == 60 else 's'
    colors = ['#b24e33', '#31698a', '#337963', '#78638e']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelcolor': '#253444', 'text.color': '#253444',
                         'xtick.color': '#526071', 'ytick.color': '#526071',
                         'axes.edgecolor': '#d0d8df', 'axes.linewidth': .8,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'hspace': .28})
    fig.subplots_adjust(left=.105, right=.97, top=.81, bottom=.16)
    model = manifest.get('model') or {}
    triangles = model.get('triangles', manifest.get('triangles'))
    framebuffer = manifest.get('framebuffer', [])
    resolution = ' × '.join(str(value) for value in framebuffer) or 'resolución no registrada'
    asset_name = Path(model.get('path', 'modelo')).stem
    outcome = {'completed': 'finalizado', 'aborted': 'detenido por protección',
               'failed': 'fallido', 'interrupted': 'interrumpido',
               'incomplete_inference': 'inferencia incompleta'}.get(summary.get('outcome'), 'sin cierre registrado')
    geometry = f'{spanish(triangles, 0)} triángulos' if number(triangles) else 'geometría no registrada'
    fig.text(.105, .955, 'GESTUR · Ensayo en Raspberry Pi 5', fontsize=20, weight='bold')
    fig.text(.105, .917, f'{asset_name} · {geometry} · {resolution} · MSAA real {manifest.get("msaa_actual", "s/d")}',
             fontsize=11)
    fig.text(.105, .888, describe_motion(manifest), fontsize=11)
    fig.text(.105, .858, f'Telemetría observada: {spanish(duration / 60, 2)} min · {len(rows)} muestras · Estado: {outcome}',
             fontsize=10, color='#526071')
    specs = [('cpu_temperature_c', 'Temperatura CPU (°C)'),
             ('system_cpu_percent', 'CPU total del sistema (%)'),
             (None, 'Dibujo real (fotogramas/s)'),
             ('process_rss_mib', 'Memoria residente (MiB)')]
    for index, (ax, (key, label), color) in enumerate(zip(axes, specs, colors)):
        ax.grid(axis='y', color='#e3e8ed', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(label, loc='left', fontsize=11, weight='bold', pad=7)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: spanish(value, 0)))
        ax.tick_params(axis='both', length=3)
        if key:
            x, y = hardware_series(rows, key, divisor)
            ax.plot(x, y, color=color, linewidth=1.25, marker='.', markersize=2.5)
            valid = [value for value in y if number(value)]
            if not valid:
                ax.text(.5, .5, 'No hay muestras disponibles', transform=ax.transAxes, ha='center')
            else:
                ax.text(.995, .96, f'mín. {spanish(min(valid))} · máx. {spanish(max(valid))}',
                        transform=ax.transAxes, ha='right', va='top', fontsize=9, color=color,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 2})
                span = max(valid) - min(valid)
                padding = max(span * .17, 2 if index == 0 else 5)
                ax.set_ylim(min(valid) - padding, max(valid) + padding)
            if index == 1:
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
        elif fps:
            centers = [(point['start_seconds'] + point['end_seconds']) / 2 / divisor for point in fps]
            widths = [(point['end_seconds'] - point['start_seconds']) / 2 / divisor for point in fps]
            ax.errorbar(centers, [point['fps'] for point in fps], xerr=widths,
                        color=color, fmt='o', markersize=4, elinewidth=1.5, capsize=2,
                        label=f'Ventanas de {window} s · extremos realmente observados')
            ax.legend(loc='upper right', frameon=False, fontsize=8.5)
            ax.set_ylim(0, max(75, max(point['fps'] for point in fps) * 1.25))
            ax.set_yticks([0, 15, 30, 45, 60, 75])
        else:
            ax.text(.5, .5, 'No hay dos contadores válidos dentro de una ventana',
                    transform=ax.transAxes, ha='center')
    axes[-1].set_xlabel(f'Tiempo desde el inicio de la telemetría ({unit})', labelpad=9)
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=8))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: spanish(value, 0)))
    axes[-1].set_xlim(min(0, rows[0]['elapsed_seconds'] / divisor), max(duration / divisor, 1))
    fig.text(.105, .093, f'FPS = Δfotogramas dibujados / Δtiempo del estado del visor; ventanas de {window} s, bordes parciales.',
             fontsize=9, color='#526071')
    fig.text(.105, .073, 'CPU: ocupación conjunta de todos los núcleos (0–100%). RSS: proceso GESTUR; incluye carga y cierre.',
             fontsize=9, color='#526071')
    fig.text(.105, .053, 'Se omiten datos ausentes y estados repetidos. No mide precisión ni latencia óptica.',
             fontsize=9, color='#526071')
    if skipped:
        fig.text(.105, .032, f'Líneas JSON incompletas omitidas: {skipped}.', fontsize=9, color='#526071')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, metadata={'Description': __doc__, 'Source': str(directory)})
    plt.close(fig)
    print(json.dumps({'output': str(output), 'input': str(directory), 'telemetry_samples': len(rows),
                      'duration_observed_seconds': duration, 'window_seconds': window,
                      'fps_windows': fps, 'skipped_json_lines': skipped}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_directory', type=Path)
    parser.add_argument('output_png', type=Path)
    args = parser.parse_args()
    if not args.input_directory.is_dir():
        parser.error('El directorio de entrada no existe.')
    if args.output_png.suffix.lower() != '.png':
        parser.error('La salida debe ser un archivo PNG.')
    plot(args.input_directory.resolve(), args.output_png.resolve())


if __name__ == '__main__':
    main()
