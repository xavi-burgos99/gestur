"""Bounded tracker benchmark; this file is not a second tracking engine.

The previous callback-count benchmark is deprecated: callbacks also publish
expiry/status updates and do not correspond to model inferences. This command
uses the tracker's real capture/pose/hand counters and reports observed values
without ranking configurations or claiming recognition accuracy.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

from pose_detector import PoseHandTracker

PARTS = ('head', 'torso', 'left_hand', 'right_hand')


def timing_summary(values):
    if not values:
        return {'samples':0,'median':None,'p95':None,'maximum':None}
    ordered = sorted(values)
    return {'samples':len(ordered),'median':statistics.median(ordered),
            'p95':ordered[min(len(ordered)-1, math.ceil(.95*len(ordered))-1)],
            'maximum':ordered[-1]}


def peak_rss_mb():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024*1024 if sys.platform == 'darwin' else 1024)
    except (ImportError, AttributeError):
        return None


def summarize(metrics, elapsed, cpu_seconds, detections, sample_count,
              inference_times, frame_ages, *, startup_seconds, error=None,
              interrupted=False):
    elapsed = max(0.0, elapsed)
    counts = {name:int(metrics.get(name,0)) for name in
              ('captured_frames','pose_frames','hand_frames','capture_failures')}
    return {
        'status':'error' if error else 'interrupted' if interrupted else 'completed',
        'error':str(error) if error else None,
        'platform':{'system':platform.system(),'machine':platform.machine(),
                    'python':platform.python_version()},
        'startup_seconds':startup_seconds,
        'elapsed_seconds':elapsed,
        'counts':counts,
        'fps':{name:counts[field]/elapsed if elapsed else 0.0 for name,field in
               (('capture','captured_frames'),('pose','pose_frames'),('hands','hand_frames'))},
        'cpu_seconds':cpu_seconds,
        # 100% represents one fully utilized logical CPU, not the entire Pi.
        'cpu_percent_one_core':100*cpu_seconds/elapsed if elapsed else 0.0,
        'process_peak_rss_mb':peak_rss_mb(),
        'validity_samples':sample_count,
        'validity_sample_fraction':{part:detections[part]/sample_count if sample_count else None
                                    for part in PARTS},
        'sampled_inference_ms':timing_summary(inference_times),
        'sampled_frame_age_ms':timing_summary(frame_ages),
        'notes':[
            'FPS uses completed model/capture counters, never callback counts.',
            'Validity is a 20 Hz observation of detected flags, not recognition accuracy.',
            'Latency statistics sample the latest completed inference at 20 Hz; they are not every frame.',
            'Timing excludes model/camera startup and resource shutdown. Peak RSS includes startup.',
            'This command measures tracking only, without the 3D renderer.',
        ],
    }


def benchmark(tracker, seconds=30):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('La duración debe ser positiva y finita.')
    detections = dict.fromkeys(PARTS,0)
    samples = 0
    inference_times, frame_ages = [], []
    metrics = {}
    error = None
    interrupted = False
    start = time.monotonic()
    started = None
    cpu_start = None
    elapsed = cpu_seconds = 0.0
    startup_seconds = 0.0
    last_inference = (0,0)
    try:
        tracker.run()
        started = time.monotonic()
        cpu_start = time.process_time()
        startup_seconds = started-start
        deadline = started+seconds
        while time.monotonic() < deadline:
            metrics = tracker.get_metrics()
            if tracker.last_error is not None:
                error = tracker.last_error
                break
            if not tracker.running:
                error = RuntimeError('El tracker se detuvo antes de completar la medición.')
                break
            data = tracker.get_current_data()
            samples += 1
            for part in PARTS:
                detections[part] += bool(data[part]['detected'])
            completed = (metrics.get('pose_frames',0),metrics.get('hand_frames',0))
            if completed != last_inference:
                inference_times.append(metrics.get('inference_ms',0.0))
                frame_ages.append(metrics.get('frame_age_ms',0.0))
                last_inference = completed
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(min(.05,remaining))
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exception:
        error = exception
    finally:
        if started is not None:
            # Snapshot counters/time before stop; shutdown may finish an extra
            # inference, which is deliberately outside the measurement window.
            metrics = tracker.get_metrics()
            elapsed = time.monotonic()-started
            cpu_seconds = time.process_time()-cpu_start
        else:
            startup_seconds = time.monotonic()-start
        tracker.stop()
        if error is None and tracker.last_error is not None:
            error = tracker.last_error
    return summarize(metrics,elapsed,cpu_seconds,detections,samples,inference_times,frame_ages,
                     startup_seconds=startup_seconds,error=error,interrupted=interrupted)


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Debe ser un número positivo y finito.')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description='Medición acotada de captura e inferencia en la máquina actual.')
    parser.add_argument('--seconds',type=positive_float,default=30,help='Duración de medición; por defecto 30 segundos.')
    parser.add_argument('--hands',action='store_true',help='Añadir detección de manos al seguimiento de pose.')
    parser.add_argument('--camera',type=int,default=0,help='Índice de cámara OpenCV.')
    parser.add_argument('--output',type=Path,help='Guardar el informe JSON en esta ruta.')
    parser.add_argument('--inference-fps',type=positive_float,default=24)
    parser.add_argument('--hand-fps',type=positive_float,default=15)
    args = parser.parse_args(argv)
    tracker = PoseHandTracker(use_pose=True,use_hands=args.hands,mirror=True,
                              camera_index=args.camera,inference_fps=args.inference_fps,
                              hand_fps=args.hand_fps)
    report = benchmark(tracker,args.seconds)
    report['configuration'] = {'seconds':args.seconds,'hands':args.hands,'camera':args.camera,
                               'inference_fps':args.inference_fps,'hand_fps':args.hand_fps,
                               'width':640,'height':480,'mirror':True}
    print(f"Estado: {report['status']} | Tiempo medido: {report['elapsed_seconds']:.2f} s")
    print(f"FPS observados: cámara {report['fps']['capture']:.2f}, pose {report['fps']['pose']:.2f}, manos {report['fps']['hands']:.2f}")
    print(f"CPU: {report['cpu_percent_one_core']:.1f}% (100% = un núcleo)")
    print(f"Conteos: {json.dumps(report['counts'],ensure_ascii=False)}")
    print(f"Fracción de muestras con detección: {json.dumps(report['validity_sample_fraction'],ensure_ascii=False)}")
    print(f"Latencia muestreada (ms): {json.dumps(report['sampled_inference_ms'])}")
    if report['error']:
        print(f"Error: {report['error']}",file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(f'Informe: {args.output.resolve()}')
    return 1 if report['status'] == 'error' else 130 if report['status'] == 'interrupted' else 0


if __name__ == '__main__':
    raise SystemExit(main())
