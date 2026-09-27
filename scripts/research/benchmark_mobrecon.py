"""CPU ONNX smoke benchmark on the exact prepared input and PyTorch reference.

Run in a fresh process: no PyTorch, OpenMesh, SciPy or MANO runtime required.
This times one pre-cropped hand, not a complete two-hand tracking pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).parent / 'mobrecon_dsconv.onnx')
    parser.add_argument('--reference', type=Path, default=Path(__file__).parent / 'pytorch-reference.npz')
    parser.add_argument('--threads', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--allow-spinning', action='store_true',
                        help='Enable ORT active waiting for explicit comparison; default disabled.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.frames < 1 or args.warmup < 0:
        parser.error('Use positive threads/frames and non-negative warmup')
    if not args.model.is_file() or not args.reference.is_file():
        parser.error('Model and reference must be existing files')
    import numpy as np
    import onnxruntime as ort
    config = ort.SessionOptions()
    config.intra_op_num_threads = args.threads
    config.inter_op_num_threads = 1
    config.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    spin = '1' if args.allow_spinning else '0'
    config.add_session_config_entry('session.intra_op.allow_spinning', spin)
    config.add_session_config_entry('session.inter_op.allow_spinning', spin)
    session = ort.InferenceSession(str(args.model), config, providers=['CPUExecutionProvider'])
    reference = np.load(args.reference, allow_pickle=False)
    values = {'rgb_crop': reference['input']}
    wall, cpu = [], []
    for i in range(args.warmup + args.frames):
        t, c = time.perf_counter(), time.process_time()
        outputs = session.run(None, values)
        dt, dc = (time.perf_counter() - t)*1000, (time.process_time() - c)*1000
        if i >= args.warmup:
            wall.append(dt)
            cpu.append(dc)
    results = {}
    for name, output in zip(['mesh_meters', 'uv_normalized'], outputs):
        results[name] = {'shape': list(output.shape), 'finite': bool(np.isfinite(output).all()),
                         'max_abs_error_vs_pytorch': float(np.max(np.abs(output-reference[name])))}
    report = {'platform': platform.platform(), 'python': platform.python_version(), 'runtime': ort.__version__,
              'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'threads': args.threads, 'allow_spinning': args.allow_spinning,
              'frames': args.frames, 'warmup': args.warmup,
              'wall_median_ms': statistics.median(wall), 'wall_p95_ms': sorted(wall)[int((len(wall)-1)*.95)],
              'cpu_median_ms': statistics.median(cpu),
              'peak_rss_MB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024)/1e6,
              'outputs': results, 'providers': session.get_providers(),
              'scope': 'Static prepared single-hand crop. Inference only, without preprocessing or detection.',
              'reference': 'Agreement with PyTorch is export parity, not pose accuracy.'}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not all(x['finite'] and x['max_abs_error_vs_pytorch'] < 1e-4 for x in results.values()):
        raise SystemExit('Export parity failed')


if __name__ == '__main__':
    main()
