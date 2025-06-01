#!/usr/bin/env python3
# Head position (2-D)  +   placeholder 3-D lifter
# pip install mmpose mmdet opencv-python numpy

import cv2, time, torch, numpy as np
from mmpose.apis import MMPoseInferencer
from typing import List

###############################################################################
# 1.  Inferencers
###############################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pose2d = MMPoseInferencer(        # devuelve bbox y keypoints en pixeles
    pose2d='body',                # alias de RTMPose (COCO‐whole-body)
    device=device)

pose3d = MMPoseInferencer(        # seguirás teniéndolo cargado para después
    pose3d='human3d',
    device=device)

###############################################################################
# 2.  Utilidades
###############################################################################
def face_ids(meta) -> List[int]:
    """Índices de nariz + ojos basados en las etiquetas del modelo 2-D."""
    if not meta or 'keypoint_labels' not in meta:
        return [0, 1, 2]  # COCO
    names = meta['keypoint_labels']
    want = {'nose', 'left_eye', 'right_eye'}
    found = [i for i, n in enumerate(names) if n in want]
    return found or [0, 1, 2]

###############################################################################
# 3.  Webcam loop
###############################################################################
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError('No se pudo abrir la webcam.')

W, H = int(cap.get(3)), int(cap.get(4))
cx0, cy0 = W / 2.0, H / 2.0
PRINT_EVERY = 0.5
t0 = 0

print(f'Camara {W}×{H}px — Q para salir\n')

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('⚠️  Frame vacío'); break

        # ---------- 2-D pose ----------
        for out2d in pose2d(frame):
            people = out2d['predictions'][0]
            if not people:
                break  # nada detectado

            p = people[0]                     # primera persona
            bbox = p['bbox']                  # [x, y, w, h] ABSOLUTA
            kpts = np.asarray(p['keypoints']) # (J,3) píxeles

            ids = face_ids(out2d.get('dataset_meta', {}))
            face = kpts[ids]
            if face[:, 1].mean() < .25:       # confianza baja
                break

            x_c, y_c = face[:, :2].mean(axis=0)
            hx = np.clip((x_c - cx0) / cx0, -1, 1)
            hy = np.clip((y_c - cy0) / cy0, -1, 1)

            # ---------- prints temporizados ----------
            now = time.time()
            if now - t0 >= PRINT_EVERY:
                t0 = now
                print(f'Head X: {hx:+.2f}   Head Y: {hy:+.2f}')
                print(f'  bbox (xywh): {bbox}')
                print(f'  nariz/ojos: {face[:, :2].astype(int).tolist()}\n')

            # ---------- (opcional) 3-D lifter ----------
            # keypoints_2d = kpts[:, :2][None]  # shape (1, J, 2)
            # res3d = next(pose3d(keypoints_2d))
            # orientación = res3d['predictions'][0]['keypoints_3d']
            # ...

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
