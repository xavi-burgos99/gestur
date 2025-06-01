import time, math, threading, cv2, numpy as np, torch
from typing import List
from mmpose.apis import MMPoseInferencer

# ────────────────────────── PARÁMETROS ──────────────────────────────────────
OBJ_PATH = "teapot.obj"        # tu modelo
YAW_DEG, PITCH_DEG = 30, 40
VERBOSE = 0.5                  # s entre prints (0 = silencio)

# compartido entre hilos
head, lock = {'x': 0.0, 'y': 0.0}, threading.Lock()

# ─────────────────────────── HEAD-TRACK ─────────────────────────────────────
def face_ids(meta) -> List[int]:
    if not meta or 'keypoint_labels' not in meta:
        return [0, 1, 2]
    want = {'nose', 'left_eye', 'right_eye'}
    return [i for i, n in enumerate(meta['keypoint_labels']) if n in want] or [0,1,2]

def pose_thread():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    infer  = MMPoseInferencer(pose2d='body', device=device)
    cam    = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        raise RuntimeError('No webcam')

    W, H = int(cam.get(3)), int(cam.get(4))
    cx, cy = W/2, H/2
    ids, t0 = None, 0.0

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        for out in infer(frame):
            ppl = out['predictions'][0]
            if not ppl:
                continue
            k = np.asarray(ppl[0]['keypoints'])   # (J,2) o (J,3)
            if k.shape[1] == 3 and k[:,2].mean() < .25:
                continue
            if ids is None:
                ids = face_ids(out.get('dataset_meta', {}))
            face = k[ids]
            if face.shape[1] == 3 and face[:,2].mean() < .25:
                continue

            x, y = face[:,0].mean(), face[:,1].mean()
            hx, hy = (x-cx)/cx, (y-cy)/cy
            with lock:
                head['x'], head['y'] = float(np.clip(hx,-1,1)), float(np.clip(hy,-1,1))

            if VERBOSE and time.time()-t0 >= VERBOSE:
                t0=time.time(); print(f'Head X:{hx:+.2f}  Head Y:{hy:+.2f}')

        if cv2.waitKey(1)&0xFF in (ord('q'), ord('Q')):
            break
    cam.release()

threading.Thread(target=pose_thread, daemon=True).start()

# ──────────────────────────── PYRENDER ──────────────────────────────────────
import trimesh
import trimesh.transformations as tra
import pyrender

# malla + material PBR
mesh_tm = trimesh.load(OBJ_PATH)
material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.85, 0.87, 0.92, 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.35)
mesh = pyrender.Mesh.from_trimesh(mesh_tm, material=material, smooth=True)

scene = pyrender.Scene(bg_color=[0.13,0.13,0.17,1], ambient_light=[0.04]*3)

# luces direccionales con sombras
light_key  = pyrender.DirectionalLight(color=[1,1,1],   intensity=3.0)
light_fill = pyrender.DirectionalLight(color=[0.55,0.6,0.7], intensity=1.2)

key_pose  = tra.euler_matrix(math.radians(-45), math.radians(45), 0, 'sxyz')
fill_pose = tra.euler_matrix(math.radians(-10), math.radians(-60), 0, 'sxyz')

scene.add(light_key,  pose=key_pose)
scene.add(light_fill, pose=fill_pose)

# plano receptor de sombras
plane = trimesh.creation.box(extents=[10, .02, 10])
plane_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.85]*3+[1],
                metallicFactor=0.0, roughnessFactor=1.0)
scene.add(pyrender.Mesh.from_trimesh(plane, material=plane_mat, smooth=False),
          pose=np.eye(4))

# nodo del modelo (se actualizará cada frame)
model_pose = tra.compose_matrix(
                scale=[0.15]*3,
                translate=[0, 0.6, 0])
node_model = scene.add(mesh, pose=model_pose)

# cámara
cam = pyrender.PerspectiveCamera(yfov=np.radians(45))
cam_pose = tra.translation_matrix([0,1.5,4])
scene.add(cam, pose=cam_pose)

# viewer
viewer = pyrender.Viewer(scene, run_in_thread=True)   # sin extras “Raymond”

# ─────────────────────────– LOOP de animación ───────────────────────────────
while viewer.is_active:
    with lock:
        yaw   = -head['x'] * YAW_DEG
        pitch = head['y'] * PITCH_DEG

    # matriz compuesta: T · R_pitch · R_yaw · S
    T  = tra.translation_matrix([0,0.6,0])
    Rx = tra.rotation_matrix(math.radians(-pitch), [1,0,0])
    Ry = tra.rotation_matrix(math.radians( yaw ), [0,1,0])
    S  = tra.scale_matrix(0.25)
    scene.set_pose(node_model, pose=T @ Rx @ Ry @ S)

    time.sleep(1/60)     # ≈60 fps
