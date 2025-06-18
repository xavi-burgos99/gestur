import time
import math
import numpy as np
import trimesh
import trimesh.transformations as tra
import pyrender
from typing import Dict


class Visualizer:
    def __init__(self, obj_path: str = "teapot.obj", yaw_degrees: float = 30, pitch_degrees: float = 40):
        self.obj_path = obj_path
        self.yaw_degrees = yaw_degrees
        self.pitch_degrees = pitch_degrees

        # Configuración de la escena
        self.scene = None
        self.model_node = None
        self.viewer = None

        # Nodos adicionales para manos (preparado para futuro)
        self.left_hand_node = None
        self.right_hand_node = None

        self._setup_scene()

    def _setup_scene(self):
        """Configura la escena 3D con modelo, luces y cámara."""
        # Cargar malla y material
        mesh_tm = trimesh.load(self.obj_path)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.85, 0.87, 0.92, 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.35
        )
        mesh = pyrender.Mesh.from_trimesh(mesh_tm, material=material, smooth=True)

        # Crear escena
        self.scene = pyrender.Scene(
            bg_color=[0.13, 0.13, 0.17, 1],
            ambient_light=[0.04, 0.04, 0.04]
        )

        # Configurar luces direccionales
        self._setup_lights()

        # Añadir plano receptor de sombras
        self._setup_ground_plane()

        # Añadir modelo principal (controlado por cabeza)
        model_pose = tra.compose_matrix(
            scale=[0.15, 0.15, 0.15],
            translate=[0, 0.6, 0]
        )
        self.model_node = self.scene.add(mesh, pose=model_pose)

        # Configurar cámara
        self._setup_camera()

    def _setup_lights(self):
        """Configura las luces de la escena."""
        # Luz principal
        light_key = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
        key_pose = tra.euler_matrix(
            math.radians(-45), math.radians(45), 0, 'sxyz'
        )
        self.scene.add(light_key, pose=key_pose)

        # Luz de relleno
        light_fill = pyrender.DirectionalLight(
            color=[0.55, 0.6, 0.7], intensity=1.2
        )
        fill_pose = tra.euler_matrix(
            math.radians(-10), math.radians(-60), 0, 'sxyz'
        )
        self.scene.add(light_fill, pose=fill_pose)

    def _setup_ground_plane(self):
        """Configura el plano receptor de sombras."""
        plane = trimesh.creation.box(extents=[10, 0.02, 10])
        plane_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.85, 0.85, 0.85, 1],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        self.scene.add(
            pyrender.Mesh.from_trimesh(plane, material=plane_material, smooth=False),
            pose=np.eye(4)
        )

    def _setup_camera(self):
        """Configura la cámara de la escena."""
        cam = pyrender.PerspectiveCamera(yfov=np.radians(45))
        cam_pose = tra.translation_matrix([0, 1.5, 4])
        self.scene.add(cam, pose=cam_pose)

    def update_pose(self, pose_data: Dict):
        """
        Actualiza la pose del modelo basado en los datos de detección.

        Args:
            pose_data: Diccionario con datos de pose en formato:
                      {'head': {'x': float, 'y': float},
                       'hands': {'left': {...}, 'right': {...}}}
        """
        head_data = pose_data.get('head', {'x': 0.0, 'y': 0.0})

        # Calcular rotaciones basadas en la posición de la cabeza
        yaw = -head_data['x'] * self.yaw_degrees
        pitch = head_data['y'] * self.pitch_degrees

        # Crear matriz de transformación compuesta: T · R_pitch · R_yaw · S
        T = tra.translation_matrix([0, 0.6, 0])
        Rx = tra.rotation_matrix(math.radians(-pitch), [1, 0, 0])
        Ry = tra.rotation_matrix(math.radians(yaw), [0, 1, 0])
        S = tra.scale_matrix(0.25)

        # Aplicar transformación al modelo
        final_pose = T @ Rx @ Ry @ S
        self.scene.set_pose(self.model_node, pose=final_pose)

        # TODO: Implementar visualización de manos en el futuro
        # hands_data = pose_data.get('hands', {})
        # self._update_hands_visualization(hands_data)

    def start_viewer(self):
        """Inicia el visualizador."""
        self.viewer = pyrender.Viewer(self.scene, run_in_thread=True)

    def is_active(self) -> bool:
        """Verifica si el visualizador está activo."""
        return self.viewer and self.viewer.is_active

    def close(self):
        """Cierra el visualizador."""
        if self.viewer:
            self.viewer.close_external()

# Función principal de demostración
def main():
    """Función principal que combina detector y visualizador."""
    from pose_detector import PoseDetector

    # Configuración
    OBJ_PATH = "teapot.obj"
    YAW_DEG, PITCH_DEG = 30, 40

    # Inicializar componentes
    detector = PoseDetector(verbose=0.5)
    visualizer = Visualizer(
        obj_path=OBJ_PATH,
        yaw_degrees=YAW_DEG,
        pitch_degrees=PITCH_DEG
    )

    try:
        # Iniciar detección y visualización
        detector.start()
        visualizer.start_viewer()

        # Loop principal de actualización
        while visualizer.is_active():
            pose_data = detector.get_pose_data()
            visualizer.update_pose(pose_data)
            time.sleep(1 / 60)  # ~60 FPS

    except KeyboardInterrupt:
        print("Interrumpido por el usuario")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Limpiar recursos
        detector.stop()
        visualizer.close()


if __name__ == "__main__":
    main()
