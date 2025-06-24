# visualizer.py
from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, AmbientLight, Vec3, Filename, getModelPath, loadPrcFileData
import sys, os

loadPrcFileData('', 'load-file-type p3assimp')

if sys.platform == 'darwin':
    loadPrcFileData('', 'win-size 1920 1080')
    loadPrcFileData('', 'fullscreen-windowed true')
else:
    loadPrcFileData('', 'fullscreen true')


class ControlledObjViewer(ShowBase):
    def __init__(self, obj_path):
        super().__init__()

        # --- Configurar ruta de búsqueda de texturas ---
        obj_dir = os.path.dirname(os.path.abspath(obj_path))
        getModelPath().appendPath(obj_dir)

        # --- Escena base ---
        self.disableMouse()
        self.cam.set_pos(0, 1, -25)
        self.cam.look_at(0, 0, 0)

        # --- Cargar modelo con materiales ---
        print(f"Cargando: {obj_path}")
        self.model = self.loader.loadModel(Filename.from_os_specific(obj_path))

        if self.model:
            self.model.reparent_to(self.render)
            self.model.set_scale(1)
            self.model.set_pos(0, 0, 0)
            self.model.set_hpr(0, 0, 0)
            self.model.setDepthOffset(1)
            print("Texturas detectadas:", self.model.find_all_textures())
            print("Modelo cargado exitosamente")
        else:
            print("Error: No se pudo cargar el modelo")
            sys.exit(1)

        # Activar shaders automáticos para texturas
        self.render.set_shader_auto()

        # Estado actual del modelo
        self.current_state = {
            'position': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0],  # [yaw, pitch, roll]
            'scale': [1.0, 1.0, 1.0]
        }

    def update_model(self, **kwargs):
        """
        Actualiza múltiples propiedades del modelo de una vez

        Args:
            position: [x, y, z] posición del modelo
            rotation: [yaw, pitch, roll] rotación en grados
            scale: [sx, sy, sz] escala del modelo o float para escala uniforme
        """
        if not self.model:
            return

        # Actualizar posición
        if 'position' in kwargs:
            pos = kwargs['position']
            if len(pos) == 3:
                self.current_state['position'] = list(pos)
                self.model.set_pos(*pos)

        # Actualizar rotación
        if 'rotation' in kwargs:
            rot = kwargs['rotation']
            if len(rot) == 3:
                self.current_state['rotation'] = list(rot)
                self.model.set_hpr(*rot)  # yaw, pitch, roll

        # Actualizar escala
        if 'scale' in kwargs:
            scale = kwargs['scale']
            if isinstance(scale, (int, float)):
                # Escala uniforme
                self.current_state['scale'] = [scale, scale, scale]
                self.model.set_scale(scale)
            elif len(scale) == 3:
                # Escala por componente
                self.current_state['scale'] = list(scale)
                self.model.set_scale(*scale)

    def get_current_state(self):
        """Retorna el estado actual del modelo"""
        return self.current_state.copy()

    # Métodos de compatibilidad con el código anterior
    def set_model_rotation_limited(self, pitch=0, yaw=0, roll=0):
        """Método de compatibilidad"""
        self.update_model(rotation=[yaw, pitch, roll])

    def set_model_position(self, x=0, y=0, z=0):
        """Método de compatibilidad"""
        self.update_model(position=[x, y, z])

    def set_model_scale(self, scale=1):
        """Método para escala"""
        self.update_model(scale=scale)
