from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, AmbientLight, Vec3, Filename, getModelPath, loadPrcFileData
import sys, argparse, os

loadPrcFileData('', 'load-file-type p3assimp')
if sys.platform == 'darwin':
    loadPrcFileData('', 'win-size 1920 1080')
    loadPrcFileData('', 'fullscreen-windowed true')
else:
    loadPrcFileData('', 'fullscreen true')

class ObjViewer(ShowBase):
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
        model = self.loader.loadModel(Filename.from_os_specific(obj_path))

        if model:
            model.reparent_to(self.render)
            model.set_scale(1)
            model.set_pos(0, 0, 0)
            model.set_hpr(0, 0, 0)
            model.setDepthOffset(1)
            print("Texturas detectadas:", model.find_all_textures())
            print("Modelo cargado exitosamente")
        else:
            print("Error: No se pudo cargar el modelo")
            sys.exit(1)

        # Activar shaders automáticos para texturas
        self.render.set_shader_auto()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador .obj con materiales")
    parser.add_argument("obj", help="Ruta al archivo .obj")
    args = parser.parse_args()

    app = ObjViewer(args.obj)
    app.run()
