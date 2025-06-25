# controller.py (modificado)
import sys
import argparse
from pose_detector import PoseHandTracker
from visualizer import ControlledObjViewer
from control_system import (ControlSystem, create_default_control_system)


class PoseController:
    """Controlador simple que actúa como puente entre detector de pose y sistema de control"""

    def __init__(self, obj_path, control_system=None, smoothing_config=None, verbose=False):
        self.verbose = verbose
        self.smoothing_config = smoothing_config or self._default_smoothing_config()

        # Inicializar componentes
        self._init_pose_tracker()
        self._init_visualizer(obj_path)

        # Sistema de control (puede ser personalizado o default)
        self.control_system = control_system or create_default_control_system()

        if verbose:
            print("=== CONTROLADOR CON ROTACIÓN CONTINUA ===")
            print(f"Objeto: {obj_path}")
            print("Mapeos de control:")
            for mapping_info in self.control_system.get_mappings_info():
                print(f"  • {mapping_info}")
            print("=" * 40)

    def _default_smoothing_config(self):
        """Configuración básica de suavizado"""
        return {
            'alpha': 0.3,
            'decay_rate': 0.1,
            'center_x': 0.5,
            'center_y': 0.5
        }

    def _init_pose_tracker(self):
        """Inicializa el detector de pose con configuración mínima"""
        print("Iniciando detector de pose...")
        self.pose_tracker = PoseHandTracker(
            response_time_ms=50,
            smoothing_time_ms=50,  # Smoothing mínimo en el detector
            use_pose=True,
            use_hands=True,
            mirror=True,
            verbose=self.verbose
        )
        self.pose_tracker.subscribe(self._on_pose_update)

    def _init_visualizer(self, obj_path):
        """Inicializa el visualizador"""
        print("Iniciando visualizador 3D...")
        self.visualizer = ControlledObjViewer(obj_path)

    def _on_pose_update(self, pose_data):
        """
        Callback simple que pasa datos al sistema de control
        El controlador NO hace ningún procesamiento de mapeo
        """
        # Pasar datos directamente al sistema de control
        output = self.control_system.process_input(pose_data)

        # Aplicar resultado al visualizador
        if output:
            self.visualizer.update_model(**output)

        if self.verbose:
            self._print_debug_info(pose_data, output)

    def _print_debug_info(self, input_data, output):
        """Debug mejorado del controlador"""
        head = input_data.get('head', {})

        if head.get('detected'):
            head_x = head.get('x', 0)

            # Determinar zona y estado
            if head_x > 0.75:
                status = " [🔄 ROTANDO DERECHA]"
            elif head_x < 0.25:
                status = " [🔄 ROTANDO IZQUIERDA]"
            else:
                status = " [⏸️  ZONA NEUTRA]"

            print(f"Cabeza X: {head_x:.3f}{status}")

        if output:
            rotation_yaw = output.get('rotation', [0, 0, 0])[0]
            print(f"Rotación Yaw: {rotation_yaw:.1f}°")

        print("-" * 50)

    def run(self):
        """Ejecuta el controlador"""
        try:
            print("🔄 CONTROLADOR CON ROTACIÓN CONTINUA iniciado")
            print("=" * 50)
            print("FUNCIONAMIENTO:")
            print("• Zona neutra (25%-75%): Sin rotación")
            print("• Cabeza > 75%: ROTACIÓN CONTINUA hacia la derecha")
            print("• Cabeza < 25%: ROTACIÓN CONTINUA hacia la izquierda")
            print("• El objeto rota indefinidamente mientras estés en extremos")
            print("=" * 50)
            print("Presiona Ctrl+C para salir.")

            self.pose_tracker.run()
            self.visualizer.run()
        except KeyboardInterrupt:
            print("\nDeteniendo sistema...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Limpia recursos"""
        if hasattr(self, 'pose_tracker'):
            self.pose_tracker.stop()


def main():
    parser = argparse.ArgumentParser(description="Controlador con rotación continua en extremos")
    parser.add_argument("obj", help="Ruta al archivo .obj")
    parser.add_argument("--control-preset",
                        choices=["default", "continuous"],
                        default="continuous", help="Preset de sistema de control")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo debug")

    args = parser.parse_args()

    # Crear sistema de control según preset
    control_system = create_default_control_system()

    # Crear y ejecutar controlador
    controller = PoseController(
        obj_path=args.obj,
        control_system=control_system,
        verbose=args.verbose
    )

    controller.run()


if __name__ == "__main__":
    main()
