# controller.py
import sys
import argparse
from pose_detector import PoseHandTracker
from visualizer import ControlledObjViewer
from control_system import ControlSystem, create_default_control_system


class PoseController:
    """Controlador simple que actúa como puente entre detector de pose y sistema de control"""

    def __init__(self, obj_path, control_system=None, smoothing_config=None, verbose=False):
        """
        Args:
            obj_path: Ruta al archivo .obj
            control_system: Instancia de ControlSystem (si no se proporciona, usa el default)
            smoothing_config: Configuración de suavizado básico
            verbose: Mostrar información de debug
        """
        self.verbose = verbose
        self.smoothing_config = smoothing_config or self._default_smoothing_config()

        # Inicializar componentes
        self._init_pose_tracker()
        self._init_visualizer(obj_path)

        # Sistema de control (puede ser personalizado o default)
        self.control_system = control_system or create_default_control_system()

        if verbose:
            print("=== CONTROLADOR INICIADO ===")
            print(f"Objeto: {obj_path}")
            print(f"Smoothing: alpha={self.smoothing_config['alpha']}, decay={self.smoothing_config['decay_rate']}")
            print("Mapeos de control:")
            for mapping_info in self.control_system.get_mappings_info():
                print(f"  • {mapping_info}")
            print("=" * 30)

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
        """Debug básico del controlador"""
        head = input_data.get('head', {})
        left_hand = input_data.get('left_hand', {})
        right_hand = input_data.get('right_hand', {})

        detected_parts = []
        if head.get('detected'):
            detected_parts.append(f"cabeza({head.get('x', 0):.3f},{head.get('y', 0):.3f})")
        if left_hand.get('detected'):
            detected_parts.append(f"mano_izq({left_hand.get('x', 0):.3f},{left_hand.get('y', 0):.3f})")
        if right_hand.get('detected'):
            detected_parts.append(f"mano_der({right_hand.get('x', 0):.3f},{right_hand.get('y', 0):.3f})")

        print(f"Detectado: {', '.join(detected_parts) if detected_parts else 'nada'}")
        if output:
            print(f"Aplicado: {output}")
        print("-" * 40)

    def run(self):
        """Ejecuta el controlador"""
        try:
            print("Controlador iniciado. Presiona Ctrl+C para salir.")
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
    parser = argparse.ArgumentParser(description="Controlador de pose simple")
    parser.add_argument("obj", help="Ruta al archivo .obj")
    parser.add_argument("--control-preset", choices=["default", "gaming", "precise"],
                        default="default", help="Preset de sistema de control")
    parser.add_argument("--smoothing-alpha", type=float, default=0.3,
                        help="Factor de suavizado (0.1=suave, 0.9=reactivo)")
    parser.add_argument("--decay-rate", type=float, default=0.1,
                        help="Velocidad de decaimiento al centro")
    parser.add_argument("--verbose", "-v", action="store_true", help="Modo debug")

    args = parser.parse_args()

    # Configuración de suavizado
    smoothing_config = {
        'alpha': args.smoothing_alpha,
        'decay_rate': args.decay_rate,
        'center_x': 0.5,
        'center_y': 0.5
    }

    # Crear sistema de control según preset
    if args.control_preset == "default":
        control_system = create_default_control_system()
    elif args.control_preset == "gaming":
        control_system = create_gaming_control_system()
    elif args.control_preset == "precise":
        control_system = create_precise_control_system()

    # Crear y ejecutar controlador
    controller = PoseController(
        obj_path=args.obj,
        control_system=control_system,
        smoothing_config=smoothing_config,
        verbose=args.verbose
    )

    controller.run()


if __name__ == "__main__":
    main()
