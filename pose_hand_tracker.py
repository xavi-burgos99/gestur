import time
import statistics
import threading
from collections import defaultdict
import csv
import json
from pose_detector import PoseHandTracker


class PerformanceBenchmark:
    def __init__(self, target_frames=500):
        self.target_frames = target_frames
        self.results = []

        # Configuraciones a probar
        self.test_configurations = [
            # Solo pose - diferentes tiempos de respuesta
            {"name": "pose_only_fast", "use_pose": True, "use_hands": False,
             "response_time_ms": 16, "smoothing_time_ms": 100},
            {"name": "pose_only_medium", "use_pose": True, "use_hands": False,
             "response_time_ms": 33, "smoothing_time_ms": 200},
            {"name": "pose_only_slow", "use_pose": True, "use_hands": False,
             "response_time_ms": 50, "smoothing_time_ms": 500},

            # Solo manos - diferentes configuraciones
            {"name": "hands_only_fast", "use_pose": False, "use_hands": True,
             "response_time_ms": 16, "smoothing_time_ms": 100},
            {"name": "hands_only_medium", "use_pose": False, "use_hands": True,
             "response_time_ms": 33, "smoothing_time_ms": 200},

            # Pose + manos - configuraciones combinadas
            {"name": "full_tracking_fast", "use_pose": True, "use_hands": True,
             "response_time_ms": 16, "smoothing_time_ms": 100},
            {"name": "full_tracking_medium", "use_pose": True, "use_hands": True,
             "response_time_ms": 33, "smoothing_time_ms": 200},
            {"name": "full_tracking_slow", "use_pose": True, "use_hands": True,
             "response_time_ms": 50, "smoothing_time_ms": 500},

            # Sin suavizado
            {"name": "no_smoothing_pose", "use_pose": True, "use_hands": False,
             "response_time_ms": 33, "smoothing_time_ms": 0},
            {"name": "no_smoothing_full", "use_pose": True, "use_hands": True,
             "response_time_ms": 33, "smoothing_time_ms": 0},
        ]

    def run_benchmark(self):
        """Ejecuta el benchmark completo para todas las configuraciones"""
        print(f"🚀 Iniciando benchmark de rendimiento para {self.target_frames} frames")
        print(f"📊 Configuraciones a probar: {len(self.test_configurations)}")
        print("-" * 80)

        for i, config in enumerate(self.test_configurations, 1):
            print(f"\n[{i}/{len(self.test_configurations)}] Probando configuración: {config['name']}")
            result = self._test_configuration(config)
            self.results.append(result)

            # Mostrar resumen rápido
            print(f"   ✅ FPS promedio: {result['fps_stats']['mean']:.2f}")
            print(f"   ⏱️  Duración total: {result['total_duration']:.2f}s")
            print(f"   🎯 Frames procesados: {result['frames_processed']}")

        self._generate_report()

    def _test_configuration(self, config):
        """Prueba una configuración específica y recopila métricas"""

        # Configurar parámetros adicionales por defecto
        full_config = {
            "response_time_ms": 50,
            "smoothing_time_ms": 500,
            "use_pose": True,
            "use_hands": True,
            "mirror": True,
            "invert_hands": False,
            "verbose": False,
            **config  # Sobrescribe con los parámetros específicos
        }

        # Métricas de rendimiento
        metrics = {
            "frame_times": [],
            "processing_times": [],
            "detection_counts": {"head": 0, "torso": 0, "left_hand": 0, "right_hand": 0},
            "callback_intervals": [],
            "frames_processed": 0
        }

        # Callback para medir rendimiento
        def performance_callback(data):
            current_time = time.time()

            # Contar detecciones
            for part in ["head", "torso", "left_hand", "right_hand"]:
                if data[part]["detected"]:
                    metrics["detection_counts"][part] += 1

            # Medir intervalo entre callbacks
            if hasattr(performance_callback, 'last_time'):
                interval = current_time - performance_callback.last_time
                metrics["callback_intervals"].append(interval)

            performance_callback.last_time = current_time
            metrics["frames_processed"] += 1

        try:
            # Crear tracker con la configuración
            tracker = PoseHandTracker(**{k: v for k, v in full_config.items() if k != 'name'})
            tracker.subscribe(performance_callback)

            # Iniciar medición
            start_attention = time.time()
            tracker.run()

            # Esperar hasta alcanzar el número objetivo de frames
            while metrics["frames_processed"] < self.target_frames:
                time.sleep(0.1)  # Pequeña pausa para no saturar CPU

                # Timeout de seguridad (máximo 2 minutos)
                if time.time() - start_attention > 120:
                    print(f"   ⚠️  Timeout alcanzado para {config['name']}")
                    break

            end_time = time.time()
            total_duration = end_time - start_attention

            # Detener tracker
            tracker.stop()

            # Calcular estadísticas
            fps_values = [1.0 / interval for interval in metrics["callback_intervals"] if interval > 0]

            result = {
                "config_name": config["name"],
                "config_params": full_config,
                "total_duration": total_duration,
                "frames_processed": metrics["frames_processed"],
                "target_fps": 1000.0 / full_config["response_time_ms"],
                "fps_stats": self._calculate_stats(fps_values),
                "interval_stats": self._calculate_stats(metrics["callback_intervals"]),
                "detection_stats": metrics["detection_counts"],
                "detection_rates": {
                    part: count / metrics["frames_processed"] * 100
                    for part, count in metrics["detection_counts"].items()
                } if metrics["frames_processed"] > 0 else {}
            }

            return result

        except Exception as e:
            print(f"   ❌ Error en configuración {config['name']}: {e}")
            return {
                "config_name": config["name"],
                "config_params": full_config,
                "error": str(e),
                "total_duration": 0,
                "frames_processed": 0
            }

    def _calculate_stats(self, values):
        """Calcula estadísticas detalladas de una lista de valores"""
        if not values:
            return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0,
                    "median": 0, "p95": 0, "p99": 0}

        sorted_values = sorted(values)
        count = len(values)

        return {
            "count": count,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if count > 1 else 0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(0.95 * count)] if count > 0 else 0,
            "p99": sorted_values[int(0.99 * count)] if count > 0 else 0
        }

    def _generate_report(self):
        """Genera un reporte completo de los resultados"""
        print("\n" + "=" * 80)
        print("📊 REPORTE COMPLETO DE RENDIMIENTO")
        print("=" * 80)

        # Tabla resumen
        print(
            f"\n{'CONFIGURACIÓN':<20} {'FPS PROM':<10} {'FPS STD':<10} {'DURACIÓN':<10} {'FRAMES':<8} {'DETECCIONES %'}")
        print("-" * 80)

        for result in self.results:
            if 'error' in result:
                print(f"{result['config_name']:<20} {'ERROR':<10}")
                continue

            fps_mean = result['fps_stats']['mean']
            fps_std = result['fps_stats']['std']
            duration = result['total_duration']
            frames = result['frames_processed']

            # Promedio de detecciones
            det_avg = statistics.mean(result['detection_rates'].values()) if result['detection_rates'] else 0

            print(f"{result['config_name']:<20} {fps_mean:<10.2f} {fps_std:<10.2f} "
                  f"{duration:<10.2f} {frames:<8d} {det_avg:<12.1f}")

        # Detalles por configuración
        print("\n" + "=" * 80)
        print("📈 DETALLES POR CONFIGURACIÓN")
        print("=" * 80)

        for result in self.results:
            if 'error' in result:
                continue

            print(f"\n🔧 **{result['config_name'].upper()}**")
            print(f"   Parámetros: response_time={result['config_params']['response_time_ms']}ms, "
                  f"smoothing={result['config_params']['smoothing_time_ms']}ms")
            print(f"   Pose: {result['config_params']['use_pose']}, "
                  f"Manos: {result['config_params']['use_hands']}")

            fps = result['fps_stats']
            print(f"   📊 FPS: μ={fps['mean']:.2f} σ={fps['std']:.2f} "
                  f"min={fps['min']:.2f} max={fps['max']:.2f} p95={fps['p95']:.2f}")

            print(f"   ⏱️  Duración total: {result['total_duration']:.2f}s "
                  f"({result['frames_processed']} frames)")

            # Tasas de detección
            print("   🎯 Tasas de detección:")
            for part, rate in result['detection_rates'].items():
                print(f"      {part}: {rate:.1f}%")

        # Guardar resultados
        self._save_results()

        # Mejores configuraciones
        self._show_best_configurations()

    def _save_results(self):
        """Guarda los resultados en archivos CSV y JSON"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Guardar en JSON
        json_filename = f"benchmark_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Guardar en CSV
        csv_filename = f"benchmark_summary_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Headers
            writer.writerow([
                'Config Name', 'Use Pose', 'Use Hands', 'Response Time (ms)',
                'Smoothing Time (ms)', 'Total Duration (s)', 'Frames Processed',
                'FPS Mean', 'FPS Std', 'FPS Min', 'FPS Max', 'FPS P95',
                'Head Detection %', 'Torso Detection %', 'Left Hand Detection %', 'Right Hand Detection %'
            ])

            # Data rows
            for result in self.results:
                if 'error' in result:
                    continue

                writer.writerow([
                    result['config_name'],
                    result['config_params']['use_pose'],
                    result['config_params']['use_hands'],
                    result['config_params']['response_time_ms'],
                    result['config_params']['smoothing_time_ms'],
                    result['total_duration'],
                    result['frames_processed'],
                    result['fps_stats']['mean'],
                    result['fps_stats']['std'],
                    result['fps_stats']['min'],
                    result['fps_stats']['max'],
                    result['fps_stats']['p95'],
                    result['detection_rates'].get('head', 0),
                    result['detection_rates'].get('torso', 0),
                    result['detection_rates'].get('left_hand', 0),
                    result['detection_rates'].get('right_hand', 0)
                ])

        print(f"\n💾 Resultados guardados en:")
        print(f"   📄 {json_filename}")
        print(f"   📊 {csv_filename}")

    def _show_best_configurations(self):
        """Muestra las mejores configuraciones según diferentes criterios"""
        valid_results = [r for r in self.results if 'error' not in r and r['frames_processed'] > 0]

        if not valid_results:
            return

        print("\n" + "=" * 80)
        print("🏆 MEJORES CONFIGURACIONES")
        print("=" * 80)

        # Mayor FPS promedio
        best_fps = max(valid_results, key=lambda x: x['fps_stats']['mean'])
        print(f"\n🚀 **Mayor FPS promedio:** {best_fps['config_name']}")
        print(f"   FPS: {best_fps['fps_stats']['mean']:.2f} ± {best_fps['fps_stats']['std']:.2f}")

        # Menor variabilidad en FPS
        best_stability = min(valid_results, key=lambda x: x['fps_stats']['std'])
        print(f"\n📈 **Mayor estabilidad (menor desviación):** {best_stability['config_name']}")
        print(f"   FPS: {best_stability['fps_stats']['mean']:.2f} ± {best_stability['fps_stats']['std']:.2f}")

        # Mayor tasa de detección promedio
        best_detection = max(valid_results, key=lambda x: statistics.mean(x['detection_rates'].values()) if x[
            'detection_rates'] else 0)
        avg_detection = statistics.mean(best_detection['detection_rates'].values()) if best_detection[
            'detection_rates'] else 0
        print(f"\n🎯 **Mayor tasa de detección:** {best_detection['config_name']}")
        print(f"   Detección promedio: {avg_detection:.1f}%")

        # Mejor balance (FPS alto + detección alta)
        def balance_score(result):
            fps_score = result['fps_stats']['mean'] / 60.0  # Normalizar a 60 FPS
            det_score = statistics.mean(result['detection_rates'].values()) / 100.0 if result['detection_rates'] else 0
            return fps_score * det_score

        best_balance = max(valid_results, key=balance_score)
        print(f"\n⚖️  **Mejor balance FPS/Detección:** {best_balance['config_name']}")
        print(f"   FPS: {best_balance['fps_stats']['mean']:.2f}, "
              f"Detección: {statistics.mean(best_balance['detection_rates'].values()):.1f}%")


def main():
    """Función principal para ejecutar el benchmark"""
    print("🎯 Benchmark de Rendimiento - PoseHandTracker")
    print("=" * 50)

    # Crear y ejecutar benchmark
    benchmark = PerformanceBenchmark(target_frames=500)

    try:
        benchmark.run_benchmark()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el benchmark: {e}")

    print(f"\n✅ Benchmark completado!")


if __name__ == "__main__":
    main()
