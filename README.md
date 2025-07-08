# Gestur: Control de interfaces 3D por gestos y pose
  
Sistema de interacción con entornos digitales mediante visión artificial, eliminando la necesidad de hardware especializado para exploración inmersiva de contenidos virtuales


## Table of contents

- [Introducción](#introducción)
- [Características](#características)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
  - [Requisitos previos](#requisitos-previos)
  - [Instalación desde código fuente](#instalación-desde-código-fuente)
- [Uso](#uso)
  - [Ejecución básica](#ejecución-básica)
  - [Configuración personalizada](#configuración-personalizada)
- [Componentes](#componentes)
  - [PoseDetector](#posedetector)
  - [ControlSystem](#controlsystem)
  - [Visualizer](#visualizer)
  - [Controller](#controller)
- [Configuración](#configuración)
- [Ejemplos](#ejemplos)
- [Especificaciones técnicas](#especificaciones-técnicas)
- [Bugs y solicitudes de características](#bugs-y-solicitudes-de-características)
- [Creadores](#creadores)
- [Licencia](#licencia)

## Introducción

**Gestur** es un sistema de control de interfaces 3D mediante gestos y pose corporal desarrollado específicamente para museos y exposiciones interactivas. Utiliza tecnologías avanzadas de detección de gestos y seguimiento corporal en tiempo real, permitiendo una exploración inmersiva y natural de contenidos virtuales sin necesidad de hardware especializado.

El proyecto aprovecha la cámara web integrada en la mayoría de ordenadores y combina los avances en visión por computador —especialmente en el reconocimiento de poses mediante redes neuronales— para transformar gestos naturales en un sistema de control 3D completamente basado en software.

## Características

* **Control sin contacto**: Interacción completamente hands-free mediante movimientos naturales de cabeza y torso
* **Hardware mínimo**: Funciona únicamente con una cámara RGB estándar, sin sensores especializados
* **Tiempo real**: Procesamiento fluido a 17-24 FPS en Raspberry Pi 5
* **Gestos intuitivos**: Sistema de aprendizaje implícito que no requiere instrucciones
* **Arquitectura modular**: Componentes intercambiables y configurables
* **Multiplataforma**: Compatible con Windows, macOS y Linux
* **Optimizado para museos**: Diseñado específicamente para espacios culturales y educativos

## Estructura del proyecto

```text
gestur/
├── controller.py          # Controlador principal y orquestador
├── pose_detector.py       # Detección de pose con MediaPipe
├── visualizer.py          # Visualizador 3D con Panda3D
├── control_system.py      # Sistema de mapeo de gestos
├── requirements.txt       # Dependencias del proyecto
└── README.md             # Documentación
```

## Instalación

### Requisitos previos

Asegúrate de tener los siguientes componentes instalados en tu sistema:

- **Python 3.9 o superior**
- **Cámara RGB** (webcam integrada o USB)
- **OpenCV** para procesamiento de video
- **MediaPipe** para detección de pose
- **Panda3D** para renderizado 3D

### Instalación desde código fuente

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/gestur/gestur.git
   cd gestur
   ```

2. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verifica la instalación**:
   ```bash
   python controller.py --help
   ```

## Uso

### Ejecución básica

Para ejecutar Gestur con un modelo 3D:

```bash
python controller.py modelo.obj
```

### Configuración personalizada

Ejecutar con modo debug activado:

```bash
python controller.py modelo.obj --verbose
```

Especificar preset de control:

```bash
python controller.py modelo.obj --control-preset continuous --verbose
```

### Gestos de control

El sistema reconoce los siguientes gestos naturales:

- **Zona neutra (25%-75%)**: Sin rotación, modelo estático
- **Cabeza > 75%**: Rotación continua hacia la derecha
- **Cabeza < 25%**: Rotación continua hacia la izquierda
- **Acercarse**: Ampliación del modelo (escala 1:1.75)
- **Alejarse**: Modelo en tamaño normal (escala 1:1)

## Componentes

### PoseDetector

El componente `PoseHandTracker` es responsable de la detección en tiempo real de la posición y orientación de la cabeza, torso y manos2].

#### Características principales

- **MediaPipe Integration**: Utiliza MediaPipe Pose 2D para detección robusta2]
- **Suavizado temporal**: Filtrado exponencial para eliminar ruido
- **Múltiples fuentes**: Detección simultánea de cabeza, torso y manos
- **Configuración flexible**: Parámetros ajustables de respuesta y suavizado

#### Configuración

```python
tracker = PoseHandTracker(
    response_time_ms=10,     # Tiempo de respuesta
    smoothing_time_ms=50,    # Tiempo de suavizado
    use_pose=True,           # Activar detección de pose
    use_hands=False,         # Desactivar manos en Pi
    mirror=True,             # Efecto espejo
    verbose=False            # Modo debug
)
```

### ControlSystem

El sistema de control traduce las poses detectadas en comandos 3D mediante mapeos configurables.

#### Tipos de mapeo

- **Rotación híbrida**: Combinación de control proporcional y continuo
- **Mapeo estándar**: Relación directa entre pose y transformación
- **Suavizado exponencial**: Filtros temporales para estabilidad

#### Controlador híbrido

```python
hybrid_controller = HybridRotationController(
    max_degrees=30.0,                    # Rotación máxima
    left_threshold=0.25,                 # Umbral izquierdo
    right_threshold=0.75,                # Umbral derecho
    continuous_speed_degrees_per_second=100.0,  # Velocidad continua
    center=0.5,                          # Centro neutral
    invert=True                          # Invertir dirección
)
```

### Visualizer

El componente `ControlledObjViewer` maneja el renderizado 3D usando Panda3D.

#### Características

- **Múltiples formatos**: Soporte para OBJ y GLTF
- **Texturas**: Carga automática de materiales
- **60 FPS**: Renderizado independiente del detector
- **Pantalla completa**: Optimizado para instalaciones

#### Métodos principales

```python
# Actualizar modelo con múltiples propiedades
visualizer.update_model(
    position=[x, y, z],      # Posición
    rotation=[yaw, pitch, roll],  # Rotación
    scale=factor             # Escala
)

# Estado actual
current_state = visualizer.get_current_state()
```

### Controller

El controlador principal `PoseController` actúa como orquestador entre todos los componentes.

#### Funciones principales

- **Inicialización**: Configuración automática de componentes
- **Callbacks**: Manejo de eventos de pose
- **Cleanup**: Liberación de recursos
- **Debug**: Información detallada en modo verbose

## Configuración

### Suavizado exponencial

```python
smoother_config = {
    'alpha': 0.3,           # Factor de suavizado (0-1)
    'decay_rate': 0.1,      # Velocidad de decaimiento
    'center_x': 0.5,        # Centro horizontal
    'center_y': 0.5         # Centro vertical
}
```

### Sistema de control por defecto

El sistema incluye mapeos preconfigurados:

- **head_x_hybrid_roll**: Control híbrido de rotación en roll
- **head_x_to_yaw**: Mapeo de cabeza a rotación yaw
- **head_y_to_pitch**: Mapeo de cabeza a rotación pitch  
- **head_scale_to_model_scale**: Control de escala por proximidad

## Ejemplos

### Ejemplo básico

```python
from pose_detector import PoseHandTracker
from visualizer import ControlledObjViewer
from control_system import create_default_control_system
from controller import PoseController

# Crear controlador
controller = PoseController(
    obj_path="modelo.obj",
    control_system=create_default_control_system(),
    verbose=True
)

# Ejecutar sistema
controller.run()
```

### Configuración personalizada

```python
# Sistema de control personalizado
control_system = ControlSystem()

# Agregar mapeos específicos
control_system.add_mapping(ControlMapping(
    name="custom_rotation",
    input_extractor=extractors['head_x'],
    output_applier=appliers['rotation_yaw'](max_degrees=45.0),
    smoother=ExponentialSmoother(alpha=0.5)
))
```

## Especificaciones técnicas

### Rendimiento

| Plataforma | FPS Promedio | Detección Manos | Uso CPU |
|------------|--------------|-----------------|---------|
| Mac M2 Pro | 22-24 FPS | 49.5% | Medio |
| Raspberry Pi 5 | 17-24 FPS | < 15% | Alto |

### Requisitos mínimos

- **CPU**: Dual-core 1.5 GHz (Raspberry Pi 5 o superior)
- **RAM**: 4 GB
- **Cámara**: RGB 720p mínimo, 1080p recomendado
- **Almacenamiento**: 2 GB para dependencias

### Latencia

- **Detección**: < 50ms
- **Renderizado**: 16ms (60 FPS)
- **Total**: < 100ms para respuesta perceptible

## Bugs y solicitudes de características

¿Tienes un bug o una solicitud de característica? Por favor busca en los issues existentes y cerrados. Si tu problema o idea no está contemplada, [abre un nuevo issue](https://github.com/gestur/issues/new).

## Creadores

Este proyecto fue desarrollado como Trabajo de Fin de Grado en la Escola d'Enginyeria de la Universitat Autònoma de Barcelona (UAB).

### Xavier Burgos
- Email: xavi@dzin.es
- Profesor: Prof. Fernando Vilariño
- Centro: Computer Vision Center - Dep. Computer Science UAB
- Curso: 2024/2025

### Agradecimientos

- **Prof. Fernando Vilariño**: Director del TFG y guía constante
- **Centro de Visión por Computador (CVC)**: Cesión de hardware y servidor de cálculo
- **Fran Iglesias y Fundación Épica – La Fura dels Baus**: Facilitar pantalla 4K para prototipo
- **Cátedra UAB–Cruïlla (TSI-100929-2023-2)**: Apoyo institucional

## Licencia

Este proyecto está desarrollado como trabajo académico en la Universitat Autònoma de Barcelona. Para información sobre uso y distribución, consulta los términos específicos del proyecto académico.
