export const CONTROL_OPTIONS = [
  {
    id: "rotation_yaw",
    label: "Giro horizontal",
    description: "Girar a izquierda y derecha.",
    icon: { subject: "model", motion: "yaw" },
  },
  {
    id: "rotation_pitch",
    label: "Giro vertical",
    description: "Girar hacia arriba y abajo.",
    icon: { subject: "model", motion: "pitch" },
  },
  {
    id: "rotation_roll",
    label: "Inclinación lateral",
    description: "Inclinar a un lado y al otro.",
    icon: { subject: "model", motion: "roll" },
  },
  {
    id: "position_x",
    label: "Desplazamiento horizontal",
    description: "Mover a izquierda y derecha.",
    icon: { subject: "model", motion: "translate-x" },
  },
  {
    id: "position_z",
    label: "Desplazamiento vertical",
    description: "Mover hacia arriba y abajo.",
    icon: { subject: "model", motion: "translate-y" },
  },
  {
    id: "position_y",
    label: "Desplazamiento en profundidad",
    description: "Acercar y alejar el modelo.",
    icon: { subject: "model", motion: "depth" },
  },
  {
    id: "scale_uniform",
    label: "Zoom (tamaño)",
    description: "Aumentar y reducir el modelo.",
    icon: { subject: "model", motion: "zoom" },
  },
];

const head = [
  [
    "x",
    "Desplazamiento horizontal",
    "Mueve la cabeza a izquierda y derecha.",
    "translate-x",
  ],
  [
    "y",
    "Desplazamiento vertical",
    "Mueve la cabeza hacia arriba y abajo.",
    "translate-y",
  ],
  [
    "scale",
    "Acercar y alejar",
    "Acerca o aleja la cabeza de la cámara.",
    "depth",
  ],
  ["yaw", "Giro horizontal", "Mira a izquierda y derecha.", "yaw"],
  ["pitch", "Giro vertical", "Mira hacia arriba y abajo.", "pitch"],
  [
    "roll",
    "Inclinación lateral",
    "Inclina la cabeza hacia los hombros.",
    "roll",
  ],
];
const hand = [
  [
    "x",
    "Desplazamiento horizontal",
    "Mueve la mano a izquierda y derecha.",
    "translate-x",
  ],
  [
    "y",
    "Desplazamiento vertical",
    "Mueve la mano hacia arriba y abajo.",
    "translate-y",
  ],
  ["yaw", "Giro horizontal", "Gira la palma a izquierda y derecha.", "yaw"],
  [
    "pitch",
    "Giro vertical",
    "Inclina la palma hacia delante y atrás.",
    "pitch",
  ],
  [
    "roll",
    "Inclinación lateral (3D)",
    "Inclina la palma a un lado y al otro.",
    "roll",
  ],
  ["pinch", "Pinza", "Junta o separa el pulgar y el índice.", "pinch"],
  ["openness", "Abrir y cerrar", "Abre la mano o cierra el puño.", "openness"],
  [
    "rotation",
    "Giro en pantalla (2D)",
    "Gira la mano en el plano de la imagen.",
    "roll",
  ],
];

export const GESTURE_OPTIONS = [
  ...head.map(([id, label, description, motion]) => ({
    id: `head_${id}`,
    label,
    description,
    category: "Cabeza",
    icon: { subject: "head", motion },
  })),
  ...["left", "right"].flatMap((side) =>
    hand.map(([id, label, description, motion]) => ({
      id: `${side}_hand_${id}`,
      label,
      description,
      category: side === "left" ? "Mano izquierda" : "Mano derecha",
      icon: { subject: "hand", motion, side },
    })),
  ),
  {
    id: "hands_center_x",
    label: "Desplazamiento horizontal",
    description: "Mueve las dos manos a izquierda y derecha.",
    category: "Ambas manos",
    icon: { subject: "hands", motion: "translate-x" },
  },
  {
    id: "hands_center_y",
    label: "Desplazamiento vertical",
    description: "Mueve las dos manos hacia arriba y abajo.",
    category: "Ambas manos",
    icon: { subject: "hands", motion: "translate-y" },
  },
  {
    id: "hands_distance",
    label: "Separación total",
    description: "Junta o separa las manos en la imagen.",
    category: "Ambas manos",
    icon: { subject: "hands", motion: "spread" },
  },
  {
    id: "hands_separation_x",
    label: "Separación horizontal",
    description: "Junta o separa las manos horizontalmente.",
    category: "Ambas manos",
    icon: { subject: "hands", motion: "spread-x" },
  },
];

export function responseOptions(output) {
  const result = [{ value: "absolute", label: "Proporcional" }];
  if (output.startsWith("rotation_"))
    result.push({ value: "hybrid", label: "Continua en los extremos" });
  if (output === "scale_uniform")
    result.push({ value: "stepped", label: "Dos tamaños" });
  return result;
}

export function modeChanges(control, mode) {
  if (!responseOptions(control.output).some((option) => option.value === mode))
    return {};
  if (mode === "hybrid") {
    const center =
      control.center > 0.01 && control.center < 0.99 ? control.center : 0.5;
    return {
      mode,
      center,
      left_threshold:
        control.left_threshold > 0 && control.left_threshold < center
          ? control.left_threshold
          : Math.max(0.01, center / 2),
      right_threshold:
        control.right_threshold < 1 && control.right_threshold > center
          ? control.right_threshold
          : Math.min(0.99, (1 + center) / 2),
      continuous_speed: control.continuous_speed ?? 100,
    };
  }
  if (mode === "stepped")
    return {
      mode,
      threshold: control.threshold ?? 0.4,
      small_scale: control.small_scale ?? 1,
      large_scale: control.large_scale ?? 1.75,
      transition_ms: control.transition_ms ?? 750,
      hysteresis: control.hysteresis ?? 0.02,
    };
  return { mode };
}

export function createControlMapping(id, output, input) {
  if (
    !CONTROL_OPTIONS.some((option) => option.id === output) ||
    !GESTURE_OPTIONS.some((option) => option.id === input)
  )
    return null;
  return {
    id,
    input,
    output,
    mode: "absolute",
    enabled: true,
    scale: output.startsWith("rotation_")
      ? 90
      : output === "scale_uniform"
        ? 1
        : 10,
    invert: false,
    center: 0.5,
  };
}
