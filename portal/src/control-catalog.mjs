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
const torso = [
  [
    "x",
    "Desplazamiento horizontal",
    "Mueve el cuerpo a izquierda y derecha.",
    "translate-x",
  ],
  [
    "y",
    "Desplazamiento vertical",
    "Mueve el cuerpo hacia arriba y abajo.",
    "translate-y",
  ],
  [
    "scale",
    "Acercar y alejar",
    "Acerca o aleja el cuerpo de la cámara.",
    "depth",
  ],
  ["yaw", "Giro horizontal", "Gira el cuerpo a izquierda y derecha.", "yaw"],
  [
    "pitch",
    "Giro vertical",
    "Inclina el cuerpo hacia delante y atrás.",
    "pitch",
  ],
  [
    "roll",
    "Inclinación lateral",
    "Inclina el cuerpo a un lado y al otro.",
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
  [
    "scale",
    "Acercar y alejar",
    "Acerca o aleja la mano de la cámara.",
    "depth",
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

function movementPath(body, id, side = null) {
  if (["x", "y", "scale"].includes(id))
    return { body, side, movement: "translation", axis: id };
  if (["yaw", "pitch", "roll"].includes(id))
    return { body, side, movement: "rotation", axis: id };
  if (["openness", "pinch"].includes(id)) return { body, side, movement: id };
  return null;
}

export const GESTURE_OPTIONS = [
  ...head.map(([id, label, description, motion]) => ({
    id: `head_${id}`,
    label,
    description,
    category: "Cabeza",
    icon: { subject: "head", motion },
    selection: movementPath("head", id),
  })),
  ...torso.map(([id, label, description, motion]) => ({
    id: `torso_${id}`,
    label,
    description,
    category: "Cuerpo",
    icon: { subject: "body", motion },
    selection: movementPath("body", id),
  })),
  ...["left", "right"].flatMap((side) =>
    hand.map(([id, label, description, motion]) => ({
      id: `${side}_hand_${id}`,
      label,
      description,
      category: side === "left" ? "Mano izquierda" : "Mano derecha",
      icon: { subject: "hand", motion, side },
      selection: movementPath("hands", id, side),
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
    label: "Distancia entre manos",
    description: "Junta o separa las manos en la imagen.",
    category: "Ambas manos",
    icon: { subject: "hands", motion: "spread" },
    selection: { body: "combined", combined: "distance" },
  },
  {
    id: "hands_separation_x",
    label: "Separación horizontal",
    description: "Junta o separa las manos horizontalmente.",
    category: "Ambas manos",
    icon: { subject: "hands", motion: "spread-x" },
  },
];

export function emptyGestureSelection() {
  return {
    body: null,
    side: null,
    movement: null,
    axis: null,
    combined: null,
  };
}

export function selectionForGesture(input) {
  const option = GESTURE_OPTIONS.find((item) => item.id === input);
  return {
    ...emptyGestureSelection(),
    ...(option?.selection || {}),
  };
}

export function gestureFromSelection(selection) {
  return (
    GESTURE_OPTIONS.find(
      (option) =>
        option.selection &&
        Object.entries(option.selection).every(
          ([key, value]) => selection[key] === value,
        ),
    )?.id || null
  );
}

export function changeGestureSelection(current, field, value) {
  if (field === "body") return { ...emptyGestureSelection(), [field]: value };
  const descendants = {
    side: ["movement", "axis", "combined"],
    movement: ["axis", "combined"],
    axis: [],
    combined: [],
  };
  if (!Object.hasOwn(descendants, field)) return current;
  const next = { ...current, [field]: value };
  for (const key of descendants[field]) next[key] = null;
  return next;
}

export function gestureSteps(selection) {
  const steps = [
    {
      id: "body",
      label: "Parte del cuerpo",
      options: [
        { id: "head", label: "Cabeza", icon: { subject: "head" } },
        { id: "body", label: "Cuerpo", icon: { subject: "body" } },
        { id: "hands", label: "Manos", icon: { subject: "hands" } },
        {
          id: "combined",
          label: "Combinado",
          icon: { subject: "hands", motion: "spread" },
        },
      ],
    },
  ];
  if (!selection.body) return steps;
  if (selection.body === "combined") {
    steps.push({
      id: "combined",
      label: "Gesto combinado",
      options: [
        {
          id: "distance",
          label: "Distancia entre manos",
          icon: { subject: "hands", motion: "spread" },
        },
      ],
    });
    return steps;
  }
  if (selection.body === "hands") {
    steps.push({
      id: "side",
      label: "Mano",
      options: [
        {
          id: "left",
          label: "Mano izquierda",
          icon: { subject: "hand", side: "left" },
        },
        {
          id: "right",
          label: "Mano derecha",
          icon: { subject: "hand", side: "right" },
        },
      ],
    });
    if (!selection.side) return steps;
  }
  const subject = selection.body === "hands" ? "hand" : selection.body;
  const icon = (motion) => ({
    subject,
    motion,
    side: selection.side || undefined,
  });
  steps.push({
    id: "movement",
    label: "Tipo de movimiento",
    options: [
      { id: "translation", label: "Desplazamiento", icon: icon("translate") },
      { id: "rotation", label: "Rotación", icon: icon("rotate") },
      ...(selection.body === "hands"
        ? [
            {
              id: "openness",
              label: "Apertura de la mano",
              icon: icon("openness"),
            },
            { id: "pinch", label: "Pinza", icon: icon("pinch") },
          ]
        : []),
    ],
  });
  if (!selection.movement || ["openness", "pinch"].includes(selection.movement))
    return steps;
  const axes =
    selection.movement === "translation"
      ? [
          ["x", "Horizontal", "translate-x"],
          ["y", "Vertical", "translate-y"],
          ["scale", "Profundidad", "depth"],
        ]
      : [
          ["yaw", "Giro horizontal", "yaw"],
          ["pitch", "Giro vertical", "pitch"],
          ["roll", "Inclinación lateral", "roll"],
        ];
  steps.push({
    id: "axis",
    label: "Eje",
    options: axes.map(([id, label, motion]) => ({
      id,
      label,
      icon: icon(motion),
    })),
  });
  return steps;
}

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
    !GESTURE_OPTIONS.some((option) => option.id === input && option.selection)
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
