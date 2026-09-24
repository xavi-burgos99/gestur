import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  MantineProvider,
  AppShell,
  Container,
  Group,
  Stack,
  Title,
  Text,
  Button,
  Tabs,
  Paper,
  Badge,
  SimpleGrid,
  NumberInput,
  Select,
  Switch,
  TextInput,
  PasswordInput,
  Modal,
  Alert,
  Loader,
  FileButton,
  Accordion,
  Divider,
  ActionIcon,
  Tooltip,
  ThemeIcon,
  Box,
} from "@mantine/core";
import {
  IconCube,
  IconAdjustmentsHorizontal,
  IconSettings,
  IconHandMove,
  IconArrowUpRight,
  IconUpload,
  IconCheck,
  IconWifi,
  IconLock,
  IconLockOpen,
  IconLogout,
  IconPlus,
  IconTrash,
  IconArrowRight,
  IconAlertCircle,
  IconRefresh,
  IconDeviceDesktop,
  IconBolt,
} from "@tabler/icons-react";
import "@mantine/core/styles.css";
import "./styles.css";

const inputs = {
  head_x: "Cabeza · horizontal",
  head_y: "Cabeza · vertical",
  head_scale: "Cabeza · distancia",
  hands_center_x: "Manos · horizontal",
  hands_center_y: "Manos · vertical",
  hands_distance: "Manos · distancia",
  hands_separation_x: "Manos · separación",
  left_hand_pitch: "Mano izquierda · inclinación",
  right_hand_pitch: "Mano derecha · inclinación",
  left_hand_yaw: "Mano izquierda · giro lateral",
  right_hand_yaw: "Mano derecha · giro lateral",
  left_hand_rotation: "Mano izquierda · giro",
  right_hand_rotation: "Mano derecha · giro",
  left_hand_pinch: "Mano izquierda · pinza",
  right_hand_pinch: "Mano derecha · pinza",
};
const outputs = {
  rotation_yaw: "Giro horizontal",
  rotation_pitch: "Giro vertical",
  rotation_roll: "Inclinación lateral",
  position_x: "Desplazar horizontalmente",
  position_y: "Desplazar en profundidad",
  position_z: "Desplazar verticalmente",
  scale_uniform: "Tamaño del modelo",
};
const options = (obj) =>
  Object.entries(obj).map(([value, label]) => ({ value, label }));
async function api(url, { method = "GET", body } = {}) {
  const form = body instanceof FormData;
  const response = await fetch(`/api/${url}`, {
    method,
    headers: {
      "X-Gestur-Request": "1",
      ...(!form && body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { body: form ? body : JSON.stringify(body) } : {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401 && url !== "session")
      window.dispatchEvent(new Event("gestur-expired"));
    throw new Error(data.error || "No hay conexión con el dispositivo.");
  }
  return data;
}
function Numeric({
  label,
  value,
  onChange,
  min = 0,
  max = 360,
  step = 1,
  suffix,
  description,
}) {
  return (
    <NumberInput
      label={label}
      description={description}
      value={value}
      onChange={(v) => onChange(v === "" ? 0 : Number(v))}
      min={min}
      max={max}
      step={step}
      suffix={suffix}
      decimalScale={3}
    />
  );
}
function SectionTitle({ eyebrow, title, description, action }) {
  return (
    <Group justify="space-between" align="flex-end" mb={30}>
      <div>
        <Text className="eyebrow">{eyebrow}</Text>
        <Title order={1}>{title}</Title>
        <Text c="dimmed" mt={8}>
          {description}
        </Text>
      </div>
      {action}
    </Group>
  );
}
function ModelArt({ builtin }) {
  return (
    <div
      className={`model-art ${builtin ? "stone" : "digital"}`}
      aria-hidden="true"
    >
      {builtin ? (
        <svg viewBox="0 0 260 180">
          <defs>
            <linearGradient id="stone" x2="1" y2="1">
              <stop stopColor="#c2baad" />
              <stop offset="1" stopColor="#8d8b80" />
            </linearGradient>
          </defs>
          <ellipse
            cx="130"
            cy="159"
            rx="61"
            ry="8"
            fill="#45534a"
            opacity=".1"
          />
          <path d="M87 58 155 39 183 53 116 74Z" fill="#d3ccbf" />
          <path d="M87 58 116 74 116 91 88 75Z" fill="#9b9b8c" />
          <path d="M116 74 183 53 181 73 116 91Z" fill="#b8b3a4" />
          <path d="m97 80 19 11 58-18-12 44-43 14-17-11Z" fill="url(#stone)" />
          <path d="m105 120 14 11 43-14v26l-44 15-14-9Z" fill="#ada898" />
          <path d="m119 131 43-14v26l-44 15Z" fill="#98998b" />
          <g fill="none" stroke="#e0d9c9" strokeWidth="3">
            <path d="m105 89 5 24 9 9 6-19 5-12 7 23 8-26 9 18 11-26" />
            <path d="m111 141 7 6 35-12" />
          </g>
        </svg>
      ) : (
        <IconCube size={76} stroke={1} />
      )}
    </div>
  );
}
function Models({ config, setConfig, notify }) {
  const [runtime, setRuntime] = useState({ online: false });
  useEffect(() => {
    const poll = () =>
      api("runtime")
        .then(setRuntime)
        .catch(() => setRuntime({ online: false }));
    poll();
    const timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, []);
  const [models, setModels] = useState([]);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const load = () => {
    setLoading(true);
    setError("");
    api("models")
      .then((d) => setModels(d.models))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);
  async function upload(file) {
    if (!file) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      await api("models", { method: "POST", body: form });
      notify("Modelo importado. Ya puedes activarlo.");
      load();
    } catch (e) {
      notify(e.message, true);
    } finally {
      setBusy(false);
    }
  }
  async function activate(id) {
    setBusy(true);
    try {
      const result = await api("models/active", {
        method: "PUT",
        body: { id },
      });
      setConfig(result.config);
      notify(
        "Selección guardada. El visualizador intentará cargar el modelo en unos segundos.",
      );
    } catch (e) {
      notify(e.message, true);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <SectionTitle
        eyebrow="TU COLECCIÓN"
        title="Modelos 3D"
        description="Elige la pieza que responde a tus movimientos."
        action={
          <FileButton onChange={upload} accept=".zip,application/zip">
            {(props) => (
              <Button
                {...props}
                leftSection={<IconUpload size={18} />}
                loading={busy}
              >
                Importar modelo
              </Button>
            )}
          </FileButton>
        }
      />
      <Alert
        color={runtime.online ? (runtime.error ? "red" : "teal") : "gray"}
        mb="lg"
        title={
          runtime.online
            ? runtime.error
              ? "El visor no ha podido cargar la selección"
              : "Visor conectado"
            : "Visor no conectado"
        }
      >
        {runtime.online
          ? runtime.error ||
            (runtime.rendered_model === config.active_model
              ? "El modelo seleccionado se está mostrando en el dispositivo."
              : "El visor está aplicando la selección.")
          : "Puedes guardar cambios. Se aplicarán cuando el visualizador esté en marcha."}
      </Alert>
      {error && (
        <Alert color="red" mb="lg" title="No se pudo cargar la colección">
          {error}
          <Button mt="sm" variant="light" onClick={load}>
            Reintentar
          </Button>
        </Alert>
      )}
      {loading ? (
        <Loader />
      ) : (
        <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="xl">
          {models.map((model) => (
            <Paper
              className={`model-card ${config.active_model === model.id ? "selected" : ""}`}
              key={model.id}
              withBorder
            >
              <ModelArt builtin={model.builtin} />
              <Stack p="xl" gap="md">
                <Group justify="space-between">
                  <Title order={3}>{model.name}</Title>
                  <Badge
                    variant="light"
                    color={config.active_model === model.id ? "teal" : "gray"}
                  >
                    {runtime.online && runtime.rendered_model === model.id
                      ? "En pantalla"
                      : config.active_model === model.id
                        ? "Seleccionado"
                        : model.format}
                  </Badge>
                </Group>
                <Text c="dimmed" size="sm">
                  {model.builtin
                    ? "Colección original · Siempre disponible"
                    : `${model.format} · ${(model.size / 1024 / 1024).toFixed(1)} MB`}
                </Text>
                <Button
                  fullWidth
                  variant={
                    config.active_model === model.id ? "light" : "default"
                  }
                  leftSection={
                    config.active_model === model.id ? (
                      <IconCheck size={17} />
                    ) : (
                      <IconArrowUpRight size={17} />
                    )
                  }
                  disabled={busy || config.active_model === model.id}
                  onClick={() => activate(model.id)}
                >
                  {config.active_model === model.id
                    ? "Modelo seleccionado"
                    : "Mostrar en pantalla"}
                </Button>
              </Stack>
            </Paper>
          ))}
          <Paper className="import-guide" p="xl" withBorder>
            <ThemeIcon variant="light" size={48} radius="xl">
              <IconUpload size={24} />
            </ThemeIcon>
            <Title order={3} mt="xl">
              Una nueva perspectiva
            </Title>
            <Text c="dimmed" mt="sm" size="sm">
              Sube un ZIP con un único modelo OBJ, glTF o GLB. Incluye sus
              materiales y texturas en las carpetas originales.
            </Text>
            <Divider my="lg" />
            <Text size="xs" c="dimmed">
              Hasta 100 MB por ZIP · 250 MB al descomprimir · 500 archivos
            </Text>
          </Paper>
        </SimpleGrid>
      )}
      <Paper className="note-panel" mt={28} p="lg">
        <Group wrap="nowrap" align="flex-start">
          <IconDeviceDesktop size={23} />
          <div>
            <Text fw={600} size="sm">
              Tu modelo, en movimiento
            </Text>
            <Text size="sm" c="dimmed">
              La selección se guarda en el dispositivo. Los cambios de gestos y
              sensibilidad se ajustan en Parámetros.
            </Text>
          </div>
        </Group>
      </Paper>
    </>
  );
}
function Parameters({ config, setConfig, defaults, notify }) {
  const [draft, setDraft] = useState(() => structuredClone(config));
  const [saving, setSaving] = useState(false);
  useEffect(() => setDraft(structuredClone(config)), [config]);
  const dirty = JSON.stringify(draft) !== JSON.stringify(config);
  const update = (group, key, value) =>
    setDraft((d) => ({ ...d, [group]: { ...d[group], [key]: value } }));
  const mapping = (i, changes) =>
    setDraft((d) => ({
      ...d,
      controls: {
        ...d.controls,
        mappings: d.controls.mappings.map((m, n) =>
          n === i ? { ...m, ...changes } : m,
        ),
      },
    }));
  function mode(i, value) {
    const m = draft.controls.mappings[i];
    mapping(i, {
      mode: value,
      ...(value === "hybrid"
        ? {
            output: m.output.startsWith("rotation_")
              ? m.output
              : "rotation_yaw",
            left_threshold: 0.25,
            center: 0.5,
            right_threshold: 0.75,
            continuous_speed: 100,
          }
        : value === "stepped"
          ? {
              output: "scale_uniform",
              threshold: 0.4,
              small_scale: 1,
              large_scale: 1.75,
              transition_ms: 750,
              hysteresis: 0.02,
            }
          : {}),
    });
  }
  async function save() {
    setSaving(true);
    try {
      const d = await api("config", { method: "PUT", body: draft });
      setConfig(d.config);
      notify(
        "Parámetros guardados. El dispositivo los aplicará automáticamente.",
      );
    } catch (e) {
      notify(e.message, true);
    } finally {
      setSaving(false);
    }
  }
  return (
    <>
      <SectionTitle
        eyebrow="A TU MANERA"
        title="Parámetros"
        description="Ajusta cómo se transforma cada gesto en movimiento."
        action={
          <Button
            leftSection={<IconCheck size={18} />}
            disabled={!dirty}
            loading={saving}
            onClick={save}
          >
            {dirty ? "Guardar cambios" : "Todo guardado"}
          </Button>
        }
      />
      <SimpleGrid cols={{ base: 1, md: 3 }} mb="xl">
        <Paper p="xl" withBorder>
          <Text className="eyebrow">RECONOCIMIENTO</Text>
          <Title order={3} mb="lg">
            Cuerpo y manos
          </Title>
          <Stack>
            <Switch
              label="Seguir cabeza y cuerpo"
              checked={draft.tracking.use_pose}
              onChange={(e) =>
                update("tracking", "use_pose", e.currentTarget.checked)
              }
            />
            <Switch
              label="Reconocer las manos"
              description="Activa giro de muñeca, pinza y separación."
              checked={draft.tracking.use_hands}
              onChange={(e) =>
                update("tracking", "use_hands", e.currentTarget.checked)
              }
            />
            <Switch
              label="Movimiento en espejo"
              checked={draft.tracking.mirror}
              onChange={(e) =>
                update("tracking", "mirror", e.currentTarget.checked)
              }
            />
          </Stack>
        </Paper>
        <Paper p="xl" withBorder>
          <Text className="eyebrow">RESPUESTA</Text>
          <Title order={3} mb="lg">
            Suavidad del gesto
          </Title>
          <Numeric
            label="Suavizado"
            description="Más tiempo suaviza el movimiento y aumenta la latencia."
            value={draft.controls.smoothing_ms}
            onChange={(v) => update("controls", "smoothing_ms", v)}
            max={1000}
            step={10}
            suffix=" ms"
          />
        </Paper>
        <Paper p="xl" withBorder>
          <Text className="eyebrow">REPOSO</Text>
          <Title order={3} mb="lg">
            Volver al centro
          </Title>
          <Numeric
            label="Tiempo sin detectar a nadie"
            value={draft.controls.reset_timeout_seconds}
            onChange={(v) => update("controls", "reset_timeout_seconds", v)}
            max={30}
            step={0.5}
            suffix=" s"
          />
          <Text c="dimmed" size="xs" mt="sm">
            0 inicia el regreso inmediatamente.
          </Text>
        </Paper>
      </SimpleGrid>
      <Group justify="space-between" mb="md">
        <Title order={2}>Gestos y movimientos</Title>
        <Button
          variant="subtle"
          leftSection={<IconPlus size={17} />}
          disabled={draft.controls.mappings.length >= 32}
          onClick={() =>
            update("controls", "mappings", [
              ...draft.controls.mappings,
              {
                id: `gesture_${Date.now()}`,
                input: "right_hand_rotation",
                output: "rotation_yaw",
                mode: "absolute",
                enabled: false,
                scale: 90,
                invert: false,
                center: 0.5,
              },
            ])
          }
        >
          Añadir gesto
        </Button>
      </Group>
      <Text c="dimmed" size="sm" mb="lg">
        Cada movimiento puede tener un gesto activo. Desactiva un gesto antes de
        sustituirlo.
      </Text>
      <Accordion variant="separated" radius="md">
        {draft.controls.mappings.map((m, i) => (
          <Accordion.Item value={m.id} key={m.id}>
            <Accordion.Control>
              <Group gap="md">
                <ThemeIcon
                  color={m.enabled ? "teal" : "gray"}
                  variant="light"
                  size={38}
                >
                  <IconHandMove size={21} />
                </ThemeIcon>
                <div>
                  <Text fw={600}>
                    {inputs[m.input]} <span className="arrow">→</span>{" "}
                    {outputs[m.output]}
                  </Text>
                  <Text size="xs" c="dimmed">
                    {m.enabled ? "Activo" : "Desactivado"} ·{" "}
                    {m.mode === "hybrid"
                      ? "Continuo en los extremos"
                      : m.mode === "stepped"
                        ? "Dos tamaños"
                        : "Proporcional"}
                  </Text>
                </div>
              </Group>
            </Accordion.Control>
            <Accordion.Panel>
              <Stack pt="md">
                <Group justify="space-between">
                  <Switch
                    label="Gesto activo"
                    checked={m.enabled}
                    onChange={(e) =>
                      mapping(i, { enabled: e.currentTarget.checked })
                    }
                  />
                  <Tooltip label="Eliminar gesto">
                    <ActionIcon
                      aria-label="Eliminar gesto"
                      color="red"
                      variant="subtle"
                      onClick={() =>
                        update(
                          "controls",
                          "mappings",
                          draft.controls.mappings.filter((_, n) => n !== i),
                        )
                      }
                    >
                      <IconTrash size={18} />
                    </ActionIcon>
                  </Tooltip>
                </Group>
                <SimpleGrid cols={{ base: 1, sm: 3 }}>
                  <Select
                    label="Gesto de entrada"
                    value={m.input}
                    data={options(inputs)}
                    onChange={(v) => mapping(i, { input: v })}
                  />
                  <Select
                    label="Movimiento del objeto"
                    value={m.output}
                    data={options(outputs).filter((o) =>
                      m.mode === "hybrid"
                        ? o.value.startsWith("rotation_")
                        : m.mode === "stepped"
                          ? o.value === "scale_uniform"
                          : true,
                    )}
                    onChange={(v) => mapping(i, { output: v })}
                  />
                  <Select
                    label="Respuesta"
                    value={m.mode}
                    data={[
                      { value: "absolute", label: "Proporcional" },
                      { value: "hybrid", label: "Continua en los extremos" },
                      { value: "stepped", label: "Dos tamaños" },
                    ]}
                    onChange={(v) => mode(i, v)}
                  />
                </SimpleGrid>
                {!draft.tracking.use_hands && m.input.includes("hand") && (
                  <Alert color="yellow">
                    Activa «Reconocer las manos» para utilizar este gesto.
                  </Alert>
                )}
                {!draft.tracking.use_pose && m.input.startsWith("head") && (
                  <Alert color="yellow">
                    Activa «Seguir cabeza y cuerpo» para utilizar este gesto.
                  </Alert>
                )}
                <SimpleGrid cols={{ base: 2, md: 4 }}>
                  <Numeric
                    label="Intensidad"
                    value={m.scale}
                    onChange={(v) => mapping(i, { scale: v })}
                    max={360}
                    step={0.1}
                  />
                  <Numeric
                    label="Centro neutro"
                    value={m.center}
                    onChange={(v) => mapping(i, { center: v })}
                    max={1}
                    step={0.05}
                  />
                  {m.mode === "hybrid" && (
                    <>
                      <Numeric
                        label="Extremo izquierdo"
                        value={m.left_threshold}
                        onChange={(v) => mapping(i, { left_threshold: v })}
                        min={0.01}
                        max={0.99}
                        step={0.05}
                      />
                      <Numeric
                        label="Extremo derecho"
                        value={m.right_threshold}
                        onChange={(v) => mapping(i, { right_threshold: v })}
                        min={0.01}
                        max={0.99}
                        step={0.05}
                      />
                      <Numeric
                        label="Velocidad continua (°/s)"
                        value={m.continuous_speed}
                        onChange={(v) => mapping(i, { continuous_speed: v })}
                      />
                    </>
                  )}
                  {m.mode === "stepped" && (
                    <>
                      <Numeric
                        label="Umbral de cambio"
                        value={m.threshold}
                        onChange={(v) => mapping(i, { threshold: v })}
                        max={1}
                        step={0.05}
                      />
                      <Numeric
                        label="Margen del umbral"
                        value={m.hysteresis}
                        onChange={(v) => mapping(i, { hysteresis: v })}
                        max={0.2}
                        step={0.01}
                      />
                      <Numeric
                        label="Tamaño pequeño"
                        value={m.small_scale}
                        onChange={(v) => mapping(i, { small_scale: v })}
                        min={0.1}
                        max={5}
                        step={0.1}
                      />
                      <Numeric
                        label="Tamaño grande"
                        value={m.large_scale}
                        onChange={(v) => mapping(i, { large_scale: v })}
                        min={0.1}
                        max={5}
                        step={0.1}
                      />
                      <Numeric
                        label="Transición"
                        value={m.transition_ms}
                        onChange={(v) => mapping(i, { transition_ms: v })}
                        max={5000}
                        step={50}
                        suffix=" ms"
                      />
                    </>
                  )}
                </SimpleGrid>
                <Switch
                  label="Invertir dirección"
                  checked={m.invert}
                  onChange={(e) =>
                    mapping(i, { invert: e.currentTarget.checked })
                  }
                />
              </Stack>
            </Accordion.Panel>
          </Accordion.Item>
        ))}
      </Accordion>
      <Accordion mt="xl" variant="separated">
        <Accordion.Item value="advanced">
          <Accordion.Control>Captura y renderizado</Accordion.Control>
          <Accordion.Panel>
            <SimpleGrid cols={{ base: 1, sm: 3 }}>
              <Numeric
                label="Reconocimiento de cuerpo"
                suffix=" fps"
                value={draft.tracking.inference_fps}
                onChange={(v) => update("tracking", "inference_fps", v)}
                min={5}
                max={60}
              />
              <Numeric
                label="Reconocimiento de manos"
                suffix=" fps"
                value={draft.tracking.hand_fps}
                onChange={(v) => update("tracking", "hand_fps", v)}
                min={5}
                max={30}
              />
              <Numeric
                label="Fluidez de pantalla"
                suffix=" fps"
                value={draft.render.target_fps}
                onChange={(v) => update("render", "target_fps", v)}
                min={15}
                max={120}
              />
              <Numeric
                label="Índice de cámara"
                value={draft.tracking.camera_index}
                onChange={(v) => update("tracking", "camera_index", v)}
                max={16}
              />
              <Numeric
                label="Ancho de captura"
                suffix=" px"
                value={draft.tracking.width}
                onChange={(v) => update("tracking", "width", v)}
                min={160}
                max={1920}
              />
              <Numeric
                label="Alto de captura"
                suffix=" px"
                value={draft.tracking.height}
                onChange={(v) => update("tracking", "height", v)}
                min={120}
                max={1080}
              />
              <Numeric
                label="Suavizado del detector"
                suffix=" ms"
                value={draft.tracking.smoothing_ms}
                onChange={(v) => update("tracking", "smoothing_ms", v)}
                max={1000}
              />
              <Numeric
                label="Duración de regreso al centro"
                suffix=" s"
                value={draft.controls.reset_duration_seconds}
                onChange={(v) =>
                  update("controls", "reset_duration_seconds", v)
                }
                min={0.1}
                max={10}
                step={0.1}
              />
              <Select
                label="Suavizado de bordes"
                value={String(draft.render.antialias_samples)}
                data={[
                  { value: "0", label: "Desactivado" },
                  { value: "2", label: "2× · Equilibrado" },
                  { value: "4", label: "4× · Más calidad" },
                ]}
                onChange={(v) =>
                  update("render", "antialias_samples", Number(v))
                }
              />
            </SimpleGrid>
            <Group mt="xl">
              <Switch
                label="Ocultar cursor"
                checked={draft.render.hide_cursor}
                onChange={(e) =>
                  update("render", "hide_cursor", e.currentTarget.checked)
                }
              />
              <Switch
                label="Pantalla completa"
                checked={draft.render.fullscreen}
                onChange={(e) =>
                  update("render", "fullscreen", e.currentTarget.checked)
                }
              />
            </Group>
            <Text size="xs" c="dimmed" mt="md">
              La captura, el suavizado de bordes y la pantalla completa pueden
              requerir reiniciar el visualizador. Los gestos se actualizan en
              vivo.
            </Text>
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
      <Group justify="space-between" mt="xl">
        <Button
          variant="subtle"
          color="gray"
          onClick={() =>
            setDraft({
              ...structuredClone(defaults),
              active_model: config.active_model,
            })
          }
        >
          Restaurar parámetros originales
        </Button>
        <Button disabled={!dirty} loading={saving} onClick={save}>
          Guardar cambios
        </Button>
      </Group>
    </>
  );
}
function Settings({ notify }) {
  const [wifi, setWifi] = useState(null);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [dialog, setDialog] = useState(null);
  const [password, setPassword] = useState("");
  async function load(quiet = false) {
    if (!quiet) setLoading(true);
    try {
      const d = await api("wifi");
      setWifi(d);
      if (!quiet) setName(d.ssid);
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    load();
  }, []);
  useEffect(() => {
    if (!["pending", "applying"].includes(wifi?.job?.state)) return;
    const interval = setInterval(() => load(true), 3000);
    return () => clearInterval(interval);
  }, [wifi?.job?.state]);
  async function save(mode = "keep") {
    setBusy(true);
    try {
      const body = { ssid: name };
      if (mode === "password") body.password = password;
      if (mode === "remove") body.password = null;
      const d = await api("wifi", { method: "PUT", body });
      setWifi((w) => ({ ...w, job: d.job }));
      setDialog(null);
      setPassword("");
      notify(d.job.message);
    } catch (e) {
      notify(e.message, true);
    } finally {
      setBusy(false);
    }
  }
  const pending = ["pending", "applying"].includes(wifi?.job?.state);
  return (
    <>
      <SectionTitle
        eyebrow="TU DISPOSITIVO"
        title="Configuración"
        description="Gestiona la conexión a tu instalación Gestur."
      />
      <Paper withBorder p={{ base: "lg", sm: 32 }} maw={820}>
        <Group justify="space-between" mb="xl">
          <Group>
            <ThemeIcon size={48} radius="md" variant="light">
              <IconWifi size={25} />
            </ThemeIcon>
            <div>
              <Title order={2}>Punto de acceso Wi-Fi</Title>
              <Text c="dimmed" size="sm">
                La red local desde la que accedes a este panel.
              </Text>
            </div>
          </Group>
          {wifi && (
            <Badge color={wifi.secured ? "teal" : "gray"} variant="light">
              {wifi.secured ? "Con contraseña" : "Red abierta"}
            </Badge>
          )}
        </Group>
        {loading ? (
          <Loader />
        ) : error ? (
          <Alert color="red" title="Wi-Fi no disponible">
            {error}
            <Button variant="light" mt="md" onClick={() => load()}>
              Volver a comprobar
            </Button>
          </Alert>
        ) : (
          <Stack gap="xl">
            {wifi?.job && (
              <Alert
                color={
                  wifi.job.state === "failed"
                    ? "red"
                    : wifi.job.state === "completed"
                      ? "teal"
                      : "blue"
                }
                title={
                  pending
                    ? "Actualizando la red"
                    : wifi.job.state === "completed"
                      ? "Configuración aplicada"
                      : "Cambio no completado"
                }
              >
                {wifi.job.message}
                {pending && (
                  <Text size="sm" mt="sm">
                    Espera unos segundos y vuelve a conectarte a «
                    {wifi.job.ssid}». Si no aparece, prueba la red anterior y
                    pulsa «Comprobar estado».
                  </Text>
                )}
              </Alert>
            )}
            <TextInput
              label="Nombre de la red"
              description="En nuevas instalaciones: GESTUR y los últimos cuatro caracteres de la MAC."
              value={name}
              onChange={(e) => setName(e.currentTarget.value)}
              maxLength={32}
              disabled={pending}
            />
            <Group justify="space-between">
              <Group gap="sm">
                {wifi.secured ? (
                  <IconLock size={21} />
                ) : (
                  <IconLockOpen size={21} />
                )}
                <div>
                  <Text fw={600}>Contraseña de la red</Text>
                  <Text size="sm" c="dimmed">
                    {wifi.secured
                      ? "La red requiere una contraseña."
                      : "Cualquier persona cercana puede conectarse."}
                  </Text>
                </div>
              </Group>
              <Group>
                {wifi.secured ? (
                  <>
                    <Button
                      variant="default"
                      disabled={pending}
                      onClick={() => setDialog("password")}
                    >
                      Cambiar contraseña
                    </Button>
                    <Button
                      variant="subtle"
                      color="red"
                      disabled={pending}
                      onClick={() => setDialog("remove")}
                    >
                      Eliminar contraseña
                    </Button>
                  </>
                ) : (
                  <Button
                    variant="light"
                    leftSection={<IconPlus size={17} />}
                    disabled={pending}
                    onClick={() => setDialog("password")}
                  >
                    Añadir contraseña
                  </Button>
                )}
              </Group>
            </Group>
            <Divider />
            <Alert
              variant="light"
              color="gray"
              icon={<IconAlertCircle size={18} />}
            >
              Al aplicar cambios, el punto de acceso se reinicia y puede
              desconectarte. La clave de administración del portal es
              independiente de la contraseña Wi-Fi.
            </Alert>
            <Group justify="space-between">
              <Button
                variant="subtle"
                leftSection={<IconRefresh size={17} />}
                onClick={() => load()}
              >
                Comprobar estado
              </Button>
              <Button
                disabled={pending || name === wifi.ssid || !name.trim()}
                loading={busy}
                onClick={() => save()}
              >
                Guardar nombre
              </Button>
            </Group>
          </Stack>
        )}
      </Paper>
      <Modal
        opened={!!dialog}
        onClose={() => {
          setDialog(null);
          setPassword("");
        }}
        title={
          dialog === "remove"
            ? "Eliminar contraseña Wi-Fi"
            : wifi?.secured
              ? "Cambiar contraseña Wi-Fi"
              : "Añadir contraseña Wi-Fi"
        }
        centered
      >
        <Stack>
          {dialog === "remove" ? (
            <Text>
              La red «{name}» quedará abierta. Cualquier persona cercana podrá
              conectarse.
            </Text>
          ) : (
            <PasswordInput
              label="Nueva contraseña"
              description="Entre 8 y 63 caracteres ASCII."
              value={password}
              onChange={(e) => setPassword(e.currentTarget.value)}
              minLength={8}
              maxLength={63}
              autoComplete="new-password"
            />
          )}
          <Text size="sm" c="dimmed">
            Se aplicará en unos segundos. Después tendrás que volver a
            conectarte a la red.
          </Text>
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setDialog(null)}>
              Cancelar
            </Button>
            <Button
              color={dialog === "remove" ? "red" : "teal"}
              disabled={
                dialog === "password" && !/^[\x20-\x7e]{8,63}$/.test(password)
              }
              loading={busy}
              onClick={() => save(dialog)}
            >
              {dialog === "remove"
                ? "Eliminar contraseña"
                : "Aplicar contraseña"}
            </Button>
          </Group>
        </Stack>
      </Modal>
    </>
  );
}
function App() {
  const [auth, setAuth] = useState(null);
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [config, setConfig] = useState(null);
  const [defaults, setDefaults] = useState(null);
  const [tab, setTab] = useState("models");
  const [notice, setNotice] = useState(null);
  const notify = (message, error = false) => setNotice({ message, error });
  useEffect(() => {
    api("session")
      .then((d) => setAuth(d.authenticated))
      .catch((e) => {
        setAuth(false);
        setError(e.message);
      });
    const expire = () => {
      setAuth(false);
      setConfig(null);
    };
    window.addEventListener("gestur-expired", expire);
    return () => window.removeEventListener("gestur-expired", expire);
  }, []);
  async function loadConfig() {
    try {
      const d = await api("config");
      setConfig(d.config);
      setDefaults(d.defaults);
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }
  useEffect(() => {
    if (auth) loadConfig();
  }, [auth]);
  async function login(e) {
    e.preventDefault();
    setBusy(true);
    try {
      await api("session", { method: "POST", body: { token } });
      setAuth(true);
      setToken("");
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  if (auth === null)
    return (
      <div className="centered">
        <Loader />
      </div>
    );
  if (!auth)
    return (
      <div className="login-page">
        <Paper p={40} withBorder radius="lg" className="login-card">
          <div className="brand-mark">
            <IconHandMove size={28} />
          </div>
          <Text className="eyebrow" mt="xl">
            GESTUR / CONTROL LOCAL
          </Text>
          <Title order={1} mt="sm">
            Dale movimiento.
          </Title>
          <Text c="dimmed" mt="sm" mb="xl">
            Tu colección, tus gestos, tu espacio.
          </Text>
          <form onSubmit={login}>
            <Stack>
              <PasswordInput
                label="Clave de administración"
                description="La clave que aparece al instalar Gestur."
                value={token}
                onChange={(e) => setToken(e.currentTarget.value)}
                autoComplete="current-password"
                required
              />
              {error && <Alert color="red">{error}</Alert>}
              <Button
                type="submit"
                loading={busy}
                rightSection={<IconArrowRight size={18} />}
              >
                Abrir panel
              </Button>
            </Stack>
          </form>
          <Text c="dimmed" size="xs" mt="xl">
            Conexión directa con tu dispositivo. Sin cuentas ni servicios
            externos.
          </Text>
        </Paper>
      </div>
    );
  return (
    <AppShell header={{ height: 84 }} padding={0}>
      <AppShell.Header>
        <Container size="xl" h="100%">
          <Group justify="space-between" h="100%">
            <Group gap={12}>
              <div className="brand-mark">
                <IconHandMove size={25} />
              </div>
              <div>
                <Text className="wordmark">
                  gestur<span>®</span>
                </Text>
                <Text size="xs" c="dimmed">
                  PANEL DE CONTROL
                </Text>
              </div>
            </Group>
            <Group>
              <Badge variant="dot" color="teal" visibleFrom="sm">
                Sesión local
              </Badge>
              <Tooltip label="Cerrar sesión">
                <ActionIcon
                  size="lg"
                  variant="subtle"
                  color="gray"
                  aria-label="Cerrar sesión"
                  onClick={async () => {
                    try {
                      await api("session", { method: "DELETE" });
                      setAuth(false);
                      setConfig(null);
                    } catch (e) {
                      notify(e.message, true);
                    }
                  }}
                >
                  <IconLogout size={19} />
                </ActionIcon>
              </Tooltip>
            </Group>
          </Group>
        </Container>
      </AppShell.Header>
      <AppShell.Main>
        <Container size="xl">
          <Tabs
            value={tab}
            onChange={(v) => {
              setTab(v);
              setNotice(null);
            }}
            keepMounted={false}
            className="main-tabs"
          >
            <Tabs.List>
              <Tabs.Tab value="models" leftSection={<IconCube size={19} />}>
                Modelos 3D
              </Tabs.Tab>
              <Tabs.Tab
                value="parameters"
                leftSection={<IconAdjustmentsHorizontal size={19} />}
              >
                Parámetros
              </Tabs.Tab>
              <Tabs.Tab
                value="settings"
                leftSection={<IconSettings size={19} />}
              >
                Configuración
              </Tabs.Tab>
            </Tabs.List>
            {notice && (
              <Alert
                mt="xl"
                color={notice.error ? "red" : "teal"}
                withCloseButton
                onClose={() => setNotice(null)}
                role="status"
              >
                {notice.message}
              </Alert>
            )}
            {!config ? (
              <Box py={40}>
                {error ? (
                  <Alert color="red">
                    {error}
                    <Button mt="sm" onClick={loadConfig}>
                      Reintentar
                    </Button>
                  </Alert>
                ) : (
                  <Loader />
                )}
              </Box>
            ) : (
              <>
                <Tabs.Panel value="models">
                  <Models
                    config={config}
                    setConfig={setConfig}
                    notify={notify}
                  />
                </Tabs.Panel>
                <Tabs.Panel value="parameters">
                  <Parameters
                    config={config}
                    setConfig={setConfig}
                    defaults={defaults}
                    notify={notify}
                  />
                </Tabs.Panel>
                <Tabs.Panel value="settings">
                  <Settings notify={notify} />
                </Tabs.Panel>
              </>
            )}
          </Tabs>
          <footer>
            <Group justify="space-between">
              <Text size="xs" c="dimmed">
                GESTUR · El movimiento conecta.
              </Text>
              <Group gap={5}>
                <IconBolt size={13} />
                <Text size="xs" c="dimmed">
                  Diseñado para Raspberry Pi
                </Text>
              </Group>
            </Group>
          </footer>
        </Container>
      </AppShell.Main>
    </AppShell>
  );
}
createRoot(document.getElementById("root")).render(
  <MantineProvider
    theme={{
      primaryColor: "teal",
      primaryShade: 8,
      fontFamily:
        'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      defaultRadius: "md",
      headings: { fontFamily: "inherit", fontWeight: "600" },
      components: {
        Button: { defaultProps: { size: "sm", radius: "md" } },
        Paper: { defaultProps: { radius: "md" } },
      },
    }}
  >
    <App />
  </MantineProvider>,
);
