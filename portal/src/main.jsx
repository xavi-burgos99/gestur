import React, { useCallback, useEffect, useRef, useState } from "react";
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
  left_hand_roll: "Mano izquierda · giro 3D",
  right_hand_roll: "Mano derecha · giro 3D",
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
const formatCount = (value) => new Intl.NumberFormat("es-ES").format(value);
const importFormats =
  ".zip,.glb,.gltf,.obj,.fbx,.stl,.ply,.dae,.3ds,.off,.x,.lwo,.ase,.dxf,.ac,.ms3d,.cob,.b3d";

function ModelArt() {
  return (
    <div className="model-art digital" aria-hidden="true">
      <IconCube size={76} stroke={1} />
    </div>
  );
}
function ImportButton({ upload, disabled, loading, variant, fullWidth }) {
  const resetRef = useRef(null);
  return (
    <FileButton
      onChange={(file) => {
        upload(file);
        resetRef.current?.();
      }}
      accept={importFormats}
      resetRef={resetRef}
      disabled={disabled}
    >
      {(props) => (
        <Button
          {...props}
          leftSection={<IconUpload size={18} />}
          disabled={disabled}
          loading={loading}
          variant={variant}
          fullWidth={fullWidth}
        >
          Subir modelo
        </Button>
      )}
    </FileButton>
  );
}
function ImportWarnings({ warnings = [] }) {
  if (!warnings.length) return null;
  return (
    <details className="import-warnings">
      <summary>
        {warnings.length === 1
          ? "1 detalle de la importación"
          : `${warnings.length} detalles de la importación`}
      </summary>
      <ul>
        {warnings.map((warning, index) => (
          <li key={index}>{warning}</li>
        ))}
      </ul>
    </details>
  );
}
function Models({ config, setConfig, notify }) {
  const [runtime, setRuntime] = useState({ online: false });
  useEffect(() => {
    let stopped = false;
    const poll = () =>
      api("runtime")
        .then((data) => !stopped && setRuntime(data))
        .catch(() => !stopped && setRuntime({ online: false }));
    poll();
    const timer = setInterval(poll, 3000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, []);
  const [models, setModels] = useState([]);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [job, setJob] = useState(null);
  const [restoring, setRestoring] = useState(true);
  const [jobError, setJobError] = useState("");
  const [decisionError, setDecisionError] = useState("");
  const [deciding, setDeciding] = useState(null);
  const [dragging, setDragging] = useState(false);
  const importRevision = useRef(0);
  const importMutating = useRef(false);
  const completedJob = useRef(null);
  const dragDepth = useRef(0);
  const pending = ["processing", "awaiting_decision"].includes(job?.state);
  const importDisabled = restoring || uploading || pending;
  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api("models");
      setModels(data.models);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    let stopped = false;
    let timer;
    async function poll() {
      const revision = importRevision.current;
      try {
        if (!importMutating.current) {
          const data = await api("imports/current");
          if (!stopped && revision === importRevision.current) {
            setJob(data.job);
            setJobError("");
          }
        }
      } catch (e) {
        if (!stopped && revision === importRevision.current)
          setJobError(e.message);
      } finally {
        if (!stopped) {
          setRestoring(false);
          timer = setTimeout(poll, 2000);
        }
      }
    }
    poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);
  useEffect(() => {
    if (job?.state === "completed" && completedJob.current !== job.id) {
      completedJob.current = job.id;
      load();
    }
  }, [job?.id, job?.state, load]);
  async function upload(file) {
    if (!file || importDisabled || importMutating.current) return;
    importRevision.current += 1;
    importMutating.current = true;
    setUploading(true);
    setDecisionError("");
    setJobError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const data = await api("models", { method: "POST", body: form });
      setJob(data.job);
    } catch (e) {
      notify(e.message, true);
    } finally {
      importMutating.current = false;
      setUploading(false);
    }
  }
  async function decide(simplify) {
    if (!job || importMutating.current) return;
    importRevision.current += 1;
    importMutating.current = true;
    setDeciding(simplify);
    setDecisionError("");
    try {
      const data = await api(`imports/${job.id}/decision`, {
        method: "POST",
        body: { simplify },
      });
      setJob(data.job);
    } catch (e) {
      setDecisionError(e.message);
    } finally {
      importMutating.current = false;
      setDeciding(null);
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
        id
          ? "Selección guardada. El visualizador cargará el modelo en unos segundos."
          : "La pantalla de bienvenida se mostrará en el dispositivo.",
      );
    } catch (e) {
      notify(e.message, true);
    } finally {
      setBusy(false);
    }
  }
  function drop(event) {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (importDisabled) return;
    const files = [...event.dataTransfer.files];
    if (files.length !== 1) {
      notify(
        "Sube un modelo cada vez. Si tiene varios archivos, júntalos en un ZIP.",
        true,
      );
      return;
    }
    upload(files[0]);
  }
  const proposal = job?.proposal;
  const empty = !loading && !error && models.length === 0;
  return (
    <>
      <SectionTitle
        eyebrow="TU COLECCIÓN"
        title="Modelos 3D"
        description="Tus piezas, listas para responder a cada movimiento."
        action={
          <ImportButton
            upload={upload}
            disabled={importDisabled}
            loading={uploading}
          />
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
              ? config.active_model
                ? "El modelo seleccionado se está mostrando en el dispositivo."
                : "El dispositivo muestra la bienvenida con el QR de este portal."
              : "El visor está aplicando la selección.")
          : "Puedes guardar cambios. Se aplicarán cuando el visualizador esté en marcha."}
      </Alert>
      {jobError && (
        <Alert
          color="yellow"
          mb="lg"
          title="Reconectando con la importación"
          role="status"
        >
          {jobError} Si ya había una importación en marcha, continúa en el
          dispositivo. Volveremos a comprobar su estado automáticamente.
        </Alert>
      )}
      {uploading && (
        <Paper withBorder p="lg" mb="xl" role="status">
          <Group wrap="nowrap">
            <Loader size="sm" />
            <div>
              <Text fw={600}>Enviando tu archivo</Text>
              <Text size="sm" c="dimmed">
                Mantén esta página abierta hasta que termine la subida.
              </Text>
            </div>
          </Group>
        </Paper>
      )}
      {job && !uploading && (
        <Paper
          withBorder
          p="lg"
          mb="xl"
          className={`import-status ${job.state}`}
          role="status"
          aria-live="polite"
        >
          <Group wrap="nowrap" align="flex-start">
            {job.state === "processing" ? (
              <Loader size="sm" mt={3} />
            ) : (
              <ThemeIcon
                variant="light"
                color={job.state === "failed" ? "red" : "teal"}
                radius="xl"
              >
                {job.state === "failed" ? (
                  <IconAlertCircle size={18} />
                ) : job.state === "completed" ? (
                  <IconCheck size={18} />
                ) : (
                  <IconCube size={18} />
                )}
              </ThemeIcon>
            )}
            <div className="import-status-copy">
              <Text fw={600}>
                {job.state === "completed"
                  ? `Modelo preparado: ${job.model?.name || "Nueva pieza"}`
                  : job.state === "failed"
                    ? "No se pudo importar el modelo"
                    : job.state === "awaiting_decision"
                      ? "Tu modelo tiene muchos triángulos"
                      : "Preparando tu modelo"}
              </Text>
              <Text size="sm" c="dimmed" mt={3}>
                {job.state === "failed"
                  ? job.error || job.message
                  : job.message}
              </Text>
              {job.state === "processing" && (
                <Text size="sm" c="dimmed" mt={5}>
                  La Raspberry Pi está trabajando. Puede tardar varios minutos;
                  puedes salir y volver a esta página.
                </Text>
              )}
              <ImportWarnings warnings={job.warnings} />
            </div>
          </Group>
        </Paper>
      )}
      {error && (
        <Alert color="red" mb="lg" title="No se pudo cargar la colección">
          {error}
          <Button mt="sm" variant="light" onClick={load}>
            Reintentar
          </Button>
        </Alert>
      )}
      {loading ? (
        <Loader aria-label="Cargando modelos" />
      ) : (
        <div
          onDragEnter={(event) => {
            event.preventDefault();
            if ([...event.dataTransfer.types].includes("Files")) {
              dragDepth.current += 1;
              if (!importDisabled) setDragging(true);
            }
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            dragDepth.current = Math.max(0, dragDepth.current - 1);
            if (!dragDepth.current) setDragging(false);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = importDisabled ? "none" : "copy";
          }}
          onDrop={drop}
          className={`collection-drop-area ${dragging ? "dragging" : ""}`}
        >
          {dragging && (
            <div className="drop-overlay" aria-hidden="true">
              Suelta tu modelo aquí
            </div>
          )}
          {empty ? (
            <Paper
              withBorder
              className="empty-collection"
              p={{ base: "xl", sm: 44 }}
            >
              <div className="empty-model-art" aria-hidden="true">
                <svg viewBox="0 0 160 180" fill="none">
                  <ellipse
                    cx="80"
                    cy="154"
                    rx="49"
                    ry="8"
                    fill="#173f35"
                    opacity=".07"
                  />
                  <path d="M80 19 131 49v63l-51 30-51-30V49Z" fill="#d4e4df" />
                  <path d="m80 19 51 30-51 31-51-31Z" fill="#e9f1ed" />
                  <path d="M80 80v62l51-30V49Z" fill="#8fb7a8" />
                  <path
                    d="M80 19 29 49v63l51 30 51-30V49Z"
                    stroke="#7a9e8e"
                    strokeWidth="1.5"
                  />
                  <path
                    d="m29 49 51 31 51-31M80 80v62"
                    stroke="#7a9e8e"
                    strokeWidth="1.5"
                  />
                </svg>
              </div>
              <div className="empty-collection-copy">
                <Text className="eyebrow">TODO EMPIEZA CON UNA PIEZA</Text>
                <Title order={2}>Tu primera pieza va aquí</Title>
                <Text c="dimmed" mt="sm" maw={470}>
                  La colección está vacía. Sube un modelo 3D o arrástralo aquí y
                  Gestur lo preparará para el visualizador.
                </Text>
                <Group mt="xl">
                  <ImportButton
                    upload={upload}
                    disabled={importDisabled}
                    loading={uploading}
                  />
                  <Text size="xs" c="dimmed">
                    Archivo 3D o ZIP · Hasta 100 MB
                  </Text>
                </Group>
              </div>
            </Paper>
          ) : (
            <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }} spacing="xl">
              {models.map((model) => (
                <Paper
                  className={`model-card ${config.active_model === model.id ? "selected" : ""}`}
                  key={model.id}
                  withBorder
                >
                  <ModelArt />
                  <Stack p="xl" gap="md">
                    <Group
                      justify="space-between"
                      align="flex-start"
                      wrap="nowrap"
                    >
                      <Title order={3} className="model-name">
                        {model.name}
                      </Title>
                      <Badge
                        variant="light"
                        color={
                          config.active_model === model.id ? "teal" : "gray"
                        }
                        className="model-badge"
                      >
                        {runtime.online && runtime.rendered_model === model.id
                          ? "En pantalla"
                          : config.active_model === model.id
                            ? "Seleccionado"
                            : model.sourceFormat || model.format}
                      </Badge>
                    </Group>
                    <Text c="dimmed" size="sm">
                      {model.triangles != null &&
                        `${formatCount(model.triangles)} triángulos · `}
                      {(model.size / 1024 / 1024).toFixed(1)} MB
                    </Text>
                    <ImportWarnings warnings={model.warnings} />
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
              {!error && (
                <Paper className="import-guide" p="xl" withBorder>
                  <ThemeIcon variant="light" size={48} radius="xl">
                    <IconUpload size={24} />
                  </ThemeIcon>
                  <Title order={3} mt="xl">
                    Suma una nueva pieza
                  </Title>
                  <Text c="dimmed" mt="sm" mb="xl" size="sm">
                    Arrastra aquí un archivo 3D o un ZIP con el modelo y sus
                    texturas.
                  </Text>
                  <ImportButton
                    upload={upload}
                    disabled={importDisabled}
                    loading={uploading}
                    variant="light"
                    fullWidth
                  />
                </Paper>
              )}
            </SimpleGrid>
          )}
        </div>
      )}
      <Paper withBorder mt="xl" p="lg">
        <Group wrap="nowrap" align="flex-start">
          <ThemeIcon variant="light" radius="xl">
            <IconUpload size={19} />
          </ThemeIcon>
          <div>
            <Text fw={600} size="sm">
              Un archivo, y nos encargamos del resto
            </Text>
            <Text size="sm" c="dimmed" mt={4}>
              GLB, glTF, OBJ, FBX, STL, PLY, COLLADA y más. Si hay texturas o
              archivos auxiliares, inclúyelos en un ZIP junto al modelo. Gestur
              convierte el modelo y busca sus texturas aunque las rutas hayan
              cambiado.
            </Text>
            <Text size="xs" c="dimmed" mt={8}>
              Si una textura falta o hay varias posibles, te avisaremos. Hasta
              100 MB por subida · 250 MB al descomprimir · 500 archivos.
            </Text>
          </div>
        </Group>
      </Paper>
      <Paper className="note-panel" mt="lg" p="lg">
        <Group justify="space-between" align="center">
          <Group wrap="nowrap" align="flex-start" className="welcome-note">
            <IconDeviceDesktop size={23} className="fixed-icon" />
            <div>
              <Text fw={600} size="sm">
                {config.active_model
                  ? "Elige qué muestra tu dispositivo"
                  : "Tu dispositivo está listo para empezar"}
              </Text>
              <Text size="sm" c="dimmed">
                Sin un modelo seleccionado, el visualizador muestra una figura
                3D y un QR para abrir este portal.
              </Text>
            </div>
          </Group>
          {config.active_model && (
            <Button
              variant="white"
              disabled={busy}
              onClick={() => activate(null)}
              leftSection={<IconDeviceDesktop size={17} />}
            >
              Mostrar bienvenida
            </Button>
          )}
        </Group>
      </Paper>
      <Modal
        opened={job?.state === "awaiting_decision" && !!proposal}
        onClose={() => {}}
        withCloseButton={false}
        closeOnClickOutside={false}
        closeOnEscape={false}
        title="¿Hacemos tu modelo más ligero?"
        centered
        size="lg"
      >
        {proposal && (
          <Stack gap="lg">
            <Text size="sm" c="dimmed">
              Hemos detectado un número muy elevado de triángulos. Reducirlos
              puede ayudar a que el movimiento sea más fluido en la Raspberry
              Pi.
            </Text>
            <div className="simplify-comparison">
              <div>
                <Text size="xs" c="dimmed" fw={600}>
                  MODELO ACTUAL
                </Text>
                <Text className="triangle-count">
                  {formatCount(proposal.originalTriangles)}
                </Text>
                <Text size="sm" c="dimmed">
                  triángulos
                </Text>
              </div>
              <IconArrowRight
                size={24}
                className="simplify-arrow"
                aria-hidden="true"
              />
              <div>
                <Text size="xs" c="dimmed" fw={600}>
                  TRAS REDUCIR, APROX.
                </Text>
                <Text className="triangle-count reduced">
                  {formatCount(proposal.targetTriangles)}
                </Text>
                <Text size="sm" c="dimmed">
                  triángulos
                </Text>
              </div>
            </div>
            <Badge size="lg" variant="light">
              {formatCount(proposal.reductionPercent)} % menos triángulos
            </Badge>
            <Text size="sm">
              La Raspberry Pi realizará la reducción; puede tardar varios
              minutos. La simplificación puede cambiar algunos detalles de la
              geometría.
            </Text>
            <ImportWarnings warnings={job.warnings} />
            {decisionError && (
              <Alert color="red" title="No se pudo guardar la decisión">
                {decisionError}
              </Alert>
            )}
            <div className="simplify-actions">
              <Button
                variant="default"
                onClick={() => decide(false)}
                loading={deciding === false}
                disabled={deciding !== null}
              >
                Continuar sin simplificar
              </Button>
              <Button
                onClick={() => decide(true)}
                loading={deciding === true}
                disabled={deciding !== null}
                leftSection={<IconBolt size={17} />}
              >
                Reducir polígonos
              </Button>
            </div>
          </Stack>
        )}
      </Modal>
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
