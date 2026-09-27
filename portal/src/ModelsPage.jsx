import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Group,
  Stack,
  Title,
  Text,
  Button,
  Paper,
  Badge,
  SimpleGrid,
  Select,
  TextInput,
  Modal,
  Menu,
  Alert,
  Loader,
  FileButton,
  Divider,
  ActionIcon,
  ThemeIcon,
} from "@mantine/core";
import {
  IconCube,
  IconArrowUpRight,
  IconUpload,
  IconCheck,
  IconArrowRight,
  IconAlertCircle,
  IconDeviceDesktop,
  IconBolt,
  IconDotsVertical,
  IconPencil,
  IconTrash,
  IconRotateClockwise,
} from "@tabler/icons-react";
import { api } from "./api.mjs";
import SectionTitle from "./SectionTitle.jsx";

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
          ? "1 aviso de importación"
          : `${warnings.length} avisos de importación`}
      </summary>
      <ul>
        {warnings.map((warning, index) => (
          <li key={index}>{warning}</li>
        ))}
      </ul>
    </details>
  );
}
export default function Models({ config, setConfig, notify }) {
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
  const [catalogImport, setCatalogImport] = useState(null);
  const [restoring, setRestoring] = useState(true);
  const [jobError, setJobError] = useState("");
  const [decisionError, setDecisionError] = useState("");
  const [deciding, setDeciding] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [modelDialog, setModelDialog] = useState(null);
  const [modelName, setModelName] = useState("");
  const [modelUrl, setModelUrl] = useState("");
  const [orientation, setOrientation] = useState({ x: 0, y: 0, z: 0 });
  const [modelError, setModelError] = useState("");
  const importRevision = useRef(0);
  const importMutating = useRef(false);
  const completedJob = useRef(null);
  const dragDepth = useRef(0);
  const selectionRevision = useRef(0);
  const selectionMutating = useRef(false);
  const catalogRequest = useRef(0);
  const pending = ["processing", "awaiting_decision"].includes(job?.state);
  const importDisabled = restoring || uploading || pending || busy || loading;
  const load = useCallback(async () => {
    const revision = selectionRevision.current;
    const importedJob = completedJob.current;
    const request = ++catalogRequest.current;
    setLoading(true);
    setError("");
    try {
      const data = await api("models");
      if (
        request !== catalogRequest.current ||
        revision !== selectionRevision.current ||
        selectionMutating.current
      )
        return;
      setModels(data.models);
      setCatalogImport(importedJob);
      setConfig((current) => ({ ...current, active_model: data.active }));
    } catch (e) {
      if (
        request === catalogRequest.current &&
        revision === selectionRevision.current
      )
        setError(e.message);
    } finally {
      if (request === catalogRequest.current) setLoading(false);
    }
  }, [setConfig]);
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
    if (
      !file ||
      importDisabled ||
      importMutating.current ||
      selectionMutating.current
    )
      return;
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
    if (importDisabled || selectionMutating.current) return;
    selectionRevision.current += 1;
    selectionMutating.current = true;
    setBusy(true);
    try {
      const result = await api("models/active", {
        method: "PUT",
        body: { id },
      });
      setConfig(result.config);
      notify("Modelo seleccionado. Se mostrará en unos segundos.");
    } catch (e) {
      notify(e.message, true);
    } finally {
      selectionRevision.current += 1;
      selectionMutating.current = false;
      setBusy(false);
    }
  }
  function openModelDialog(kind, model) {
    if (importDisabled || selectionMutating.current) return;
    setModelDialog({ kind, model });
    setModelName(model.name);
    setModelUrl(model.url || "");
    setOrientation({ x: 0, y: 0, z: 0, ...model.orientation });
    setModelError("");
  }
  function closeModelDialog() {
    if (!selectionMutating.current) setModelDialog(null);
  }
  async function saveModel(event) {
    event.preventDefault();
    if (!modelDialog || importDisabled || selectionMutating.current) return;
    const name = modelName.trim();
    if (!name || name.length > 100) {
      setModelError("Escribe un nombre de entre 1 y 100 caracteres.");
      return;
    }
    const url = modelUrl.trim();
    if (new TextEncoder().encode(url).length > 2048) {
      setModelError("La URL es demasiado larga.");
      return;
    }
    if (url) {
      try {
        const parsed = new URL(url);
        if (
          !/^https?:\/\//i.test(url) ||
          !["http:", "https:"].includes(parsed.protocol) ||
          parsed.username ||
          parsed.password ||
          url.length > 2048 ||
          /[\s\\\x00-\x1f\x7f]/u.test(url)
        )
          throw new Error();
      } catch {
        setModelError(
          "Escribe una URL completa que empiece por http:// o https://, sin usuario ni contraseña.",
        );
        return;
      }
    }
    await changeModel("PATCH", {
      id: modelDialog.model.id,
      name,
      orientation,
      url: url || null,
    });
  }
  async function changeModel(method, body) {
    if (importDisabled || selectionMutating.current) return;
    selectionRevision.current += 1;
    selectionMutating.current = true;
    setBusy(true);
    setModelError("");
    let changed = false;
    try {
      const result = await api("models", { method, body });
      setModels((current) =>
        method === "DELETE"
          ? current.filter((model) => model.id !== body.id)
          : current.map((model) =>
              model.id === body.id ? result.model : model,
            ),
      );
      setConfig(result.config);
      setModelDialog(null);
      notify(method === "DELETE" ? "Modelo eliminado." : "Modelo actualizado.");
      changed = true;
    } catch (e) {
      setModelError(e.message);
    } finally {
      selectionRevision.current += 1;
      selectionMutating.current = false;
      setBusy(false);
    }
    if (changed) await load();
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
  const showImportStatus =
    job &&
    !uploading &&
    (job.state !== "completed" ||
      catalogImport !== job.id ||
      models.some((model) => model.id === job.model?.id));
  return (
    <>
      <SectionTitle
        title="Modelos 3D"
        description="Sube modelos 3D para mostrarlos en la pantalla del dispositivo."
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
          : "Los cambios se aplicarán cuando el visor esté conectado."}
      </Alert>
      {jobError && (
        <Alert
          color="yellow"
          mb="lg"
          title="Reconectando con la importación"
          role="status"
        >
          {jobError} El estado se actualizará al recuperar la conexión.
        </Alert>
      )}
      {uploading && (
        <Paper withBorder p="lg" mb="xl" role="status">
          <Group wrap="nowrap">
            <Loader size="sm" />
            <div>
              <Text fw={600}>Subiendo archivo</Text>
              <Text size="sm" c="dimmed">
                Mantén esta página abierta hasta que termine la subida.
              </Text>
            </div>
          </Group>
        </Paper>
      )}
      {showImportStatus && (
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
                  ? `Modelo importado: ${job.model?.name || "Sin nombre"}`
                  : job.state === "failed"
                    ? "No se pudo importar el modelo"
                    : job.state === "awaiting_decision"
                      ? "Reducción de triángulos pendiente"
                      : "Importando modelo"}
              </Text>
              <Text size="sm" c="dimmed" mt={3}>
                {job.state === "failed"
                  ? job.error || job.message
                  : job.message}
              </Text>
              {job.state === "processing" && (
                <Text size="sm" c="dimmed" mt={5}>
                  Puede tardar varios minutos. Puedes cerrar esta página y
                  volver más tarde.
                </Text>
              )}
              <ImportWarnings warnings={job.warnings} />
            </div>
          </Group>
        </Paper>
      )}
      {error && (
        <Alert color="red" mb="lg" title="No se pudieron cargar los modelos">
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
              Suelta el archivo aquí
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
                <Title order={2}>No hay modelos</Title>
                <Text c="dimmed" mt="sm" maw={470}>
                  Sube un archivo 3D o arrástralo aquí.
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
                  <div className="model-preview">
                    <ModelArt />
                    <Menu position="bottom-end" shadow="md" width={190}>
                      <Menu.Target>
                        <ActionIcon
                          className="model-actions"
                          variant="white"
                          color="dark"
                          size="lg"
                          aria-label={`Opciones de ${model.name}`}
                          disabled={importDisabled}
                        >
                          <IconDotsVertical size={19} stroke={1.7} />
                        </ActionIcon>
                      </Menu.Target>
                      <Menu.Dropdown>
                        <Menu.Item
                          leftSection={<IconPencil size={17} stroke={1.7} />}
                          onClick={() => openModelDialog("edit", model)}
                          disabled={importDisabled}
                        >
                          Ajustar modelo
                        </Menu.Item>
                        <Menu.Divider />
                        <Menu.Item
                          color="red"
                          leftSection={<IconTrash size={17} stroke={1.7} />}
                          onClick={() => openModelDialog("delete", model)}
                          disabled={importDisabled}
                        >
                          Eliminar
                        </Menu.Item>
                      </Menu.Dropdown>
                    </Menu>
                  </div>
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
                      disabled={
                        importDisabled || config.active_model === model.id
                      }
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
                    Añadir modelo
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
              Formatos admitidos
            </Text>
            <Text size="sm" c="dimmed" mt={4}>
              GLB, glTF, OBJ, FBX, STL, PLY, COLLADA y más. Si hay texturas o
              archivos auxiliares, inclúyelos en un ZIP junto al modelo.
            </Text>
            <Text size="xs" c="dimmed" mt={8}>
              Hasta 100 MB por subida · 250 MB al descomprimir · 500 archivos.
            </Text>
          </div>
        </Group>
      </Paper>
      {empty && (
        <Paper className="note-panel" mt="lg" p="lg">
          <Group wrap="nowrap" align="flex-start" className="welcome-note">
            <IconDeviceDesktop size={23} className="fixed-icon" />
            <div>
              <Text fw={600} size="sm">
                Pantalla de bienvenida
              </Text>
              <Text size="sm" c="dimmed">
                Mientras no haya modelos, la pantalla muestra un QR para abrir
                el portal. El primer modelo que subas se mostrará
                automáticamente.
              </Text>
            </div>
          </Group>
        </Paper>
      )}
      <Modal
        opened={modelDialog?.kind === "edit"}
        onClose={closeModelDialog}
        closeOnClickOutside={!busy}
        closeOnEscape={!busy}
        withCloseButton={!busy}
        title="Ajustar modelo"
        centered
        size="md"
      >
        <form onSubmit={saveModel}>
          <Stack gap="lg">
            <TextInput
              label="Nombre"
              value={modelName}
              onChange={(event) => setModelName(event.currentTarget.value)}
              maxLength={100}
              required
              disabled={busy || pending}
              data-autofocus
            />
            <TextInput
              label="URL"
              description="Opcional. Enlace del QR que acompaña al modelo."
              placeholder="https://..."
              type="url"
              value={modelUrl}
              onChange={(event) => setModelUrl(event.currentTarget.value)}
              maxLength={2048}
              disabled={busy || pending}
              autoComplete="url"
              spellCheck={false}
            />
            <div>
              <Text fw={600} size="sm">
                Orientación inicial
              </Text>
              <Text c="dimmed" size="sm" mt={4}>
                Giros respecto al archivo original.
              </Text>
              <SimpleGrid cols={3} spacing="sm" mt="sm">
                {["x", "y", "z"].map((axis) => (
                  <Select
                    key={axis}
                    label={`Eje ${axis.toUpperCase()}`}
                    data={[0, 90, 180, 270].map((degrees) => ({
                      value: String(degrees),
                      label: `${degrees}°`,
                    }))}
                    value={String(orientation[axis])}
                    onChange={(value) =>
                      value !== null &&
                      setOrientation((current) => ({
                        ...current,
                        [axis]: Number(value),
                      }))
                    }
                    allowDeselect={false}
                    disabled={busy || pending}
                    comboboxProps={{ withinPortal: false }}
                  />
                ))}
              </SimpleGrid>
              <Button
                variant="subtle"
                size="xs"
                mt="xs"
                px={0}
                leftSection={<IconRotateClockwise size={15} stroke={1.7} />}
                disabled={
                  busy ||
                  pending ||
                  Object.values(orientation).every((value) => value === 0)
                }
                onClick={() => setOrientation({ x: 0, y: 0, z: 0 })}
              >
                Restablecer orientación
              </Button>
            </div>
            {pending && (
              <Text c="dimmed" size="sm">
                Espera a que termine la importación para guardar.
              </Text>
            )}
            {modelError && <Alert color="red">{modelError}</Alert>}
            <Group justify="flex-end">
              <Button
                variant="default"
                onClick={closeModelDialog}
                disabled={busy}
              >
                Cancelar
              </Button>
              <Button
                type="submit"
                loading={busy}
                disabled={importDisabled || !modelName.trim()}
              >
                Guardar
              </Button>
            </Group>
          </Stack>
        </form>
      </Modal>
      <Modal
        opened={modelDialog?.kind === "delete"}
        onClose={closeModelDialog}
        closeOnClickOutside={!busy}
        closeOnEscape={!busy}
        withCloseButton={!busy}
        title="Eliminar modelo"
        centered
        size="md"
      >
        <Stack gap="lg">
          <Text size="sm">
            Se eliminará «{modelDialog?.model.name}» y sus archivos.
          </Text>
          {models.length === 1 ? (
            <Text size="sm" c="dimmed">
              Es el último modelo. El dispositivo mostrará la pantalla de
              bienvenida.
            </Text>
          ) : (
            config.active_model === modelDialog?.model.id && (
              <Text size="sm" c="dimmed">
                Está seleccionado. Se mostrará otro modelo de la biblioteca.
              </Text>
            )
          )}
          {pending && (
            <Text c="dimmed" size="sm">
              Espera a que termine la importación para eliminarlo.
            </Text>
          )}
          {modelError && <Alert color="red">{modelError}</Alert>}
          <Group justify="flex-end">
            <Button
              variant="default"
              onClick={closeModelDialog}
              disabled={busy}
            >
              Cancelar
            </Button>
            <Button
              color="red"
              loading={busy}
              disabled={importDisabled}
              leftSection={<IconTrash size={17} stroke={1.7} />}
              onClick={() =>
                modelDialog &&
                changeModel("DELETE", { id: modelDialog.model.id })
              }
            >
              Eliminar modelo
            </Button>
          </Group>
        </Stack>
      </Modal>
      <Modal
        opened={job?.state === "awaiting_decision" && !!proposal}
        onClose={() => {}}
        withCloseButton={false}
        closeOnClickOutside={false}
        closeOnEscape={false}
        title="Reducir triángulos"
        centered
        size="lg"
      >
        {proposal && (
          <Stack gap="lg">
            <Text size="sm" c="dimmed">
              El modelo tiene un número elevado de triángulos. Reducirlos puede
              mejorar la fluidez en el dispositivo.
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
              La reducción se realiza en la Raspberry Pi y puede tardar varios
              minutos. Algunos detalles del modelo pueden perderse.
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
                Reducir triángulos
              </Button>
            </div>
          </Stack>
        )}
      </Modal>
    </>
  );
}
