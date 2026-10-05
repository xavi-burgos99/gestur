import React, { useEffect, useRef, useState } from "react";
import {
  Group,
  Stack,
  Title,
  Text,
  Button,
  Paper,
  SimpleGrid,
  NumberInput,
  Select,
  Slider,
  Switch,
  Accordion,
  Box,
} from "@mantine/core";
import { IconCheck } from "@tabler/icons-react";
import { parametersEqual } from "./presets.mjs";
import { api } from "./api.mjs";
import SectionTitle from "./SectionTitle.jsx";
import ControlsEditor from "./ControlsEditor.jsx";
import PresetsEditor from "./PresetsEditor.jsx";

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
export default function Parameters({ config, setConfig, defaults, notify }) {
  const [draft, setDraft] = useState(() => structuredClone(config));
  const [saving, setSaving] = useState(false);
  const [presetBusy, setPresetBusy] = useState(false);
  const locked = saving || presetBusy;
  const previousConfig = useRef(config);
  useEffect(() => {
    const previous = previousConfig.current;
    const parametersUnchanged = parametersEqual(previous, config);
    setDraft((current) =>
      parametersUnchanged
        ? { ...current, active_model: config.active_model }
        : structuredClone(config),
    );
    previousConfig.current = config;
  }, [config]);
  const dirty = !parametersEqual(draft, config);
  const update = (group, key, value) =>
    !locked &&
    setDraft((d) => ({ ...d, [group]: { ...d[group], [key]: value } }));
  const mapping = (i, changes) =>
    !locked &&
    setDraft((d) => ({
      ...d,
      controls: {
        ...d.controls,
        mappings: d.controls.mappings.map((m, n) =>
          n === i ? { ...m, ...changes } : m,
        ),
      },
    }));
  async function save() {
    if (locked) return;
    setSaving(true);
    try {
      const d = await api("config", { method: "PUT", body: draft });
      setConfig(d.config);
      notify("Parámetros guardados.");
    } catch (e) {
      notify(e.message, true);
    } finally {
      setSaving(false);
    }
  }
  return (
    <>
      <SectionTitle
        title="Parámetros"
        description="Ajusta los controles y la visualización del modelo."
        action={
          <Button
            leftSection={<IconCheck size={18} />}
            disabled={!dirty || presetBusy}
            loading={saving}
            onClick={save}
          >
            {dirty ? "Guardar cambios" : "Sin cambios"}
          </Button>
        }
      />
      <PresetsEditor
        api={api}
        draft={draft}
        dirty={dirty}
        disabled={saving}
        onBusyChange={setPresetBusy}
        onApply={(applied) => {
          setDraft(structuredClone(applied));
          setConfig(applied);
        }}
        notify={notify}
      />
      <fieldset className="parameter-fields" disabled={locked}>
        <SimpleGrid cols={{ base: 1, md: 2, xl: 4 }} mb="xl">
          <Paper p="xl" withBorder>
            <Title order={3} mb="lg">
              Seguimiento
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
            <Title order={3} mb="lg">
              Luz ambiente
            </Title>
            <Stack gap="lg">
              <Select
                aria-label="Luz ambiente"
                value={draft.render.ambient_light ?? "none"}
                data={[
                  { value: "none", label: "Ninguna" },
                  { value: "studio", label: "Estudio" },
                  { value: "gallery", label: "Galería" },
                  { value: "sunset", label: "Atardecer" },
                  { value: "rim", label: "Contraluz" },
                ]}
                allowDeselect={false}
                onChange={(value) =>
                  value !== null && update("render", "ambient_light", value)
                }
              />
              <Box>
                <Group justify="space-between" mb="xs">
                  <Text size="sm" fw={500}>
                    Exposición
                  </Text>
                  <Text size="sm" c="dimmed">
                    {draft.render.exposure ?? 50} %
                  </Text>
                </Group>
                <Slider
                  min={0}
                  max={100}
                  step={5}
                  value={draft.render.exposure ?? 50}
                  onChange={(value) =>
                    update("render", "exposure", Math.round(value / 5) * 5)
                  }
                  thumbLabel="Exposición"
                  thumbValueText={(value) => `${value} %`}
                  label={(value) => `${value} %`}
                  marks={[
                    { value: 0, label: "0 %" },
                    { value: 50, label: "50 %" },
                    { value: 100, label: "100 %" },
                  ]}
                  mx={5}
                  mb="xl"
                />
              </Box>
            </Stack>
          </Paper>
          <Paper p="xl" withBorder>
            <Title order={3} mb="lg">
              Modo de espera
            </Title>
            <Stack gap="md">
              <Select
                aria-label="Modo de espera"
                value={draft.controls.idle_mode ?? "return"}
                data={[
                  { value: "hold", label: "Mantener posición" },
                  { value: "return", label: "Volver a origen" },
                  { value: "float", label: "Flotante" },
                ]}
                allowDeselect={false}
                onChange={(value) =>
                  value !== null && update("controls", "idle_mode", value)
                }
              />
              {draft.controls.idle_mode === "hold" ? (
                <Text c="dimmed" size="sm">
                  Conserva la última posición al perder el gesto.
                </Text>
              ) : (
                <Numeric
                  label="Tiempo de espera"
                  description="Sin detectar un gesto activo."
                  value={draft.controls.reset_timeout_seconds}
                  onChange={(v) =>
                    update("controls", "reset_timeout_seconds", v)
                  }
                  max={30}
                  step={0.5}
                  suffix=" s"
                />
              )}
            </Stack>
          </Paper>
        </SimpleGrid>
        <ControlsEditor draft={draft} update={update} mapping={mapping} />
        <Accordion mt="xl" variant="separated">
          <Accordion.Item value="advanced">
            <Accordion.Control>Captura y renderizado</Accordion.Control>
            <Accordion.Panel>
              <SimpleGrid cols={{ base: 1, sm: 3 }}>
                <Numeric
                  label="Reconocimiento de cabeza y cuerpo"
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
                  label="Fotogramas por segundo"
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
                {(draft.controls.idle_mode ?? "return") === "return" && (
                  <Numeric
                    label="Duración de regreso a origen"
                    suffix=" s"
                    value={draft.controls.reset_duration_seconds}
                    onChange={(v) =>
                      update("controls", "reset_duration_seconds", v)
                    }
                    min={0.1}
                    max={10}
                    step={0.1}
                  />
                )}
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
                Los cambios se aplican al guardar. El seguimiento puede pausarse
                unos segundos al cambiar la cámara.
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
            Restaurar valores predeterminados
          </Button>
          <Button
            disabled={!dirty || presetBusy}
            loading={saving}
            onClick={save}
          >
            Guardar cambios
          </Button>
        </Group>
      </fieldset>
    </>
  );
}
