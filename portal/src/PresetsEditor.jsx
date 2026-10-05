import React, { useEffect, useRef, useState } from "react";
import {
  ActionIcon,
  Alert,
  Button,
  Group,
  Modal,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Text,
  TextInput,
  Title,
  Tooltip,
} from "@mantine/core";
import { IconCheck, IconRefresh, IconTrash } from "@tabler/icons-react";
import { presetName, presetNameError, presetSaveBody } from "./presets.mjs";

export default function PresetsEditor({
  api,
  draft,
  dirty,
  disabled,
  onApply,
  onBusyChange,
  notify,
}) {
  const [presets, setPresets] = useState([]);
  const [selected, setSelected] = useState(null);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState("");
  const [nameError, setNameError] = useState("");
  const [dialog, setDialog] = useState(null);
  const [reload, setReload] = useState(0);
  const busyRef = useRef(false);
  const lastDialog = useRef({ action: "apply", name: "" });
  const displayedDialog = dialog ?? lastDialog.current;
  const normalizedName = presetName(name);
  const overwrites = presets.some((preset) => preset.name === normalizedName);
  const locked = disabled || loading || !loaded || !!busy;

  function openDialog(action, name) {
    const next = { action, name };
    lastDialog.current = next;
    setDialog(next);
  }

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    api("presets", { signal: controller.signal })
      .then(({ presets: list }) => {
        if (!controller.signal.aborted) {
          setPresets(list);
          setLoaded(true);
          setSelected((current) =>
            list.some((preset) => preset.name === current) ? current : null,
          );
        }
      })
      .catch((failure) => {
        if (!controller.signal.aborted) setError(failure.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [api, reload]);

  async function perform(action, request, complete) {
    if (disabled || busyRef.current) return;
    busyRef.current = true;
    setBusy(action);
    onBusyChange(true);
    setError("");
    try {
      const result = await request();
      complete(result);
    } catch (failure) {
      setError(failure.message);
    } finally {
      busyRef.current = false;
      setBusy(null);
      onBusyChange(false);
    }
  }

  function save() {
    const invalid = presetNameError(name);
    setNameError(invalid);
    if (invalid) return;
    const body = presetSaveBody(name, draft);
    perform(
      "save",
      () => api("presets", { method: "PUT", body }),
      ({ preset, presets: list, overwritten }) => {
        setPresets(list);
        setSelected(preset.name);
        setName(preset.name);
        notify(
          overwritten
            ? "Parámetros preestablecidos sobrescritos."
            : "Parámetros preestablecidos guardados.",
        );
      },
    );
  }

  function apply(target) {
    perform(
      "apply",
      () => api("presets/apply", { method: "POST", body: { name: target } }),
      ({ config }) => {
        onApply(config);
        setDialog(null);
        notify("Parámetros preestablecidos aplicados.");
      },
    );
  }

  function remove(target) {
    perform(
      "delete",
      () => api("presets", { method: "DELETE", body: { name: target } }),
      ({ presets: list }) => {
        setPresets(list);
        setSelected(null);
        if (normalizedName === target) setName("");
        setDialog(null);
        notify("Parámetros preestablecidos eliminados.");
      },
    );
  }

  return (
    <>
      <Paper withBorder p="lg" mb="xl">
        <Title order={3}>Parámetros preestablecidos</Title>
        <Text size="sm" c="dimmed" mt={4} mb="md">
          Guarda todos los parámetros de esta pestaña, incluidos los cambios sin
          guardar.
        </Text>
        {error && !dialog && (
          <Alert color="red" mb="md">
            <Group justify="space-between">
              <Text size="sm">{error}</Text>
              <Button
                size="xs"
                variant="subtle"
                color="red"
                leftSection={<IconRefresh size={14} />}
                disabled={disabled || loading || !!busy}
                onClick={() => setReload((value) => value + 1)}
              >
                Actualizar lista
              </Button>
            </Group>
          </Alert>
        )}
        <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="xl">
          <Group align="flex-end" wrap="nowrap" gap="sm">
            <Select
              label="Parámetros guardados"
              placeholder={
                loading
                  ? "Cargando…"
                  : presets.length
                    ? "Selecciona parámetros preestablecidos"
                    : "No hay parámetros preestablecidos"
              }
              data={presets.map((preset) => preset.name)}
              value={selected}
              onChange={(value) => {
                setSelected(value);
                if (value !== null) setName(value);
                setNameError("");
              }}
              disabled={locked || !presets.length}
              searchable
              nothingFoundMessage="No hay coincidencias"
              style={{ flex: 1, minWidth: 0 }}
            />
            <Button
              variant="light"
              disabled={locked || !selected}
              loading={busy === "apply" && !dialog}
              onClick={() => {
                setError("");
                if (dirty) openDialog("apply", selected);
                else apply(selected);
              }}
            >
              Cargar
            </Button>
            <Tooltip label="Eliminar parámetros">
              <ActionIcon
                size="input-sm"
                variant="subtle"
                color="gray"
                aria-label="Eliminar parámetros"
                disabled={locked || !selected}
                onClick={() => {
                  setError("");
                  openDialog("delete", selected);
                }}
              >
                <IconTrash size={18} />
              </ActionIcon>
            </Tooltip>
          </Group>
          <Group align="flex-end" gap="sm">
            <TextInput
              label="Nombre"
              placeholder="Nuevos parámetros"
              value={name}
              maxLength={80}
              error={nameError}
              disabled={locked}
              onChange={(event) => {
                setName(event.currentTarget.value);
                setNameError("");
              }}
              style={{ flex: 1, minWidth: 150 }}
            />
            <Button
              leftSection={<IconCheck size={16} />}
              disabled={locked || !normalizedName}
              loading={busy === "save"}
              onClick={save}
            >
              {overwrites ? "Sobrescribir" : "Guardar"}
            </Button>
          </Group>
        </SimpleGrid>
        <Text size="xs" c="dimmed" mt="sm">
          Guardar parámetros preestablecidos no aplica cambios. Cargarlos los
          aplica al dispositivo.
        </Text>
      </Paper>
      <Modal
        opened={!!dialog}
        onClose={() => !busy && setDialog(null)}
        title={
          displayedDialog.action === "delete"
            ? "Eliminar parámetros"
            : "Cargar parámetros"
        }
        centered
        closeOnClickOutside={!busy}
        closeOnEscape={!busy}
        withCloseButton={!busy}
      >
        <Stack>
          <Text size="sm">
            {displayedDialog.action === "delete"
              ? `Se eliminará «${displayedDialog.name}». Los parámetros del dispositivo no cambiarán.`
              : `Se aplicarán los parámetros de «${displayedDialog.name}» y se sustituirán los cambios sin guardar.`}
          </Text>
          {error && <Alert color="red">{error}</Alert>}
          <Group justify="flex-end">
            <Button
              variant="default"
              disabled={!!busy}
              onClick={() => setDialog(null)}
            >
              Cancelar
            </Button>
            <Button
              color={displayedDialog.action === "delete" ? "red" : undefined}
              loading={!!busy}
              disabled={disabled || !dialog}
              onClick={() =>
                displayedDialog.action === "delete"
                  ? remove(displayedDialog.name)
                  : apply(displayedDialog.name)
              }
            >
              {displayedDialog.action === "delete"
                ? "Eliminar parámetros"
                : "Cargar parámetros"}
            </Button>
          </Group>
        </Stack>
      </Modal>
    </>
  );
}
