import React, { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Group,
  Loader,
  Paper,
  Select,
  Stack,
  Text,
  Title,
} from "@mantine/core";
import { api } from "./api.mjs";

const sizes = [
  { value: "very_small", label: "Muy pequeño" },
  { value: "small", label: "Pequeño" },
  { value: "default", label: "Por defecto" },
  { value: "large", label: "Grande" },
  { value: "very_large", label: "Muy grande" },
];

export default function ScreenSettings({ notify }) {
  const [screen, setScreen] = useState(null);
  const [saved, setSaved] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function load() {
    try {
      const result = await api("config");
      setScreen(result.config.screen);
      setSaved(result.config.screen);
      setError("");
    } catch (failure) {
      setError(failure.message);
    }
  }
  useEffect(() => {
    load();
  }, []);
  async function save() {
    setBusy(true);
    try {
      const result = await api("screen", { method: "PUT", body: screen });
      setScreen(result.config.screen);
      setSaved(result.config.screen);
      notify("Ajustes de pantalla guardados.");
    } catch (failure) {
      notify(failure.message, true);
    } finally {
      setBusy(false);
    }
  }
  const changed = JSON.stringify(screen) !== JSON.stringify(saved);
  return (
    <Paper withBorder p={{ base: "lg", sm: 32 }} maw={820} mb="xl">
      <Title order={2} mb="sm">
        Ajustes de pantalla
      </Title>
      <Text size="sm" c="dimmed" mb="lg">
        Se aplican a la pantalla del dispositivo al guardar.
      </Text>
      {error ? (
        <Alert color="red">
          {error}
          <Button variant="subtle" onClick={load}>
            Reintentar
          </Button>
        </Alert>
      ) : !screen ? (
        <Loader />
      ) : (
        <Stack gap="lg">
          <Select
            label="Orientación de la pantalla"
            value={String(screen.orientation)}
            data={[0, 90, 180, 270].map((value) => ({
              value: String(value),
              label: `${value}º`,
            }))}
            allowDeselect={false}
            disabled={busy}
            onChange={(value) =>
              value !== null &&
              setScreen({ ...screen, orientation: Number(value) })
            }
          />
          <Select
            label="Tamaño de contenido"
            description="Textos y códigos QR."
            data={sizes}
            value={screen.content_size}
            allowDeselect={false}
            disabled={busy}
            onChange={(value) =>
              value && setScreen({ ...screen, content_size: value })
            }
          />
          <Select
            label="Tamaño del modelo"
            data={sizes}
            value={screen.model_size}
            allowDeselect={false}
            disabled={busy}
            onChange={(value) =>
              value && setScreen({ ...screen, model_size: value })
            }
          />
          <Group justify="flex-end">
            <Button onClick={save} loading={busy} disabled={!changed}>
              Guardar
            </Button>
          </Group>
        </Stack>
      )}
    </Paper>
  );
}
