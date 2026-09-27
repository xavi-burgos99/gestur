import React, { useEffect, useState } from "react";
import {
  Group,
  Stack,
  Title,
  Text,
  Button,
  Paper,
  Badge,
  TextInput,
  PasswordInput,
  Modal,
  Alert,
  Loader,
  Divider,
  ThemeIcon,
} from "@mantine/core";
import {
  IconWifi,
  IconLock,
  IconLockOpen,
  IconPlus,
  IconAlertCircle,
  IconRefresh,
} from "@tabler/icons-react";
import { DeviceSettings } from "./DeviceManagement.jsx";
import { api } from "./api.mjs";
import SectionTitle from "./SectionTitle.jsx";

export default function Settings({ notify, onOperation }) {
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
      <SectionTitle title="Configuración" />
      <DeviceSettings
        api={api}
        onOperation={onOperation}
        disabled={busy || pending}
      >
        <Paper withBorder p={{ base: "lg", sm: 32 }} maw={820}>
          <Group justify="space-between" mb="xl">
            <Group>
              <ThemeIcon size={48} radius="md" variant="light">
                <IconWifi size={25} />
              </ThemeIcon>
              <div>
                <Title order={2}>Punto de acceso Wi-Fi</Title>
                <Text c="dimmed" size="sm">
                  Red del dispositivo para acceder al portal.
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
                desconectarte. Estos cambios no modifican la contraseña del
                portal.
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
      </DeviceSettings>
      <Text size="xs" c="dimmed" mt="xl">
        Desarrollado por Xavier Burgos
      </Text>
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
              description="De 8 a 63 caracteres, sin tildes ni ñ."
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
