import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Anchor,
  Button,
  Divider,
  Group,
  Loader,
  Modal,
  Paper,
  PasswordInput,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
  Title,
} from "@mantine/core";
import {
  IconArrowRight,
  IconCheck,
  IconDeviceDesktop,
  IconRefresh,
  IconTrash,
} from "@tabler/icons-react";

import {
  deviceJobActive,
  hostnameError,
  localUrl,
  validSetupPassword,
  matchesDeviceJob,
  forgetDeviceOperation,
  rememberDeviceOperation,
} from "./device-operation.mjs";

export function DeviceOperationModal({ operation, api, onClose }) {
  const [current, setCurrent] = useState(operation);
  const [unreachable, setUnreachable] = useState(false);
  const [timedOut, setTimedOut] = useState(false);
  const [checking, setChecking] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const expectedId = useRef(operation.job?.id);
  const failed = current.job?.state === "failed";
  const completed = current.job?.state === "completed";
  const applied = current.job?.state === "rebooting";
  const target = localUrl(current.hostname);
  useEffect(() => {
    let stopped = false;
    let timer;
    let controller;
    const started = Date.now();
    setTimedOut(false);
    async function poll() {
      if (stopped) return;
      setChecking(true);
      controller = new AbortController();
      const deadline = setTimeout(() => controller.abort(), 10000);
      try {
        let data;
        try {
          data = await api("device/job", {
            silentAuth: true,
            signal: controller.signal,
          });
        } catch (error) {
          if (![401, 403].includes(error.status)) throw error;
          data = await api("setup", {
            silentAuth: true,
            signal: controller.signal,
          });
        }
        if (stopped) return;
        const job = data.job || (data.state ? data : null);
        const matching = matchesDeviceJob(job, operation, expectedId.current);
        if (matching) {
          expectedId.current ||= job.id;
          const next = rememberDeviceOperation({
            ...current,
            ...data,
            job,
            kind: operation.kind,
            startedAt: operation.startedAt,
          });
          setCurrent(next);
          setUnreachable(false);
          if (["failed", "completed"].includes(job.state)) return;
        } else {
          // A stale job or a successful HTTP request is not proof of this change.
          setUnreachable(true);
        }
      } catch {
        if (!stopped) setUnreachable(true);
      } finally {
        clearTimeout(deadline);
        if (!stopped) setChecking(false);
      }
      if (!stopped) {
        if (Date.now() - started >= 90000) setTimedOut(true);
        else
          timer = setTimeout(poll, Date.now() - started < 15000 ? 3000 : 5000);
      }
    }
    timer = setTimeout(poll, 1500);
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller?.abort();
    };
    // A poll owns its recovery snapshot; an explicit retry starts a new poll.
  }, [api, attempt, operation.kind, operation.startedAt]);

  const title = failed
    ? "No se pudo completar el cambio"
    : completed
      ? "Configuración aplicada"
      : timedOut
        ? "Comprueba la conexión"
        : unreachable
          ? "Esperando al dispositivo"
          : applied
            ? "Reiniciando el dispositivo"
            : "Aplicando cambios";
  return (
    <Modal
      opened
      onClose={failed ? onClose : () => {}}
      withCloseButton={failed}
      closeOnClickOutside={false}
      closeOnEscape={failed}
      centered
      title={title}
      size="md"
    >
      <Stack gap="lg">
        <Group align="flex-start" wrap="nowrap" aria-live="polite">
          {!failed && !completed && !timedOut && <Loader size="sm" mt={3} />}
          {completed && (
            <IconCheck size={20} color="var(--mantine-color-teal-8)" />
          )}
          <Text size="sm">
            {failed
              ? current.job.error ||
                current.job.message ||
                "El dispositivo no pudo aplicar los cambios."
              : completed
                ? "El dispositivo ha confirmado los cambios. Ya puedes volver a acceder al portal."
                : timedOut
                  ? "No se ha podido confirmar el reinicio. Vuelve a conectarte a la red y abre el portal."
                  : unreachable
                    ? "La conexión puede interrumpirse durante el reinicio. Esto no confirma que el cambio haya terminado."
                    : applied
                      ? "Los cambios se han aplicado. Espera a que el dispositivo reinicie para volver a acceder."
                      : "Guardando la configuración. El dispositivo se reiniciará al terminar."}
          </Text>
        </Group>
        {!failed && (
          <>
            <Divider />
            <Stack gap="xs">
              <Text size="sm" fw={600}>
                Volver a conectar
              </Text>
              <Text size="sm">
                {current.ssid ? (
                  <>Conéctate a la red «{current.ssid}».</>
                ) : (
                  "Vuelve a conectarte a la red del dispositivo."
                )}{" "}
                {operation.kind === "setup"
                  ? "Usa la contraseña que acabas de crear."
                  : operation.kind === "reset"
                    ? "La red quedará abierta, sin contraseña."
                    : "La contraseña Wi-Fi se mantiene."}
              </Text>
              {target && (
                <Anchor
                  href={target}
                  onClick={() => forgetDeviceOperation()}
                  className="device-address"
                >
                  {target}
                </Anchor>
              )}
              <Text size="sm" c="dimmed">
                Si el nombre no responde, accede mediante la IP del dispositivo.
              </Text>
              {current.portal_url && (
                <Anchor
                  href={current.portal_url}
                  onClick={() => forgetDeviceOperation()}
                  className="device-address"
                >
                  {current.portal_url}
                </Anchor>
              )}
              {operation.kind === "reset" && (
                <Text size="sm">
                  Al entrar aparecerá la configuración inicial.
                </Text>
              )}
            </Stack>
          </>
        )}
        <Group justify="flex-end">
          {failed ? (
            <Button variant="default" onClick={onClose}>
              Volver
            </Button>
          ) : (
            <Button
              variant="default"
              leftSection={<IconRefresh size={16} />}
              loading={checking}
              onClick={() => setAttempt((value) => value + 1)}
            >
              Comprobar estado
            </Button>
          )}
        </Group>
      </Stack>
    </Modal>
  );
}

async function submitChange(api, endpoint, method, body, details, onOperation) {
  const startedAt = Date.now();
  try {
    const data = await api(endpoint, { method, body });
    onOperation({ ...details, ...data, startedAt });
  } catch (error) {
    if (error.status) throw error;
    // A dropped response may already have applied the mutation. Check its job;
    // never submit the destructive operation a second time automatically.
    onOperation({
      ...details,
      startedAt,
      job: { kind: details.kind, state: "unknown" },
    });
  }
}

export function InitialSetup({ setup, api, onOperation }) {
  const [step, setStep] = useState(0);
  const [hostname, setHostname] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(event) {
    event.preventDefault();
    const invalidName = hostnameError(hostname, true);
    if (invalidName) {
      setError(invalidName);
      return;
    }
    if (step === 0) {
      setError("");
      setStep(1);
      return;
    }
    if (!validSetupPassword(password)) {
      setError(
        "La contraseña debe tener de 8 a 63 caracteres, sin tildes, ñ ni emojis.",
      );
      return;
    }
    if (password !== confirmation) {
      setError("Las contraseñas no coinciden.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await submitChange(
        api,
        "setup",
        "POST",
        { hostname, password, confirmation },
        {
          kind: "setup",
          hostname: hostname || setup.hostname,
          ssid: setup.ssid,
          portal_url: setup.portal_url,
        },
        onOperation,
      );
      setPassword("");
      setConfirmation("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="login-page">
      <Paper
        p={{ base: "xl", sm: 40 }}
        withBorder
        radius="lg"
        className="login-card setup-card"
      >
        <Title order={1} mb="xl">
          Gestur
        </Title>
        <form onSubmit={submit}>
          <Stack gap="lg">
            <div>
              <Text size="xs" c="dimmed" fw={600} mb={6}>
                PASO {step + 1} DE 2
              </Text>
              <Title order={2}>
                {step === 0 ? "Nombre del dispositivo" : "Contraseña de acceso"}
              </Title>
            </div>
            {step === 0 ? (
              <>
                <Text size="sm" c="dimmed">
                  Puedes conservar el nombre actual o elegir otro para acceder
                  por la red.
                </Text>
                <TextInput
                  label="Nombre"
                  placeholder={setup.hostname}
                  value={hostname}
                  onChange={(event) => setHostname(event.currentTarget.value)}
                  maxLength={63}
                  rightSection={
                    <Text size="sm" c="dimmed">
                      .local
                    </Text>
                  }
                  rightSectionWidth={65}
                  description={`Opcional. Si lo dejas vacío se conserva ${setup.hostname}.local.`}
                  autoComplete="off"
                  autoCapitalize="none"
                  spellCheck={false}
                />
              </>
            ) : (
              <>
                <Text size="sm" c="dimmed">
                  Se usará para entrar al portal y para conectarte al punto de
                  acceso Wi-Fi.
                </Text>
                <PasswordInput
                  label="Contraseña"
                  description="De 8 a 63 caracteres, sin tildes ni ñ."
                  value={password}
                  onChange={(event) => setPassword(event.currentTarget.value)}
                  minLength={8}
                  maxLength={63}
                  autoComplete="new-password"
                  required
                  disabled={busy}
                />
                <PasswordInput
                  label="Repite la contraseña"
                  value={confirmation}
                  onChange={(event) =>
                    setConfirmation(event.currentTarget.value)
                  }
                  minLength={8}
                  maxLength={63}
                  autoComplete="new-password"
                  required
                  disabled={busy}
                />
                <Text size="sm" c="dimmed">
                  Al finalizar, el dispositivo se reiniciará.
                </Text>
              </>
            )}
            {error && (
              <Alert color="red" role="alert">
                {error}
              </Alert>
            )}
            <Group justify="space-between">
              {step === 1 ? (
                <Button
                  variant="subtle"
                  disabled={busy}
                  onClick={() => {
                    setStep(0);
                    setError("");
                  }}
                >
                  Atrás
                </Button>
              ) : (
                <span />
              )}
              <Button
                type="submit"
                loading={busy}
                rightSection={<IconArrowRight size={17} />}
              >
                {step === 0 ? "Continuar" : "Guardar y reiniciar"}
              </Button>
            </Group>
          </Stack>
        </form>
      </Paper>
    </div>
  );
}

export function DeviceSettings({
  api,
  onOperation,
  disabled = false,
  children,
}) {
  const [device, setDevice] = useState(null);
  const [hostname, setHostname] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const [resetError, setResetError] = useState("");
  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const value = await api("device");
      setDevice(value);
      setHostname(value.hostname);
      if (deviceJobActive(value.job))
        onOperation({ ...value, kind: value.job.kind });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [api, onOperation]);
  useEffect(() => {
    load();
  }, [load]);
  async function rename(event) {
    event.preventDefault();
    const invalid = hostnameError(hostname);
    if (invalid) {
      setError(invalid);
      return;
    }
    setBusy(true);
    setError("");
    try {
      await submitChange(
        api,
        "device/hostname",
        "PUT",
        { hostname },
        {
          kind: "hostname",
          hostname,
          ssid: device.ssid,
          portal_url: device.portal_url,
        },
        onOperation,
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function reset(event) {
    event.preventDefault();
    if (confirmation !== "BORRAR") return;
    setBusy(true);
    setResetError("");
    try {
      await submitChange(
        api,
        "device/reset",
        "POST",
        { confirmation },
        {
          kind: "reset",
          hostname: device.default_hostname || device.hostname,
          portal_url: device.portal_url,
        },
        onOperation,
      );
      setResetOpen(false);
      setConfirmation("");
    } catch (e) {
      setResetError(e.message);
    } finally {
      setBusy(false);
    }
  }
  const locked = busy || disabled || deviceJobActive(device?.job);
  return (
    <Stack gap="xl" maw={820}>
      <Paper withBorder p={{ base: "lg", sm: 32 }}>
        <Group mb="xl" wrap="nowrap">
          <ThemeIcon size={48} radius="md" variant="light">
            <IconDeviceDesktop size={25} />
          </ThemeIcon>
          <div>
            <Title order={2}>Dispositivo</Title>
            <Text size="sm" c="dimmed">
              Nombre para acceder desde la red local.
            </Text>
          </div>
        </Group>
        {loading ? (
          <Loader size="sm" />
        ) : !device ? (
          <Alert color="red" title="Dispositivo no disponible">
            {error}
            <Button variant="light" mt="md" onClick={load}>
              Volver a comprobar
            </Button>
          </Alert>
        ) : (
          <form onSubmit={rename}>
            <Stack gap="md">
              <TextInput
                label="Nombre del dispositivo"
                value={hostname}
                maxLength={63}
                onChange={(event) => setHostname(event.currentTarget.value)}
                disabled={locked}
                rightSection={
                  <Text size="sm" c="dimmed">
                    .local
                  </Text>
                }
                rightSectionWidth={65}
                autoComplete="off"
                autoCapitalize="none"
                spellCheck={false}
                required
              />
              <Text size="sm" c="dimmed">
                Cambiar el nombre reinicia el dispositivo. La dirección IP puede
                seguir usándose.
              </Text>
              {error && (
                <Alert color="red" role="alert">
                  {error}
                </Alert>
              )}
              <Group justify="flex-end">
                <Button
                  type="submit"
                  loading={busy}
                  disabled={locked || hostname === device.hostname || !hostname}
                >
                  Guardar y reiniciar
                </Button>
              </Group>
            </Stack>
          </form>
        )}
      </Paper>
      {children}
      <Paper withBorder p={{ base: "lg", sm: 32 }}>
        <Stack gap="md">
          <Title order={2}>Restablecer dispositivo</Title>
          <Text size="sm" c="dimmed">
            Elimina los modelos y los ajustes y vuelve a la configuración
            inicial.
          </Text>
          <Group>
            <Button
              color="red"
              variant="light"
              leftSection={<IconTrash size={17} />}
              disabled={loading || !device || locked}
              onClick={() => {
                setConfirmation("");
                setResetError("");
                setResetOpen(true);
              }}
            >
              Borrar contenido y ajustes
            </Button>
          </Group>
        </Stack>
      </Paper>
      <Modal
        opened={resetOpen}
        onClose={() => {
          if (!busy) setResetOpen(false);
        }}
        title="Borrar contenido y ajustes"
        centered
        closeOnClickOutside={!busy}
        closeOnEscape={!busy}
        withCloseButton={!busy}
      >
        <form onSubmit={reset}>
          <Stack gap="lg">
            <Text>
              Se eliminarán todos los modelos, los ajustes y las contraseñas de
              acceso. El punto de acceso Wi-Fi quedará abierto y el portal
              volverá a la configuración inicial.
            </Text>
            <Text size="sm" fw={600}>
              Esta acción no se puede deshacer. El dispositivo se reiniciará.
            </Text>
            <TextInput
              label="Escribe BORRAR para confirmar"
              value={confirmation}
              onChange={(event) => setConfirmation(event.currentTarget.value)}
              autoComplete="off"
              spellCheck={false}
              disabled={busy}
              required
            />
            {resetError && (
              <Alert color="red" role="alert">
                {resetError}
              </Alert>
            )}
            <Group justify="flex-end">
              <Button
                variant="default"
                disabled={busy}
                onClick={() => setResetOpen(false)}
              >
                Cancelar
              </Button>
              <Button
                type="submit"
                color="red"
                loading={busy}
                disabled={confirmation !== "BORRAR"}
              >
                Borrar y reiniciar
              </Button>
            </Group>
          </Stack>
        </form>
      </Modal>
    </Stack>
  );
}
