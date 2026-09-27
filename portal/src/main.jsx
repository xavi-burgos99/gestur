import React, { useCallback, useEffect, useState } from "react";
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
  PasswordInput,
  Alert,
  Loader,
  ActionIcon,
  Tooltip,
  Box,
} from "@mantine/core";
import {
  IconCube,
  IconAdjustmentsHorizontal,
  IconSettings,
  IconLogout,
  IconArrowRight,
} from "@tabler/icons-react";
import { DeviceOperationModal, InitialSetup } from "./DeviceManagement.jsx";
import {
  deviceJobActive,
  forgetDeviceOperation,
  rememberDeviceOperation,
  restoreDeviceOperation,
} from "./device-operation.mjs";
import "@mantine/core/styles.css";
import "./styles.css";
import { api } from "./api.mjs";
import Models from "./ModelsPage.jsx";
import Parameters from "./ParametersPage.jsx";
import Settings from "./SettingsPage.jsx";

function App() {
  const [auth, setAuth] = useState(null);
  const [setup, setSetup] = useState(null);
  const [entryError, setEntryError] = useState("");
  const [operation, setOperation] = useState(restoreDeviceOperation);
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [config, setConfig] = useState(null);
  const [defaults, setDefaults] = useState(null);
  const [tab, setTab] = useState("models");
  const [notice, setNotice] = useState(null);
  const notify = (message, error = false) => setNotice({ message, error });
  const beginOperation = useCallback((value) => {
    setOperation(rememberDeviceOperation(value));
  }, []);
  const loadEntry = useCallback(async () => {
    setEntryError("");
    try {
      const initial = await api("setup", { silentAuth: true });
      setSetup(initial);
      if (deviceJobActive(initial.job)) {
        beginOperation({ ...initial, kind: initial.job.kind });
      }
      if (initial.required) {
        setAuth(false);
        setConfig(null);
      } else {
        const session = await api("session");
        setAuth(session.authenticated);
      }
    } catch (e) {
      setEntryError(e.message);
    }
  }, [beginOperation]);
  useEffect(() => {
    loadEntry();
    const expire = () => {
      setAuth(false);
      setConfig(null);
      api("setup", { silentAuth: true })
        .then(setSetup)
        .catch(() => {});
    };
    window.addEventListener("gestur-expired", expire);
    return () => window.removeEventListener("gestur-expired", expire);
  }, [loadEntry]);
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
  if (operation)
    return (
      <div className="login-page">
        <Paper p={40} withBorder radius="lg" className="login-card">
          <Title order={1}>Gestur</Title>
        </Paper>
        <DeviceOperationModal
          operation={operation}
          api={api}
          onClose={() => {
            forgetDeviceOperation();
            setOperation(null);
            loadEntry();
          }}
        />
      </div>
    );
  if (entryError)
    return (
      <div className="login-page">
        <Paper p={40} withBorder radius="lg" className="login-card">
          <Title order={1} mb="xl">
            Gestur
          </Title>
          <Alert color="red" title="No se puede acceder al dispositivo">
            {entryError}
          </Alert>
          <Button mt="lg" onClick={loadEntry}>
            Volver a comprobar
          </Button>
        </Paper>
      </div>
    );
  if (!setup || auth === null)
    return (
      <div className="centered">
        <Loader />
      </div>
    );
  if (setup.required)
    return (
      <InitialSetup setup={setup} api={api} onOperation={beginOperation} />
    );
  if (!auth)
    return (
      <div className="login-page">
        <Paper p={40} withBorder radius="lg" className="login-card">
          <Title order={1} mb="xl">
            Gestur
          </Title>
          <form onSubmit={login}>
            <Stack>
              <PasswordInput
                label="Contraseña"
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
                Acceder
              </Button>
            </Stack>
          </form>
        </Paper>
      </div>
    );
  return (
    <AppShell header={{ height: 84 }} padding={0}>
      <AppShell.Header>
        <Container size="xl" h="100%">
          <Group justify="space-between" h="100%">
            <Text className="wordmark">Gestur</Text>
            <Group>
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
                <Tabs.Panel value="parameters" keepMounted>
                  <Parameters
                    config={config}
                    setConfig={setConfig}
                    defaults={defaults}
                    notify={notify}
                  />
                </Tabs.Panel>
                <Tabs.Panel value="settings">
                  <Settings notify={notify} onOperation={beginOperation} />
                </Tabs.Panel>
              </>
            )}
          </Tabs>
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
