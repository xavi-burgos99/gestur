import test from "node:test";
import assert from "node:assert/strict";
import {
  hostnameError,
  validSetupPassword,
  localUrl,
  ipUrl,
  operationData,
  matchesDeviceJob,
  rememberDeviceOperation,
  restoreDeviceOperation,
} from "../src/device-operation.mjs";

test("onboarding allows keeping the hostname and validates the full DNS label", () => {
  assert.equal(hostnameError("", true), "");
  for (const name of ["gestur", "gestur-1234", "a", "1", "a".repeat(63)]) {
    assert.equal(hostnameError(name), "");
    assert.equal(localUrl(name), `http://${name}.local/`);
  }
  for (const name of [
    "",
    "GESTUR",
    "gestur.local",
    "-gestur",
    "gestur-",
    "a b",
    "a".repeat(64),
  ]) {
    assert.ok(hostnameError(name));
    assert.equal(localUrl(name), null);
  }
  assert.equal(validSetupPassword("Ascii123!"), true);
  assert.equal(validSetupPassword("gestur ñ"), false);
  assert.equal(validSetupPassword("short"), false);
  assert.equal(validSetupPassword("a".repeat(64)), false);
});

test("reconnection links accept only literal IP origins as a fallback", () => {
  assert.equal(ipUrl("http://192.168.4.1/models"), "http://192.168.4.1/");
  assert.equal(ipUrl("http://[fe80::1]:80/"), "http://[fe80::1]/");
  for (const address of [
    "http://gestur.local/",
    "javascript:alert(1)",
    "file:///etc/passwd",
    "http://999.999.1.1/",
  ]) {
    assert.equal(ipUrl(address), null);
  }
});

test("persisted recovery data excludes credentials and uses the job target", () => {
  let stored;
  const storage = {
    setItem: (_, value) => {
      stored = value;
    },
    getItem: () => stored,
    removeItem: () => {
      stored = null;
    },
  };
  const operation = rememberDeviceOperation(
    {
      kind: "setup",
      hostname: "old",
      password: "PRIVATE_PASSWORD",
      confirmation: "PRIVATE_PASSWORD",
      token: "PRIVATE_TOKEN",
      startedAt: 100,
      job: {
        id: "job-1",
        kind: "setup",
        state: "pending",
        hostname: "new-name",
        password: "PRIVATE_PASSWORD",
        private: { key: "PRIVATE_TOKEN" },
      },
    },
    storage,
  );
  assert.equal(operation.hostname, "new-name");
  assert.ok(!stored.includes("PRIVATE"));
  assert.equal(restoreDeviceOperation(storage, 101).job.id, "job-1");
  assert.equal(restoreDeviceOperation(storage, 700000), null);
  assert.equal(stored, null);
  assert.equal(
    operationData({ kind: "reset", job: {} }, "http://10.0.0.7/").portal_url,
    "http://10.0.0.7/",
  );
});

test("a stale job cannot confirm a request whose response was lost", () => {
  const operation = { kind: "hostname", hostname: "new-name" };
  const job = {
    id: "j1",
    kind: "hostname",
    hostname: "new-name",
    state: "pending",
  };
  assert.equal(matchesDeviceJob(job, operation, null), true);
  assert.equal(
    matchesDeviceJob({ ...job, hostname: "another" }, operation, null),
    false,
  );
  assert.equal(
    matchesDeviceJob({ ...job, state: "completed" }, operation, null),
    false,
  );
  assert.equal(
    matchesDeviceJob({ ...job, state: "failed" }, operation, null),
    false,
  );
  assert.equal(
    matchesDeviceJob({ ...job, state: "completed" }, operation, "j1"),
    true,
  );
  assert.equal(
    matchesDeviceJob({ ...job, id: "previous" }, operation, "j1"),
    false,
  );
  assert.equal(
    matchesDeviceJob({ ...job, kind: "reset" }, operation, "j1"),
    false,
  );
});

test("a recent failed job can be recovered after a dropped mutation response", () => {
  const operation = {
    kind: "setup",
    hostname: "gestur-1234",
    startedAt: 100000,
  };
  const job = {
    id: "new",
    kind: "setup",
    hostname: "gestur-1234",
    state: "failed",
    created_at: 100020,
  };
  assert.equal(matchesDeviceJob(job, operation, null), true);
  assert.equal(
    matchesDeviceJob({ ...job, state: "completed" }, operation, null),
    true,
  );
  assert.equal(
    matchesDeviceJob({ ...job, created_at: 1 }, operation, null),
    false,
  );
  assert.equal(
    matchesDeviceJob({ ...job, hostname: "different" }, operation, null),
    false,
  );
});
