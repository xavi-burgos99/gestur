import { readFile } from "node:fs/promises";
import { createHash, scrypt, timingSafeEqual } from "node:crypto";
import { promisify } from "node:util";

const derive = promisify(scrypt);
const validLegacy = (token) => typeof token === "string" && token.length >= 24;

// Only the root-owned device helper writes this file. Never return its contents
// to a client; revision binds existing sessions to the current credentials.
export function createCredentials({ token, deviceStatePath }) {
  async function read() {
    let state;
    try {
      const raw = await readFile(deviceStatePath, "utf8");
      if (raw.length > 16384) throw new Error("Invalid device state");
      state = JSON.parse(raw);
    } catch (error) {
      if (error.code !== "ENOENT" || !validLegacy(token)) throw error;
      state = { version: 1, setup_complete: true, legacy_auth: true };
    }
    if (state.version !== 1 || typeof state.setup_complete !== "boolean")
      throw new Error("Invalid device state");
    if (state.setup_complete) {
      const hash = state.password_hash;
      if (state.legacy_auth === true && !hash) {
        if (!validLegacy(token)) throw new Error("Missing legacy credential");
      } else if (
        !hash ||
        hash.algorithm !== "scrypt" ||
        hash.n !== 16384 ||
        hash.r !== 8 ||
        hash.p !== 1 ||
        !/^[a-f0-9]{32}$/.test(hash.salt) ||
        !/^[a-f0-9]{64}$/.test(hash.key)
      ) {
        throw new Error("Invalid device credential");
      }
    }
    return {
      required: !state.setup_complete,
      revision: createHash("sha256")
        .update(JSON.stringify(state))
        .digest("hex"),
      async verify(password) {
        if (
          !state.setup_complete ||
          typeof password !== "string" ||
          password.length > 256
        )
          return false;
        const hash = state.password_hash;
        if (!hash) {
          const actual = Buffer.from(password),
            expected = Buffer.from(token);
          return (
            actual.length === expected.length &&
            timingSafeEqual(actual, expected)
          );
        }
        const actual = await derive(
          password,
          Buffer.from(hash.salt, "hex"),
          32,
          { N: hash.n, r: hash.r, p: hash.p, maxmem: 64 * 1024 * 1024 },
        );
        return timingSafeEqual(actual, Buffer.from(hash.key, "hex"));
      },
    };
  }
  return { read };
}
