// node --test portal/src/ (sima-neat/apps#776)
import assert from "node:assert/strict";
import test from "node:test";
import { readCookieTheme, readInitialTheme, readStoredTheme, storeTheme, themeFromShellEvent, urlTheme } from "./theme.js";

const storage = (values = {}) => ({ getItem: (key) => (key in values ? values[key] : null) });
const STORED_LIGHT = { cookie: "sima-neat-theme=light", storage: storage({ "portal-theme": "light", theme: "light" }) };

test("?theme wins over the cookie, localStorage and the system, and is not to be stored", () => {
  assert.deepEqual(readInitialTheme({ ...STORED_LIGHT, search: "?theme=dark", prefersDark: false }), { theme: "dark", persist: false });
  assert.deepEqual(readInitialTheme({ ...STORED_LIGHT, cookie: "sima-neat-theme=dark", search: "?theme=light", prefersDark: true }), { theme: "light", persist: false });
});

test("?theme is found among other query parameters", () => {
  assert.equal(urlTheme("?q=yolo&theme=dark&category=detection"), "dark");
});

test("invalid ?theme values are ignored", () => {
  for (const search of ["?theme=blue", "?theme=Dark", "?theme=", "?themes=dark", ""]) {
    assert.deepEqual(readInitialTheme({ ...STORED_LIGHT, search, prefersDark: true }), { theme: "light", persist: true }, search);
  }
});

test("without ?theme the order is unchanged: cookie, then localStorage, then the system", () => {
  assert.deepEqual(readInitialTheme({ search: "", cookie: "a=1; sima-neat-theme=dark", storage: storage({ theme: "light" }), prefersDark: false }), { theme: "dark", persist: true });
  assert.deepEqual(readInitialTheme({ search: "", cookie: "", storage: storage({ theme: "dark" }), prefersDark: false }), { theme: "dark", persist: true });
  assert.deepEqual(readInitialTheme({ search: "", cookie: "", storage: storage(), prefersDark: true }), { theme: "dark", persist: true });
  assert.deepEqual(readInitialTheme({ search: "", cookie: "", storage: storage(), prefersDark: false }), { theme: "light", persist: true });
});

test("restricted storage and a bad cookie fall through to the next source", () => {
  const throwing = { getItem() { throw new Error("denied"); } };
  assert.equal(readStoredTheme(throwing), null);
  assert.equal(readCookieTheme("sima-neat-theme=%E0%A4%A"), null);
});

test("the shared header's event: persist from the header, or kept when an older header does not say", () => {
  assert.deepEqual(themeFromShellEvent({ detail: { theme: "dark", persist: false } }), { theme: "dark", persist: false });
  assert.deepEqual(themeFromShellEvent({ detail: { theme: "light", persist: true } }), { theme: "light", persist: true });
  assert.deepEqual(themeFromShellEvent({ detail: { theme: "dark" } }), { theme: "dark", persist: true }, "an older header doesn't say");
  assert.equal(
    themeFromShellEvent({ detail: { theme: "light" } }, { theme: "dark", persist: false }),
    null,
    "an older header's event doesn't replace a theme from ?theme (Codex on #833)",
  );
  assert.deepEqual(themeFromShellEvent({ detail: { theme: "light" } }, { theme: "dark", persist: true }), { theme: "light", persist: true });
  assert.equal(themeFromShellEvent({ detail: { theme: "blue" } }), null);
  assert.equal(themeFromShellEvent({}), null);
});

test("a browser that refuses localStorage and the cookie still gets ?theme (Codex on #833)", (t) => {
  const refuse = () => {
    throw new DOMException("The operation is insecure.", "SecurityError");
  };
  const fakeWindow = { location: { search: "?theme=dark" }, matchMedia: () => ({ matches: false }) };
  Object.defineProperty(fakeWindow, "localStorage", { get: refuse });
  const fakeDocument = {};
  Object.defineProperty(fakeDocument, "cookie", { get: refuse });
  globalThis.window = fakeWindow;
  globalThis.document = fakeDocument;
  t.after(() => {
    delete globalThis.window;
    delete globalThis.document;
  });
  assert.deepEqual(readInitialTheme(), { theme: "dark", persist: false });
  fakeWindow.location.search = "";
  assert.deepEqual(readInitialTheme(), { theme: "light", persist: true }, "falls back to the system preference");
});

test("storing the theme survives a refused cookie and still writes localStorage (Codex on #833)", (t) => {
  const stored = new Map();
  const fakeDocument = {};
  Object.defineProperty(fakeDocument, "cookie", {
    set() {
      throw new DOMException("The operation is insecure.", "SecurityError");
    },
  });
  globalThis.window = {
    location: { hostname: "localhost", protocol: "https:" },
    localStorage: { setItem: (key, value) => stored.set(key, value) },
  };
  globalThis.document = fakeDocument;
  t.after(() => {
    delete globalThis.window;
    delete globalThis.document;
  });
  assert.doesNotThrow(() => storeTheme("dark"));
  assert.equal(stored.get("portal-theme"), "dark");
});
