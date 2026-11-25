// node --test portal/src/ (sima-neat/apps#776)
import assert from "node:assert/strict";
import test from "node:test";
import { readCookieTheme, readInitialTheme, readStoredTheme, themeFromShellEvent, urlTheme } from "./theme.js";

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

test("the shared header's event: persist false only when it says so", () => {
  assert.deepEqual(themeFromShellEvent({ detail: { theme: "dark", persist: false } }), { theme: "dark", persist: false });
  assert.deepEqual(themeFromShellEvent({ detail: { theme: "light", persist: true } }), { theme: "light", persist: true });
  assert.deepEqual(themeFromShellEvent({ detail: { theme: "dark" } }), { theme: "dark", persist: true }, "an older header doesn't say");
  assert.equal(themeFromShellEvent({ detail: { theme: "blue" } }), null);
  assert.equal(themeFromShellEvent({}), null);
});
