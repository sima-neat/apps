// The portal's light/dark theme, shared with the Developer Center: the
// sima-neat-theme cookie and the portal-theme / theme localStorage keys.
//
// A page that embeds the portal, such as the Palette Code VS Code extension,
// asks for a theme with ?theme=dark|light (sima-neat/apps#776). It wins over
// the stored theme and is never stored ({persist: false}), so the theme the
// user picked on the site stays as it was. The theme toggle stores the user's
// choice as usual.
//
// No React here, so `node --test portal/src/` can check it.

export const THEME_COOKIE = "sima-neat-theme";
export const THEME_KEYS = ["portal-theme", "theme"];
const VALID_THEMES = new Set(["light", "dark"]);

export function normalizeTheme(value) {
  return VALID_THEMES.has(value) ? value : null;
}

/** The theme a ?theme=dark|light query asks for, or null. */
export function urlTheme(search) {
  try {
    return normalizeTheme(new URLSearchParams(search).get("theme"));
  } catch {
    return null;
  }
}

export function readCookieTheme(cookie = document.cookie) {
  const entry = cookie
    .split("; ")
    .find((item) => item.startsWith(`${THEME_COOKIE}=`));
  if (!entry) {
    return null;
  }
  try {
    return normalizeTheme(decodeURIComponent(entry.split("=").slice(1).join("=")));
  } catch {
    return null;
  }
}

export function readStoredTheme(storage = window.localStorage) {
  for (const key of THEME_KEYS) {
    try {
      const value = normalizeTheme(storage.getItem(key));
      if (value) {
        return value;
      }
    } catch {
      // Restricted storage: use the next source.
    }
  }
  return null;
}

/** read(), or fallback when the browser refuses it (sandboxed or privacy-restricted embedding). */
function safely(read, fallback) {
  try {
    return read();
  } catch {
    return fallback;
  }
}

function browserSources() {
  if (typeof window === "undefined") {
    return { search: "", cookie: "", storage: null, prefersDark: false };
  }
  // Each source on its own: a refused cookie or localStorage must not stop ?theme.
  return {
    search: safely(() => window.location.search, ""),
    cookie: safely(() => document.cookie, ""),
    storage: safely(() => window.localStorage, null),
    prefersDark: safely(() => window.matchMedia("(prefers-color-scheme: dark)").matches, false),
  };
}

/**
 * The first theme: ?theme, then the cookie, then localStorage, then the system
 * preference. persist is false for a theme from the URL.
 */
export function readInitialTheme({ search, cookie, storage, prefersDark } = browserSources()) {
  const fromUrl = urlTheme(search);
  if (fromUrl) {
    return { theme: fromUrl, persist: false };
  }
  const saved = readCookieTheme(cookie) || (storage ? readStoredTheme(storage) : null);
  return { theme: saved || (prefersDark ? "dark" : "light"), persist: true };
}

/**
 * The theme state after a developer-center-theme-change event from the shared
 * header, or null to keep current. The header says persist: false for a theme
 * from the URL (sima-neat/docs#199). An older header doesn't know ?theme: it
 * announces the stored theme without persist, so while the theme comes from
 * ?theme its events are ignored (Codex on #833). Otherwise its theme applies and
 * the portal stores it as before.
 */
export function themeFromShellEvent(event, current = { persist: true }) {
  const theme = normalizeTheme(event?.detail?.theme);
  if (!theme) {
    return null;
  }
  const said = event.detail.persist;
  if (typeof said === "boolean") {
    return { theme, persist: said };
  }
  return current.persist ? { theme, persist: true } : null;
}

function cookieDomain() {
  const { hostname } = window.location;
  if (hostname === "localhost" || /^[\d.]+$/.test(hostname)) {
    return "";
  }

  if (hostname.endsWith(".neat.sima.ai") || hostname.endsWith(".neat.paconsultings.com")) {
    const parts = hostname.split(".");
    return parts.length > 3 ? parts.slice(1).join(".") : hostname;
  }

  return "";
}

/** Stores the user's theme in the cookie and localStorage. */
export function storeTheme(theme) {
  const domain = cookieDomain();
  const domainPart = domain ? `; Domain=.${domain}` : "";
  const securePart = window.location.protocol === "https:" ? "; Secure" : "";
  try {
    document.cookie = `${THEME_COOKIE}=${encodeURIComponent(theme)}; Path=/; Max-Age=31536000; SameSite=Lax${securePart}${domainPart}`;
  } catch {
    // Restricted embedding: localStorage may still hold it.
  }
  THEME_KEYS.forEach((key) => {
    try {
      window.localStorage.setItem(key, theme);
    } catch {
      // Restricted storage: the cookie still holds it.
    }
  });
}
