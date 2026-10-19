import { marked } from "marked";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, Route, Routes, useNavigate, useParams } from "react-router-dom";
import {
  extractFilterOptions,
  findExample,
  githubUrlForExample,
  loadCatalog,
  matchesFilters,
  portalAssetUrl,
} from "./catalog";

marked.setOptions({ breaks: false });

const THEME_COOKIE = "sima-neat-theme";
const THEME_KEYS = ["portal-theme", "theme"];
const VALID_THEMES = new Set(["light", "dark"]);
const SHOW_DEVELOPER_CENTER_NAV = import.meta.env.BASE_URL.replace(/\/+$/, "") === "/examples";

function normalizeTheme(value) {
  return VALID_THEMES.has(value) ? value : null;
}

function readCookieTheme() {
  const entry = document.cookie
    .split("; ")
    .find((cookie) => cookie.startsWith(`${THEME_COOKIE}=`));
  return normalizeTheme(entry ? decodeURIComponent(entry.split("=").slice(1).join("=")) : null);
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

function writeCookieTheme(theme) {
  const domain = cookieDomain();
  const domainPart = domain ? `; Domain=.${domain}` : "";
  const securePart = window.location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `${THEME_COOKIE}=${encodeURIComponent(theme)}; Path=/; Max-Age=31536000; SameSite=Lax${securePart}${domainPart}`;
}

function readStoredTheme() {
  for (const key of THEME_KEYS) {
    const value = normalizeTheme(window.localStorage.getItem(key));
    if (value) {
      return value;
    }
  }
  return null;
}

function writeStoredTheme(theme) {
  THEME_KEYS.forEach((key) => window.localStorage.setItem(key, theme));
}

function normalizeMarkdownAssetUrls(markdown = "") {
  const assetPrefixes = ["example-assets", "category-assets"];
  return assetPrefixes.reduce((content, prefix) => {
    const assetUrl = portalAssetUrl(`${prefix}/`);
    return content
      .replace(new RegExp(`(\\]\\()(?:\\./)?${prefix}/`, "g"), `$1${assetUrl}`)
      .replace(new RegExp(`(\\bsrc=["'])(?:\\./)?${prefix}/`, "g"), `$1${assetUrl}`);
  }, markdown);
}

function renderMarkdown(markdown = "") {
  return marked.parse(normalizeMarkdownAssetUrls(markdown));
}

function readInitialTheme() {
  if (typeof window === "undefined") {
    return "light";
  }

  const savedTheme = readCookieTheme() || readStoredTheme();
  if (savedTheme) {
    return savedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function ensureDeveloperCenterShellStylesheet() {
  const href = "/developer-center-shell.css";
  if (document.querySelector(`link[href="${href}"]`)) {
    return;
  }

  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.appendChild(link);
}

function ensureDeveloperCenterShellScript(onLoad, onError) {
  const src = "/developer-center-shell.js";
  const existing = document.querySelector(`script[src="${src}"]`);
  if (window.DeveloperCenterShell) {
    onLoad();
    return undefined;
  }

  if (existing) {
    existing.addEventListener("load", onLoad, { once: true });
    return () => existing.removeEventListener("load", onLoad);
  }

  const script = document.createElement("script");
  script.src = src;
  script.async = true;
  script.addEventListener("load", onLoad, { once: true });
  script.addEventListener("error", () => {
    console.warn("Developer Center shell runtime is unavailable.");
    onError?.();
  }, { once: true });
  document.head.appendChild(script);
  return () => script.removeEventListener("load", onLoad);
}

function useDeveloperCenterShell(setTheme, setUseShellFallback) {
  useEffect(() => {
    if (!SHOW_DEVELOPER_CENTER_NAV) {
      return undefined;
    }

    let cancelled = false;
    ensureDeveloperCenterShellStylesheet();

    const mountShell = () => {
      if (!cancelled) {
        if (window.DeveloperCenterShell) {
          setUseShellFallback(false);
          window.DeveloperCenterShell.mount("#developer-center-shell", { active: "examples" });
        } else {
          setUseShellFallback(true);
        }
      }
    };
    const cleanupScriptListener = ensureDeveloperCenterShellScript(mountShell, () => {
      if (!cancelled) {
        setUseShellFallback(true);
      }
    });
    const fallbackTimer = window.setTimeout(() => {
      if (!cancelled && !window.DeveloperCenterShell) {
        setUseShellFallback(true);
      }
    }, 1200);
    const syncTheme = (event) => {
      const theme = normalizeTheme(event.detail?.theme);
      if (theme) {
        setTheme(theme);
      }
    };

    window.addEventListener("developer-center-theme-change", syncTheme);
    return () => {
      cancelled = true;
      window.clearTimeout(fallbackTimer);
      cleanupScriptListener?.();
      window.removeEventListener("developer-center-theme-change", syncTheme);
    };
  }, [setTheme, setUseShellFallback]);
}

function App() {
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(readInitialTheme);
  const [useShellFallback, setUseShellFallback] = useState(false);
  const toggleTheme = () => setTheme((current) => (current === "light" ? "dark" : "light"));
  useDeveloperCenterShell(setTheme, setUseShellFallback);

  useEffect(() => {
    let cancelled = false;
    loadCatalog()
      .then((data) => {
        if (!cancelled) {
          setCatalog(data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    writeStoredTheme(theme);
    writeCookieTheme(theme);
  }, [theme]);

  useEffect(() => {
    const syncTheme = () => {
      const cookieTheme = readCookieTheme();
      if (cookieTheme) {
        setTheme(cookieTheme);
      }
    };

    window.addEventListener("focus", syncTheme);
    document.addEventListener("visibilitychange", syncTheme);
    return () => {
      window.removeEventListener("focus", syncTheme);
      document.removeEventListener("visibilitychange", syncTheme);
    };
  }, []);

  if (error) {
    return (
      <AppFrame theme={theme} onToggleTheme={toggleTheme} useShellFallback={useShellFallback}>
        <div className="portal-shell">
          <div className="state-panel error-panel">{error}</div>
        </div>
      </AppFrame>
    );
  }

  if (!catalog) {
    return (
      <AppFrame theme={theme} onToggleTheme={toggleTheme} useShellFallback={useShellFallback}>
        <div className="portal-shell">
          <div className="state-panel">Loading app catalog...</div>
        </div>
      </AppFrame>
    );
  }

  return (
    <AppFrame theme={theme} onToggleTheme={toggleTheme} useShellFallback={useShellFallback}>
      <Routes>
        <Route path="/" element={<CatalogPage catalog={catalog} />} />
        <Route path="/app/*" element={<DetailPage catalog={catalog} />} />
      </Routes>
    </AppFrame>
  );
}

function AppFrame({ children, theme, onToggleTheme, useShellFallback }) {
  return (
    <>
      {SHOW_DEVELOPER_CENTER_NAV && !useShellFallback ? <div id="developer-center-shell" /> : null}
      {SHOW_DEVELOPER_CENTER_NAV && useShellFallback ? (
        <DeveloperCenterShellFallback theme={theme} onToggleTheme={onToggleTheme} />
      ) : null}
      {children}
    </>
  );
}

function DeveloperCenterShellFallback({ theme, onToggleTheme }) {
  return (
    <div className="developer-center-fallback-shell">
      <header className="navbar dev-center-navbar">
        <div className="navbar__inner">
          <div className="navbar__items">
            <a className="navbar__brand" href="/" aria-label="Developer Center home">
              <span className="navbar__logo">
                <img src="/img/sima-logo.png" alt="" />
              </span>
              <span className="navbar__title">Developer Center</span>
            </a>
            <div className="navbar__items navbar__items--desktop">
              <div className="navbar__item">
                <a className="navbar__link" href="/hardware">Hardware</a>
              </div>
              <div className="navbar__item">
                <a className="navbar__link" href="/software">Software</a>
              </div>
              <div className="navbar__item">
                <a className="navbar__link navbar__link--active" href="/examples">Examples</a>
              </div>
              <a className="navbar__item navbar__link" href="https://huggingface.co/simaai" target="_blank" rel="noreferrer">
                Models
                <svg width="13.5" height="13.5" viewBox="0 0 24 24" aria-label="(opens in new tab)" className="iconExternalLink">
                  <path fill="currentColor" d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3ZM5 5h6v2H7v10h10v-4h2v6H5V5Z" />
                </svg>
              </a>
              <a className="navbar__item navbar__link" href="https://developer.sima.ai" target="_blank" rel="noreferrer">
                Community
                <svg width="13.5" height="13.5" viewBox="0 0 24 24" aria-label="(opens in new tab)" className="iconExternalLink">
                  <path fill="currentColor" d="M14 3h7v7h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H14V3ZM5 5h6v2H7v10h10v-4h2v6H5V5Z" />
                </svg>
              </a>
            </div>
          </div>
          <div className="navbar__items navbar__items--right">
            <div className="color-mode-toggle">
              <button
                className="color-mode-toggle-button"
                type="button"
                aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
                title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
                onClick={onToggleTheme}
              >
                {theme === "light" ? (
                  <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" className="color-mode-toggle-icon">
                    <path fill="currentColor" d="M12 9c1.65 0 3 1.35 3 3s-1.35 3-3 3-3-1.35-3-3 1.35-3 3-3m0-2c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5ZM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1Zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1ZM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1Zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1ZM5.99 4.58a1 1 0 0 0-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41L5.99 4.58Zm12.37 12.37a1 1 0 0 0-1.41 1.41l1.06 1.06a1 1 0 0 0 1.41-1.41l-1.06-1.06ZM19.42 5.99a1 1 0 0 0-1.41-1.41l-1.06 1.06a1 1 0 0 0 1.41 1.41l1.06-1.06ZM7.05 18.36a1 1 0 0 0-1.41-1.41l-1.06 1.06a1 1 0 0 0 1.41 1.41l1.06-1.06Z" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" className="color-mode-toggle-icon">
                    <path fill="currentColor" d="M9.37 5.51A7.4 7.4 0 0 0 16.5 14.9c.68 0 1.35-.09 1.99-.27A7.03 7.03 0 0 1 12 19c-3.86 0-7-3.14-7-7 0-2.93 1.81-5.45 4.37-6.49ZM12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.4 5.4 0 0 1-4.4 2.26 5.4 5.4 0 0 1-3.14-9.8A9.2 9.2 0 0 0 12 3Z" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>
    </div>
  );
}

function CatalogPage({ catalog }) {
  const [query, setQuery] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [filters, setFilters] = useState({
    category: "",
    model: "",
  });
  const options = extractFilterOptions(catalog.examples);
  const search = query.trim().toLowerCase();
  const filtered = catalog.examples.filter((example) => (
    matchesFilters(example, filters)
    && [example.name, example.summary, example.category, formatTaskLabel(example.category), example.model,
      example.difficulty, example.languages, example.status, ...(example.tags || [])]
      .join(" ").toLowerCase().includes(search)
  ));

  return (
    <div className="portal-shell">
      <header className="hero">
        <div className="hero-copy">
          <h1>Neat Examples</h1>
          <p className="hero-text">
            Learn how to build production-grade applications by exploring these examples.
          </p>
          <div className="hero-actions">
            <a className="hero-action" href="https://github.com/sima-neat/apps" target="_blank" rel="noreferrer">
              <span className="hero-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" focusable="false">
                  <path
                    d="M12 2C6.48 2 2 6.58 2 12.23c0 4.52 2.87 8.36 6.84 9.71.5.1.68-.22.68-.49 0-.24-.01-1.04-.01-1.88-2.78.62-3.37-1.21-3.37-1.21-.45-1.18-1.11-1.49-1.11-1.49-.91-.64.07-.63.07-.63 1 .08 1.53 1.06 1.53 1.06.9 1.57 2.36 1.12 2.93.86.09-.67.35-1.12.63-1.38-2.22-.26-4.56-1.14-4.56-5.09 0-1.13.39-2.05 1.03-2.77-.1-.26-.45-1.31.1-2.73 0 0 .84-.28 2.75 1.06A9.32 9.32 0 0 1 12 6.84c.85 0 1.71.12 2.51.35 1.91-1.34 2.75-1.06 2.75-1.06.55 1.42.2 2.47.1 2.73.64.72 1.03 1.64 1.03 2.77 0 3.96-2.34 4.83-4.57 5.08.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.59.69.49A10.25 10.25 0 0 0 22 12.23C22 6.58 17.52 2 12 2Z"
                    fill="currentColor"
                  />
                </svg>
              </span>
              <span>GitHub</span>
            </a>
            <button className="hero-action" type="button" onClick={() => setShowDownloadModal(true)}>
              <span className="hero-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" focusable="false">
                  <path
                    d="M12 3a1 1 0 0 1 1 1v8.59l2.3-2.29a1 1 0 1 1 1.4 1.41l-4 4a1 1 0 0 1-1.4 0l-4-4a1 1 0 1 1 1.4-1.41L11 12.59V4a1 1 0 0 1 1-1Zm-7 14a1 1 0 0 1 1 1v1h12v-1a1 1 0 1 1 2 0v2a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-2a1 1 0 0 1 1-1Z"
                    fill="currentColor"
                  />
                </svg>
              </span>
              <span>Download</span>
            </button>
          </div>
        </div>
        <div className="hero-stat">
          <span>{filtered.length}</span>
          <p>{filtered.length === catalog.examples.length ? "examples" : `of ${catalog.examples.length} examples`}</p>
        </div>
      </header>

      <section className="catalog-toolbar" aria-label="Search and filter examples">
        <label className="catalog-search">
          <span className="sr-only">Search examples</span>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="m21 21-4.35-4.35m2.35-5.15A7.5 7.5 0 1 1 4 11.5a7.5 7.5 0 0 1 15 0Z" />
          </svg>
          <input
            type="search"
            placeholder="Search examples, tasks, or models…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <button
          className="filter-toggle"
          type="button"
          aria-expanded={filtersOpen}
          aria-controls="example-filters"
          onClick={() => setFiltersOpen((open) => !open)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M7 12h10m-7 5h4" /></svg>
          Filters
        </button>
      </section>

      <div className="portal-layout">
        <aside id="example-filters" className="filter-panel" data-open={filtersOpen}>
          <FilterGroup
            label="Task"
            value={filters.category}
            options={options.categories}
            formatLabel={formatTaskLabel}
            onChange={(value) => setFilters((current) => ({ ...current, category: value }))}
          />
          <FilterGroup
            label="Model"
            value={filters.model}
            options={options.models}
            onChange={(value) => setFilters((current) => ({ ...current, model: value }))}
          />
          <button
            className="clear-button"
            type="button"
            onClick={() => {
              setFilters({ category: "", model: "" });
              setQuery("");
            }}
          >
            Clear Filters
          </button>
        </aside>
        <main className="card-grid">
          {filtered.map((example) => (
            <ExampleCard key={example.id} example={example} />
          ))}
          {filtered.length === 0 ? (
            <div className="empty-state">
              <h2>No examples matched the current filters.</h2>
              <p>Try clearing a filter.</p>
            </div>
          ) : null}
        </main>
      </div>
      {showDownloadModal ? <AppsDownloadDialog onClose={() => setShowDownloadModal(false)} /> : null}
    </div>
  );
}

function AppsDownloadDialog({ onClose }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    dialog.showModal();
    return () => dialog.close();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className="download-modal"
      aria-labelledby="download-modal-title"
      onClose={onClose}
      onClick={(event) => {
        if (event.target !== event.currentTarget) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
            event.clientY < bounds.top || event.clientY > bounds.bottom) {
          event.currentTarget.close();
        }
      }}
    >
      <div className="download-modal-header">
        <div>
          <p className="eyebrow">Download</p>
          <h2 id="download-modal-title">Download Neat Apps</h2>
        </div>
        <button
          className="modal-close"
          type="button"
          autoFocus
          aria-label="Dismiss download instructions"
          onClick={() => dialogRef.current.close()}
        >
          <svg viewBox="0 0 24 24" focusable="false">
            <path
              d="M6.7 5.3a1 1 0 0 0-1.4 1.4L10.6 12l-5.3 5.3a1 1 0 1 0 1.4 1.4l5.3-5.3 5.3 5.3a1 1 0 0 0 1.4-1.4L13.4 12l5.3-5.3a1 1 0 1 0-1.4-1.4L12 10.6 6.7 5.3Z"
              fill="currentColor"
            />
          </svg>
        </button>
      </div>
      <p className="download-modal-copy">
        To install the complete Neat Apps bundle, run:
      </p>
      <pre className="download-code">
        <code>{`sima-cli neat install apps`}</code>
      </pre>
    </dialog>
  );
}

function ExampleCard({ example }) {
  const fallbackImage = portalAssetUrl(`category-assets/${slugTone(example.category)}.svg`);
  const cardImage = portalAssetUrl(example.image_path || `category-assets/${slugTone(example.category)}.svg`);
  const displayName = formatDisplayLabel(example.name || example.binary_name || example.id);
  const summaryHtml = marked.parseInline(example.summary || "No summary available.");

  return (
    <Link className="app-card" to={`/app/${encodeURIComponent(example.id)}`}>
      <div className="card-image">
        <img
          src={cardImage}
          alt={displayName}
          onError={(event) => {
            if (event.currentTarget.dataset.fallbackApplied !== "true") {
              event.currentTarget.dataset.fallbackApplied = "true";
              event.currentTarget.src = fallbackImage;
            }
          }}
        />
      </div>
      <div className="card-body">
        <p className="card-category">{formatTaskLabel(example.category)}</p>
        <h2>{displayName}</h2>
        <div className="card-summary" dangerouslySetInnerHTML={{ __html: summaryHtml }} />
      </div>
    </Link>
  );
}

function DetailPage({ catalog }) {
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const { "*": appPath = "" } = useParams();
  const navigate = useNavigate();
  const decodedId = decodeURIComponent(appPath);
  const example = findExample(catalog, decodedId);
  const githubUrl = githubUrlForExample(example);
  const displayName = example?.name || example?.binary_name || decodedId;
  const previewImage = example?.image_path ? portalAssetUrl(example.image_path) : "";
  const summaryHtml = marked.parseInline(example?.summary || "No summary available.");
  const docPanelRef = useRef(null);
  const [activeSection, setActiveSection] = useState("");
  const groups = useMemo(() => {
    const content = example?.sections || [];
    const setup = new Set([
      "prerequisites", "install-apps", "prepare-the-model", "prepare-insight",
      "prepare-an-rtsp-source", "configure", "build-the-host-application",
    ]);
    const run = new Set(["run", "expected-result", "benchmark-results", "performance"]);
    return [
      {id: "setup", title: "Setup", sections: content.filter(({slug}) => setup.has(slug))},
      {id: "execution", title: "Run", sections: content.filter(({slug}) => run.has(slug))},
      {id: "use-your-own-model", title: "Use Your Own Model", sections: content.filter(({slug}) => slug === "use-your-own-model")},
      {id: "reference", title: "Reference", sections: content.filter(({slug}) => (
        !["metadata", "concept", "preview", "use-your-own-model"].includes(slug) && !setup.has(slug) && !run.has(slug)
      ))},
    ].filter((group) => group.sections.length);
  }, [example]);

  useEffect(() => {
    setActiveSection(groups[0]?.id || "");
    const target = document.getElementById(window.location.hash.slice(1));
    if (target) target.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [decodedId, groups]);

  useEffect(() => {
    const targets = [...(docPanelRef.current?.querySelectorAll("[data-doc-group]") || [])];
    if (!targets.length) return undefined;
    const updateActiveSection = () => {
      let current = targets[0].id;
      for (const target of targets) {
        if (target.getBoundingClientRect().top <= 140) current = target.id;
      }
      setActiveSection(current);
    };
    updateActiveSection();
    window.addEventListener("scroll", updateActiveSection, {passive: true});
    return () => window.removeEventListener("scroll", updateActiveSection);
  }, [groups]);

  if (!example) {
    return (
      <div className="portal-shell">
        <div className="state-panel error-panel">
          Example not found.
          <button className="clear-button" type="button" onClick={() => navigate("/")}>Back to catalog</button>
        </div>
      </div>
    );
  }

  return (
    <main className="detail-shell">
      <nav className="detail-breadcrumbs" aria-label="Breadcrumb">
        <Link to="/">Examples</Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">{displayName}</span>
      </nav>

      <header className={`detail-hero${previewImage ? "" : " detail-hero-no-image"}`}>
        <div className="detail-hero-copy">
          <h1>{displayName}</h1>
          <div className="detail-meta">
            <Chip label={formatTaskLabel(example.category)} tone="status" />
            <Chip label={example.difficulty || "Unspecified"} tone="difficulty" />
            <Chip label={example.status || "experimental"} tone="status" />
            {example.languages ? <Chip label={example.languages} tone="languages" /> : null}
            {example.model ? <ModelChip label={example.model} url={example.model_url} /> : null}
            {example.binary_name ? <Chip label={example.binary_name} tone="binary" /> : null}
          </div>
          <p className="hero-text" dangerouslySetInnerHTML={{ __html: summaryHtml }} />
          <div className="detail-actions">
            {githubUrl ? (
              <a
                className="example-source-link"
                href={githubUrl}
                target="_blank"
                rel="noreferrer"
                aria-label="View source on GitHub"
                title="View source on GitHub"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path
                    d="M12 2C6.48 2 2 6.59 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.03-.01-1.87-2.78.62-3.37-1.21-3.37-1.21-.45-1.2-1.11-1.52-1.11-1.52-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.89 1.58 2.34 1.12 2.91.86.09-.67.35-1.12.63-1.38-2.22-.26-4.56-1.15-4.56-5.14 0-1.14.4-2.07 1.05-2.8-.11-.27-.46-1.33.1-2.77 0 0 .86-.28 2.82 1.07A9.54 9.54 0 0 1 12 6.92c.85 0 1.71.12 2.51.36 1.96-1.35 2.82-1.07 2.82-1.07.56 1.44.21 2.5.1 2.77.65.73 1.05 1.66 1.05 2.8 0 4-2.34 4.87-4.57 5.13.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.8 0 .27.18.59.69.49A10.27 10.27 0 0 0 22 12.25C22 6.59 17.52 2 12 2Z"
                    fill="currentColor"
                  />
                </svg>
              </a>
            ) : null}
            <button className="hero-action download-apps-action" type="button" onClick={() => setShowDownloadModal(true)}>
              <span className="hero-action-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 3v12m-5-5 5 5 5-5M5 17v4h14v-4" />
                </svg>
              </span>
              Download
            </button>
          </div>
        </div>
        {previewImage ? (
          <figure id="preview" className="detail-preview">
            <img src={previewImage} alt={`${displayName} preview`} />
          </figure>
        ) : null}
      </header>

      <nav className="detail-section-nav" aria-label="Example sections">
        {groups.map((group) => (
          <a key={group.id} href={`#${group.id}`} aria-current={activeSection === group.id ? "location" : undefined}>
            {group.title}
          </a>
        ))}
      </nav>

      <article ref={docPanelRef} className="example-documentation">
        {groups.map((group) => (
          <section key={group.id} id={group.id} data-doc-group="" className="example-doc-group">
            <h2>{group.title}</h2>
            {group.sections.map((section) => {
              const html = renderMarkdown(section.markdown);
              if (!html.trim()) return null;
              return (
                <section key={section.slug} id={section.slug === group.id ? undefined : section.slug} className="doc-section">
                  {section.title !== group.title ? <h3>{section.title}</h3> : null}
                  <div className="markdown-body" dangerouslySetInnerHTML={{ __html: html }} />
                </section>
              );
            })}
          </section>
        ))}
      </article>
      {showDownloadModal ? <AppsDownloadDialog onClose={() => setShowDownloadModal(false)} /> : null}
    </main>
  );
}

function FilterGroup({ label, value, options, onChange, formatLabel = formatDisplayLabel }) {
  return (
    <label className="filter-group">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">All</option>
        {options.map((option) => (
          <option key={option} value={option}>{formatLabel(option)}</option>
        ))}
      </select>
    </label>
  );
}

function Chip({ label, tone }) {
  return <span className={`chip chip-${tone}`}>{label}</span>;
}

function ModelChip({ label, url }) {
  if (!url) {
    return <Chip label={label} tone="model" />;
  }

  return (
    <a
      className="chip chip-model chip-link"
      href={url}
      target="_blank"
      rel="noreferrer"
      title={`Download ${label}`}
    >
      {label}
    </a>
  );
}

function formatTaskLabel(value) {
  if (value === "classification") return "Image Classification";
  if (value === "genai") return "GenAI";
  return formatDisplayLabel(value);
}

function slugTone(category) {
  return category.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function formatDisplayLabel(value) {
  if (!value) {
    return "";
  }

  const needsFormatting = value.includes("_") || /^[a-z0-9-]+$/i.test(value);
  if (!needsFormatting) {
    return value;
  }

  const acronyms = new Map([
    ["rtsp", "RTSP"],
    ["insight", "Insight"],
    ["yolo", "YOLO"],
    ["mpk", "MPK"],
    ["api", "API"],
    ["sdk", "SDK"],
    ["fps", "FPS"],
    ["rgb", "RGB"],
    ["nv12", "NV12"],
  ]);

  return value
    .replace(/[-/]+/g, " ")
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => {
      const lower = part.toLowerCase();
      if (acronyms.has(lower)) {
        return acronyms.get(lower);
      }
      if (/^yolov\d+[a-z0-9-]*$/i.test(part)) {
        return part.replace(/^yolo/i, "YOLO");
      }
      if (/^midas_v?\d+/i.test(part)) {
        return part.replace(/^midas/i, "MiDaS");
      }
      if (/^v\d+(\.\d+)?$/i.test(part)) {
        return part.toUpperCase();
      }
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

export default App;
