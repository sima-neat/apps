import { marked } from "marked";
import { useDeferredValue, useEffect, useRef, useState } from "react";
import { Link, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { extractFilterOptions, findExample, githubUrlForExample, loadCatalog, matchesFilters } from "./catalog";

marked.setOptions({ breaks: true });

function readInitialTheme() {
  if (typeof window === "undefined") {
    return "light";
  }

  const savedTheme = window.localStorage.getItem("portal-theme");
  if (savedTheme === "light" || savedTheme === "dark") {
    return savedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function App() {
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(readInitialTheme);

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
    window.localStorage.setItem("portal-theme", theme);
  }, [theme]);

  if (error) {
    return (
      <div className="portal-shell">
        <ThemeToggle theme={theme} onToggle={() => setTheme((current) => (current === "light" ? "dark" : "light"))} />
        <div className="state-panel error-panel">{error}</div>
      </div>
    );
  }

  if (!catalog) {
    return (
      <div className="portal-shell">
        <ThemeToggle theme={theme} onToggle={() => setTheme((current) => (current === "light" ? "dark" : "light"))} />
        <div className="state-panel">Loading app catalog...</div>
      </div>
    );
  }

  return (
    <>
      <ThemeToggle theme={theme} onToggle={() => setTheme((current) => (current === "light" ? "dark" : "light"))} />
      <Routes>
        <Route path="/" element={<CatalogPage catalog={catalog} />} />
        <Route path="/app/:appId" element={<DetailPage catalog={catalog} />} />
      </Routes>
    </>
  );
}

function CatalogPage({ catalog }) {
  const [query, setQuery] = useState("");
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  const [filters, setFilters] = useState({
    category: "",
    difficulty: "",
    languages: "",
    status: "",
    model: "",
    tag: "",
  });
  const deferredQuery = useDeferredValue(query);
  const options = extractFilterOptions(catalog.examples);
  const filtered = catalog.examples.filter((example) => matchesFilters(example, filters, deferredQuery));

  return (
    <div className="portal-shell">
      <header className="hero">
        <div className="hero-copy">
          <div className="brand-row">
            <img className="brand-logo" src="./sima-logo.png" alt="SiMa.ai" />
            <p className="eyebrow">SiMa NEAT Apps Portal</p>
          </div>
          <h1>Discover SiMa NEAT reference examples that expedite the path from proof of concept to product.</h1>
          <p className="hero-text">
            Browse source-first applications, search across tags and models, and drill into structured
            example documentation generated from the repo itself.
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
        <div className="hero-search">
          <input
            id="catalog-search"
            className="search-input"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by app name, tag, model, or category"
          />
          <p className="search-meta">{filtered.length} of {catalog.examples.length} examples</p>
        </div>
      </header>

      <div className="portal-layout">
        <main className="card-grid">
          {filtered.map((example) => (
            <ExampleCard key={example.id} example={example} />
          ))}
          {filtered.length === 0 ? (
            <div className="empty-state">
              <h2>No examples matched the current search.</h2>
              <p>Try clearing a filter or using a broader search term.</p>
            </div>
          ) : null}
        </main>

        <aside className="filter-panel">
          <FilterGroup
            label="Category"
            value={filters.category}
            options={options.categories}
            onChange={(value) => setFilters((current) => ({ ...current, category: value }))}
          />
          <FilterGroup
            label="Difficulty"
            value={filters.difficulty}
            options={options.difficulties}
            onChange={(value) => setFilters((current) => ({ ...current, difficulty: value }))}
          />
          <FilterGroup
            label="Languages"
            value={filters.languages}
            options={options.languages}
            onChange={(value) => setFilters((current) => ({ ...current, languages: value }))}
          />
          <FilterGroup
            label="Status"
            value={filters.status}
            options={options.statuses}
            onChange={(value) => setFilters((current) => ({ ...current, status: value }))}
          />
          <FilterGroup
            label="Model"
            value={filters.model}
            options={options.models}
            onChange={(value) => setFilters((current) => ({ ...current, model: value }))}
          />
          <FilterGroup
            label="Tag"
            value={filters.tag}
            options={options.tags}
            onChange={(value) => setFilters((current) => ({ ...current, tag: value }))}
          />
          <button
            className="clear-button"
            type="button"
            onClick={() => setFilters({ category: "", difficulty: "", languages: "", status: "", model: "", tag: "" })}
          >
            Clear Filters
          </button>
        </aside>
      </div>
      {showDownloadModal ? (
        <div className="modal-backdrop" role="presentation" onClick={() => setShowDownloadModal(false)}>
          <div
            className="download-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="download-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="download-modal-header">
              <div>
                <p className="eyebrow">Download</p>
                <h2 id="download-modal-title">Install prebuilt apps from any branch</h2>
              </div>
              <button
                className="modal-close"
                type="button"
                aria-label="Dismiss download instructions"
                onClick={() => setShowDownloadModal(false)}
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
              To download the prebuilt apps package from any branch, run this command from the Modalix DevKit:
            </p>
            <pre className="download-code">
              <code>{`mkdir /media/nvme/neat-apps && cd /media/nvme/neat-apps &&
wget -O /tmp/install.sh https://apps.sima-neat.com/tools/install-neat-apps.sh &&
bash /tmp/install.sh`}</code>
            </pre>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function ExampleCard({ example }) {
  const fallbackImage = `./category-assets/${slugTone(example.category)}.svg`;
  const displayName = formatDisplayLabel(example.name || example.binary_name || example.id);
  const summaryHtml = marked.parseInline(example.summary || "No summary available.");

  return (
    <Link className="app-card" to={`/app/${encodeURIComponent(example.id)}`}>
      <div className="card-image">
        <img
          src={example.image_path ? `./${example.image_path}` : fallbackImage}
          alt={displayName}
        />
      </div>
      <div className="card-body">
        <p className="card-category">{example.category}</p>
        <h2>{displayName}</h2>
        <div className="card-summary" dangerouslySetInnerHTML={{ __html: summaryHtml }} />
      </div>
    </Link>
  );
}

function DetailPage({ catalog }) {
  const { appId = "" } = useParams();
  const navigate = useNavigate();
  const decodedId = decodeURIComponent(appId);
  const example = findExample(catalog, decodedId);
  const githubUrl = githubUrlForExample(example);
  const sections = (example?.sections || []).filter((section) => section.slug !== "metadata");
  const displayName = example?.name || example?.binary_name || decodedId;
  const binaryLabel = example?.binary_name || "";
  const modelLabel = example?.model || "";
  const modelUrl = example?.model_url || "";
  const summaryHtml = marked.parseInline(example?.summary || "No summary available.");
  const pathLabel = decodedId
    .split("/")
    .join(" / ");
  const docPanelRef = useRef(null);
  const [activeSection, setActiveSection] = useState(sections[0]?.slug || "");

  useEffect(() => {
    setActiveSection(sections[0]?.slug || "");
  }, [decodedId, sections]);

  useEffect(() => {
    if (!sections.length) {
      return undefined;
    }

    const root = docPanelRef.current;
    if (!root) {
      return undefined;
    }

    const targets = sections
      .map((section) => root.querySelector(`#${CSS.escape(section.slug)}`))
      .filter(Boolean);

    if (!targets.length) {
      return undefined;
    }

    const updateActiveSection = () => {
      const readingOffset = Math.max(96, Math.round(root.clientHeight * 0.18));
      const readingLine = root.scrollTop + readingOffset;
      let nextActive = targets[0].id;
      let bestDistance = Number.POSITIVE_INFINITY;

      for (const target of targets) {
        const targetTop =
          root.scrollTop +
          target.getBoundingClientRect().top -
          root.getBoundingClientRect().top;
        const distance = Math.abs(targetTop - readingLine);
        if (distance < bestDistance) {
          bestDistance = distance;
          nextActive = target.id;
        }
      }

      setActiveSection((current) => (current === nextActive ? current : nextActive));
    };

    updateActiveSection();
    root.addEventListener("scroll", updateActiveSection, { passive: true });
    return () => root.removeEventListener("scroll", updateActiveSection);
  }, [sections]);

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
    <div className="detail-shell">
      <div className="detail-topbar">
        <button
          className="back-link icon-link"
          type="button"
          onClick={() => navigate("/")}
          aria-label="Back to catalog"
          title="Back to catalog"
        >
          <span aria-hidden="true">⌂</span>
        </button>
        <div className="detail-path">{pathLabel}</div>
      </div>

      <section className="detail-hero">
        <div className="detail-hero-copy">
          <p className="eyebrow">{example.category}</p>
          <h1>{displayName}</h1>
          <p className="hero-text" dangerouslySetInnerHTML={{ __html: summaryHtml }} />
          <div className="detail-meta">
            <Chip label={example.difficulty || "Unspecified"} tone="difficulty" />
            <Chip label={example.status || "experimental"} tone="status" />
            {example.languages ? <Chip label={example.languages} tone="languages" /> : null}
            {example.model ? <ModelChip label={modelLabel} url={modelUrl} /> : null}
            <Chip label={binaryLabel} tone="binary" />
            {githubUrl ? (
              <a
                className="chip chip-github"
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
          </div>
        </div>
      </section>

      <div className="detail-layout">
        <article ref={docPanelRef} className="doc-panel">
          {sections.length > 0 ? (
            <div className="doc-content">
              {sections.map((section) => (
                <section key={section.slug} id={section.slug} className="doc-section">
                  <header className="section-header">
                    <p className="section-kicker">Section</p>
                    <h2>{section.title}</h2>
                  </header>
                  <div
                    className="markdown-body"
                    dangerouslySetInnerHTML={{ __html: marked.parse(section.markdown || "") }}
                  />
                </section>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <h2>No sections available.</h2>
            </div>
          )}
        </article>

        <nav className="toc-panel" aria-label="Table of contents">
          <p className="toc-title">On this page</p>
          {sections.map((section) => (
            <button
              key={section.slug}
              className={`toc-button ${activeSection === section.slug ? "active" : ""}`}
              type="button"
              onClick={() => {
                const root = docPanelRef.current;
                const target = root?.querySelector(`#${CSS.escape(section.slug)}`);
                if (root && target) {
                  const topPadding = 32;
                  const targetTop =
                    root.scrollTop +
                    target.getBoundingClientRect().top -
                    root.getBoundingClientRect().top;
                  root.scrollTo({ top: Math.max(targetTop - topPadding, 0), behavior: "smooth" });
                }
                setActiveSection(section.slug);
              }}
            >
              {section.title}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
}

function FilterGroup({ label, value, options, onChange }) {
  return (
    <label className="filter-group">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">All</option>
        {options.map((option) => (
          <option key={option} value={option}>{formatDisplayLabel(option)}</option>
        ))}
      </select>
    </label>
  );
}

function ThemeToggle({ theme, onToggle }) {
  return (
    <button
      className="theme-toggle"
      type="button"
      onClick={onToggle}
      aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
      title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
    >
      <span className={`theme-toggle-track ${theme === "dark" ? "dark" : ""}`}>
        <span className="theme-toggle-label">{theme === "light" ? "Light" : "Dark"}</span>
        <span className="theme-toggle-thumb" />
      </span>
    </button>
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
    ["optiview", "OptiView"],
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
