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
  const [filters, setFilters] = useState({
    category: "",
    difficulty: "",
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
          <p className="eyebrow">SiMa NEAT Apps Portal</p>
          <h1>Discover SiMa NEAT reference examples that expedite the path from proof of concept to product.</h1>
          <p className="hero-text">
            Browse source-first applications, search across tags and models, and drill into structured
            example documentation generated from the repo itself.
          </p>
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
            onClick={() => setFilters({ category: "", difficulty: "", status: "", model: "", tag: "" })}
          >
            Clear Filters
          </button>
        </aside>
      </div>
    </div>
  );
}

function ExampleCard({ example }) {
  const fallbackImage = `./category-assets/${slugTone(example.category)}.svg`;

  return (
    <Link className="app-card" to={`/app/${encodeURIComponent(example.id)}`}>
      <div className="card-image">
        <img
          src={example.image_path ? `./${example.image_path}` : fallbackImage}
          alt={example.name}
        />
      </div>
      <div className="card-body">
        <p className="card-category">{example.category}</p>
        <h2>{example.name}</h2>
        <p className="card-summary">{example.summary || "No summary available."}</p>
        <div className="card-meta">
          <Chip label={example.difficulty || "Unspecified"} tone="difficulty" />
          <Chip label={example.status || "experimental"} tone="status" />
          {example.model ? <Chip label={example.model} tone="model" /> : null}
        </div>
        <div className="tag-row">
          {(example.tags || []).slice(0, 4).map((tag) => (
            <span key={tag} className="tag-pill">{tag}</span>
          ))}
        </div>
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
      const scrollAnchor = root.scrollTop + 48;
      let nextActive = targets[0].id;

      for (const target of targets) {
        if (target.offsetTop <= scrollAnchor) {
          nextActive = target.id;
        } else {
          break;
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
        <button className="back-link" type="button" onClick={() => navigate("/")}>Back to catalog</button>
        <div className="detail-path">{example.id}</div>
      </div>

      <section className="detail-hero">
        <div className="detail-hero-copy">
          <p className="eyebrow">{example.category}</p>
          <h1>{example.name}</h1>
          <p className="hero-text">{example.summary || "No summary available."}</p>
          <div className="detail-meta">
            <Chip label={example.difficulty || "Unspecified"} tone="difficulty" />
            <Chip label={example.status || "experimental"} tone="status" />
            {example.model ? <Chip label={example.model} tone="model" /> : null}
            <Chip label={example.binary_name} tone="binary" />
          </div>
        </div>
        <div className="detail-hero-card">
          <h2>Metadata</h2>
          <dl>
            <MetaItem label="Category" value={example.category} />
            <MetaItem label="Difficulty" value={example.difficulty || "Unspecified"} />
            <MetaItem label="Status" value={example.status || "experimental"} />
            <MetaItem label="Model" value={example.model || "Not specified"} />
            <MetaItem label="Binary" value={example.binary_name} />
          </dl>
          {githubUrl ? (
            <a className="source-link" href={githubUrl} target="_blank" rel="noreferrer">
              View Source on GitHub
            </a>
          ) : null}
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
                  root.scrollTo({ top: target.offsetTop - 24, behavior: "smooth" });
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
          <option key={option} value={option}>{option}</option>
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

function MetaItem({ label, value }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}

function slugTone(category) {
  return category.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

export default App;
