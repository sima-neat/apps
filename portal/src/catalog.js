const GITHUB_REPO_ROOT = "https://github.com/sima-neat/apps/tree/main";
const PORTAL_BASE_PATH = import.meta.env.BASE_URL;

export function portalAssetUrl(path) {
  return `${PORTAL_BASE_PATH}${path.replace(/^\/+/, "")}`;
}

export async function loadCatalog() {
  const response = await fetch(portalAssetUrl("catalog.json"));
  if (!response.ok) {
    throw new Error(`Failed to load catalog.json (${response.status})`);
  }
  return response.json();
}

export function extractFilterOptions(examples) {
  const categories = new Set();
  const models = new Set();

  for (const example of examples) {
    if (example.category) {
      categories.add(example.category);
    }
    if (example.model) {
      models.add(example.model);
    }
  }

  return {
    categories: [...categories].sort(),
    models: [...models].sort(),
  };
}

export function matchesFilters(example, filters) {
  return (!filters.category || example.category === filters.category)
    && (!filters.model || example.model === filters.model);
}

export function findExample(catalog, id) {
  return catalog.examples.find((example) => example.id === id) || null;
}

export function githubUrlForExample(example) {
  if (!example?.source_path) {
    return "";
  }
  return `${GITHUB_REPO_ROOT}/${example.source_path}`;
}
