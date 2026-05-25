const GITHUB_REPO_ROOT = "https://github.com/sima-neat/apps/tree/main";

export async function loadCatalog() {
  const response = await fetch("./catalog.json");
  if (!response.ok) {
    throw new Error(`Failed to load catalog.json (${response.status})`);
  }
  return response.json();
}

export function extractFilterOptions(examples) {
  const difficulties = new Set();
  const languages = new Set();
  const statuses = new Set();
  const models = new Set();
  const tags = new Set();
  const categories = new Set();

  for (const example of examples) {
    if (example.difficulty) {
      difficulties.add(example.difficulty);
    }
    if (example.languages) {
      languages.add(example.languages);
    }
    if (example.status) {
      statuses.add(example.status);
    }
    if (example.model) {
      models.add(example.model);
    }
    if (example.category) {
      categories.add(example.category);
    }
    for (const tag of example.tags || []) {
      tags.add(tag);
    }
  }

  return {
    categories: [...categories].sort(),
    difficulties: [...difficulties].sort(),
    languages: [...languages].sort(),
    statuses: [...statuses].sort(),
    models: [...models].sort(),
    tags: [...tags].sort(),
  };
}

export function matchesFilters(example, filters, query) {
  const haystack = [
    example.name,
    example.summary,
    example.category,
    example.languages,
    example.model,
    example.binary_name,
    ...(example.tags || []),
  ]
    .join(" ")
    .toLowerCase();

  const q = query.trim().toLowerCase();
  if (q && !haystack.includes(q)) {
    return false;
  }
  if (filters.category && example.category !== filters.category) {
    return false;
  }
  if (filters.difficulty && example.difficulty !== filters.difficulty) {
    return false;
  }
  if (filters.languages && example.languages !== filters.languages) {
    return false;
  }
  if (filters.status && example.status !== filters.status) {
    return false;
  }
  if (filters.model && example.model !== filters.model) {
    return false;
  }
  if (filters.tag && !(example.tags || []).includes(filters.tag)) {
    return false;
  }
  return true;
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
