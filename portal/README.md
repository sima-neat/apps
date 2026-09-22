# Apps Portal

Static frontend for browsing SiMa NEAT app examples.

The portal reads `catalog.json`, which is generated from the per-example `README.md` metadata and section content.

## Features

- searchable app cards
- metadata filters for category, difficulty, status, model, and tags
- detail view per example
- tabbed rendering of README-derived sections

## Prerequisites

Node.js and npm, and Python 3.10 or newer.

Python is required because `npm run dev` and `npm run build` both shell out to
`scripts/generate_catalog.py` to rebuild `catalog.json`. Those scripts use
`X | None` type annotations, which Python 3.9 evaluates at runtime and rejects
with `TypeError: unsupported operand type(s) for |`. macOS ships Python 3.9 as
the system `python3`, so check your version before the first build:

```bash
python3 --version
```

On Debian and Ubuntu:

```bash
sudo apt update
sudo apt install -y nodejs npm python3
```

On macOS, with [Homebrew](https://brew.sh):

```bash
brew install node python@3.11
```

If your default `python3` is older than 3.10, run the npm scripts with a newer
interpreter on `PATH` ahead of it, or regenerate the catalog manually:

```bash
python3.11 ../scripts/generate_catalog.py > public/catalog.json
npx vite
```

## Run

Install dependencies:

```bash
cd <apps-repo-root>/portal
npm install
```

Start the development server:

```bash
npm run dev
```

This automatically regenerates `public/catalog.json` from:

```bash
../scripts/generate_catalog.py
```

## Build

```bash
cd <apps-repo-root>/portal
npm run build
```

The static site output is written to:

```bash
dist/
```
