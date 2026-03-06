# Apps Portal

Static frontend for browsing SiMa NEAT app examples.

The portal reads `catalog.json`, which is generated from the per-example `README.md` metadata and section content.

## Features

- searchable app cards
- metadata filters for category, difficulty, status, model, and tags
- detail view per example
- tabbed rendering of README-derived sections

## Prerequisites

Install Node.js and npm if not already available:

```bash
sudo apt update
sudo apt install -y nodejs npm
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
