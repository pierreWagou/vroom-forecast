---
name: docs
description: MkDocs Material documentation site — structure, conventions, and how to add or edit pages
---

## Documentation Stack

- **MkDocs Material** with custom Turo-purple theming
- Config: `mkdocs.yml`
- Content: `docs/` directory
- Served locally on **http://localhost:8100** (via `mprocs` or `docker compose`)

## How Pages Work

Most docs pages are **symlinks** to sub-project READMEs:

```
docs/index.md        -> ../README.md
docs/features.md     -> ../features/README.md
docs/training.md     -> ../training/README.md
docs/promotion.md    -> ../promotion/README.md
docs/serving.md      -> ../serving/README.md
docs/airflow.md      -> ../airflow/README.md
docs/exploration.md  -> ../exploration/README.md
docs/ui.md           -> ../ui/README.md
```

**To update a docs page, edit the source README** (e.g., `training/README.md`),
not the symlink in `docs/`.

The only non-symlinked content pages are:
- `docs/api.md` — hand-written API reference
- `docs/assets/` — logo, favicon, custom CSS

## Adding a New Page

1. Write the markdown file (or create a symlink to an existing README)
2. Add the page to the `nav:` section in `mkdocs.yml`
3. MkDocs hot-reloads automatically when running locally

## Navigation Structure

```yaml
nav:
  - Home: index.md
  - Feature Store: features.md
  - Pipeline:
    - Materialization: airflow.md
    - Training: training.md
    - Promotion: promotion.md
  - Serving:
    - Overview: serving.md
    - API Reference: api.md
  - Frontend: ui.md
  - Exploration: exploration.md
```

## Markdown Features Available

MkDocs Material extensions are configured — use them freely:

- **Admonitions:** `!!! note`, `!!! warning`, `!!! tip`, `!!! info`
- **Collapsible blocks:** `??? note "Title"` (collapsed), `???+ note` (open)
- **Content tabs:** `=== "Tab 1"` / `=== "Tab 2"`
- **Mermaid diagrams:** fenced code blocks with ` ```mermaid `
- **Code highlighting:** fenced blocks with language annotation + copy button
- **Task lists:** `- [x] Done` / `- [ ] Todo`
- **Tables, footnotes, definition lists**

## Writing Conventions

- Keep content in sub-project READMEs when possible (single source of truth)
- Use `docs/api.md` as the model for standalone docs pages
- Prefer Mermaid for diagrams over images
- Use admonitions for callouts (`!!! info`, `!!! warning`)
- Use content tabs for alternative code examples (curl, Python, etc.)
