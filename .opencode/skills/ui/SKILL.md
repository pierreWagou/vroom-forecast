---
name: ui
description: Next.js + shadcn/ui frontend — Turo-inspired design, API consumption, SSE streams, TypeScript conventions
---

## Role

You are a frontend engineer building a demo UI for an ML prediction service.

## Overview

Next.js frontend (`ui/`) that consumes the Ray Serve prediction API. Built with
shadcn/ui components and Turo-inspired design language.

## Rules

- All components from shadcn/ui; no per-component color overrides — use globals.css
- Dark mode: next-themes with system preference default, class-based toggling
- API client in `lib/api.ts` — keep types in sync with `serving/schemas.py`
- No Python in this directory; this is a pure npm project

## Tabs

| Tab | Description |
|---|---|
| **Simulation** | Configure vehicle attributes via sliders, predict reservation count |
| **Catalog** | Fleet vehicles with history + new arrivals pending prediction |
| **Benchmark** | Latency comparison: raw features vs online store path |
| **Feature Store** | Operational view of offline/online stores |

## Design Language

- Turo purple primary (`#593CFB`), teal accent (`#00B8A9`)
- Dark mode via `next-themes`
- Pill-shaped buttons, card-heavy layout
- shadcn/ui base-nova style with neutral base color

## Configuration

| Env var | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Ray Serve API base URL |

## SSE Streams

- **`GET /events`** — model promotion events. Triggers health refresh on new champion.
- **`GET /vehicles/events`** — vehicle materialization events. The FeatureMaterializer
  publishes to `vroom-forecast:vehicle-materialized` on Redis pub/sub when a new
  arrival's features are written to the online store. The SSE endpoint forwards
  these to the UI.

## File Layout

```
ui/
  src/
    app/
      layout.tsx          # Root layout (theme provider, fonts)
      page.tsx            # Main page (all tabs)
    lib/
      api.ts              # API client (fetch wrappers)
      utils.ts            # cn() helper
    components/
      theme-provider.tsx  # next-themes wrapper
      error-boundary.tsx  # React error boundary
      ui/                 # shadcn/ui primitives (badge, button, card, etc.)
  package.json
  tsconfig.json
  next.config.ts
  postcss.config.mjs
  eslint.config.mjs
  components.json         # shadcn config
```

## Dependencies

next, react, @base-ui/react, shadcn, class-variance-authority, clsx,
lucide-react, next-themes, tailwind-merge. Dev: tailwindcss v4, typescript,
eslint-config-next.

## Run Locally

```bash
cd ui && npm install && npm run dev
```

## Standards

- Lint: `npm run lint` (eslint-config-next with core-web-vitals + TypeScript)
- Type check: `npm run typecheck` (tsc --noEmit)
- Path alias: `@/*` maps to `./src/*`
