# UI

Next.js + shadcn/ui frontend that consumes the serving API.

## Running

```bash
cd ui && npm install
cd ui && npm run dev      # http://localhost:3000
cd ui && npm run build    # production build
cd ui && npm run lint     # ESLint
```

## Architecture

```
src/
  app/
    layout.tsx       — Root layout, ThemeProvider, TooltipProvider, fonts
    page.tsx         — Main page: Simulation, Catalog, Benchmark, Feature Store tabs
    globals.css      — Turo-themed shadcn color palette (light + dark mode)
  components/
    theme-provider.tsx — next-themes wrapper (system/light/dark)
    error-boundary.tsx — React error boundary
    ui/              — shadcn components (card, button, slider, switch, etc.)
  lib/
    api.ts           — Typed API client (predict, benchmark, vehicles, features, stores)
    utils.ts         — shadcn cn() utility
```

## Tabs

| Tab | Description |
|-----|-------------|
| Simulation | Configure vehicle attributes and simulate reservation count |
| Catalog | Fleet vehicles (with history) and new arrivals (pending prediction) |
| Benchmark | Latency benchmark: raw features vs online store path |
| Feature Store | Operational view of offline (Parquet) and online (Redis) stores |

## Configuration

`NEXT_PUBLIC_API_URL` — Serving API base URL (default: `http://localhost:8000`).
Set in `.env.local` for local dev.

## Design

Turo-inspired visual identity:
- Primary color: Turo purple (`#593CFB`)
- Accent: Teal (`#00B8A9`)
- Dark mode toggle (system preference by default, via next-themes)
- Pill-shaped buttons, card-heavy layout, generous white space
- shadcn global theme (all colors via CSS custom properties in `globals.css`)
