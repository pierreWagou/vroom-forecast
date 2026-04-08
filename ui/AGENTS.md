# UI — Agent Instructions

Read README.md for full context.

You are a frontend engineer building a demo UI for an ML prediction service.

- Next.js App Router with TypeScript; all components from shadcn/ui
- Turo-inspired design: purple primary (#593CFB), teal accent, pill buttons
- Dark mode: next-themes with system preference default, class-based toggling
- All colors via shadcn global theme in `globals.css` — no per-component overrides
- API client in `lib/api.ts` — keep types in sync with `serving/schemas.py`
- Linted with eslint-config-next (TypeScript + Core Web Vitals)
- No Python in this directory; this is a pure npm project
- Tabs: Simulation (predict), Catalog (fleet + new arrivals), Benchmark, Feature Store

## SSE event streams

The UI uses two Server-Sent Events connections:

1. **`GET /events`** — model promotion events. Triggers health refresh on new champion.
2. **`GET /vehicles/events`** — vehicle materialization events. The FeatureMaterializer
   publishes to `vroom-forecast:vehicle-materialized` on Redis pub/sub when a new
   arrival's features are written to the online store. The SSE endpoint forwards
   these to the UI, replacing the previous polling approach.
