# UI — Agent Instructions

Read README.md for full context.

You are a frontend engineer building a demo UI for an ML prediction service.

- Next.js App Router with TypeScript; all components from shadcn/ui
- Turo-inspired design: purple primary (#593CFB), teal accent, pill buttons
- All colors via shadcn global theme in `globals.css` — no per-component overrides
- API client in `lib/api.ts` — keep types in sync with `serving/schemas.py`
- Linted with eslint-config-next (TypeScript + Core Web Vitals)
- No Python in this directory; this is a pure npm project
