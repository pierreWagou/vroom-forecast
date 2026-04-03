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
    layout.tsx       — Root layout, TooltipProvider, fonts
    page.tsx         — Main page: predict form + benchmark tabs
    globals.css      — Turo-themed shadcn color palette
  components/ui/     — shadcn components (card, button, slider, switch, etc.)
  lib/
    api.ts           — Typed API client (predict, health, benchmark)
    utils.ts         — shadcn cn() utility
```

## Configuration

`NEXT_PUBLIC_API_URL` — Serving API base URL (default: `http://localhost:8000`).
Set in `.env.local` for local dev.

## Design

Turo-inspired visual identity:
- Primary color: Turo purple (`#593CFB`)
- Accent: Teal (`#00B8A9`)
- Pill-shaped buttons, card-heavy layout, generous white space
- shadcn global theme (all colors via CSS custom properties in `globals.css`)
