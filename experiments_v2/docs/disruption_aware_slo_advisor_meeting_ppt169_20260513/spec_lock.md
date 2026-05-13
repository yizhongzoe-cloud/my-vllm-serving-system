# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## colors
- bg: #FFFFFF
- text: #111827
- text_secondary: #4B5563
- text_tertiary: #9CA3AF
- border: #D1D5DB
- subtle_fill: #F3F4F6

## typography
- font_family: Arial, "Helvetica Neue", sans-serif
- title_family: Georgia, "Times New Roman", serif
- body_family: Arial, "Helvetica Neue", sans-serif
- emphasis_family: Georgia, "Times New Roman", serif
- code_family: Consolas, "Courier New", monospace
- body: 18
- title: 32
- subtitle: 22
- annotation: 14
- cover_title: 56
- footnote: 11

## icons
- library: tabler-outline
- stroke_width: 2
- inventory: clock, skull, bolt, target, cube, router, cpu, database, refresh, hourglass, network, history, chart-bar, file-export, arrow-right, server

## page_rhythm
- P01: anchor
- P02: dense
- P03: dense
- P04: dense
- P05: breathing
- P06: anchor
- P07: dense
- P08: dense
- P09: dense
- P10: dense
- P11: dense
- P12: anchor

## forbidden
- Mixing icon libraries
- Any color other than the colors section above (no accent / no brand color)
- rgba()
- `<style>`, `class`, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<script>`, `<iframe>`, `<symbol>`+`<use>`
- `<g opacity>` (set opacity on each child element individually)
- HTML named entities in text (write `—`, `→`, `≥`, `≤`, `×`, `≈` as raw Unicode)
