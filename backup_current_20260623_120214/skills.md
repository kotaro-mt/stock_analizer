# Skills used by stock_future

## frontend-design (anthropics/skills)

**Location:** [.claude/skills/frontend-design/SKILL.md](.claude/skills/frontend-design/SKILL.md)
**Upstream:** https://github.com/anthropics/skills/tree/main/skills/frontend-design

Purpose — create distinctive, production-grade frontend interfaces that
avoid generic "AI slop" aesthetics. Commit to a clear conceptual
direction and execute with precision.

### Non-negotiables from the skill

- Pick ONE aesthetic direction and execute it with intentionality
- Refined minimalism needs restraint and precision
- **Avoid Inter / Roboto / Arial / system fonts** — use characterful type
- Dominant colours with sharp accents, not timid evenly-distributed palettes
- Avoid cliches like purple gradients on white backgrounds
- Match code complexity to aesthetic vision

### Chosen direction: 京都ターミナル (Kyoto Terminal)

A Japanese-editorial × financial-terminal hybrid. Refined, data-first,
authoritative, with Japanese editorial gravitas. Fits a tool built
primarily around 東証プライム 銘柄 analysis — the Japanese context
earns a Japanese visual language.

**Typography (Google Fonts):**
- Display: **Shippori Mincho** — traditional mincho serif for headings
- Body: **IBM Plex Sans JP** — readable across kana / kanji / Latin
- Numerics: **IBM Plex Mono** — tabular figures for prices and ratios

**Palette — washi paper × sumi ink × shu vermilion:**

| Token | Hex | Usage |
|---|---|---|
| paper | `#F4EFE3` | page background (washi) |
| surface | `#FBF8EF` | card / chart surfaces |
| ink | `#14110E` | text, borders, structural lines |
| ink-muted | `#5C554A` | secondary labels, axis ticks |
| shu (朱) | `#B7362E` | signature accent, rising candles, resistance |
| forest | `#2E6B47` | falling candles, support lines |
| navy (紺) | `#1E3A5F` | rising trend lines, indicators |
| copper | `#8A6E3A` | falling trend lines, MACD signal |
| gold | `#9B7421` | all-time high / low markers |

**Details:**
- SVG grain overlay on the page for paper texture
- Square corners (no border-radius) — editorial print feel
- Hard offset shadows `3px 3px 0 ink` — stamped-ink look, not soft drop shadows
- Thin ink horizontal rules as editorial dividers
- Wide-tracked small-caps vermilion labels for sidebar section headers
- Hanko (印) square accent next to the hero title, rotated -2°
- Tabular numerics on every price, ratio, percentage, delta
- Card hover reveals a vermilion bottom rule via scaleX transform
