# Brand assets

**The official Peaceable Street Capital logo** (Jim, Oct 8 2026: "This is our official logo").
Use these files; never re-extract a logo from a deck or a web page.

| File | What it is | Use it for |
|---|---|---|
| `docs/brand/Peaceable Street Logo5.jpg` | THE MASTER, as Jim supplied it: 2264 x 2290, **CMYK** JPEG with its ICC profile | print vendors; the source of every derivative. Do not edit it. |
| `docs/brand/psc-logo-srgb-full.png` | full-size sRGB conversion (through the embedded profile, perceptual intent) | documents, Excel / Word / PDF exports, emails |
| `vue_app/src/assets/brand/psc-logo.png` | 900 px sRGB PNG | the app: `import logo from '@/assets/brand/psc-logo.png'` (Vite hashes it). Board package cover uses it. |

**Why the conversion:** browsers handle CMYK JPEGs inconsistently (some draw them with shifted
colours, some not at all), so anything on screen uses the sRGB copies. To make another size, convert
from the master with Pillow's `ImageCms.profileToProfile(..., outputMode="RGB")` using the master's
embedded `icc_profile` -- a plain `.convert("RGB")` ignores the profile and dulls the greens.

Other copies exist in the Accounting team's SharePoint
(`x - Management/Administrative/Logo - PSC/`: `.ai`, `.eps`, high-quality `.pdf` / `.jpg`) if a vector
is ever needed.
