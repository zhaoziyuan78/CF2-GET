# CF²-GET demo page

Static demo page for *CF²-GET: Conditional Flow for Counterfactual Symbolic Music Genre Transfer* (submitted to ICASSP 2027).
It needs no build step: GitHub Pages serves the folder as it is.

## What you fill in

There are only two things to add.

### 1. GitHub repo link

In `config.js`, fill in `links.code`:

```js
links: {
  code: "https://github.com/zhaoziyuan78/CF2-GET",   // ← your repo
  paper: ""                                           // optional arXiv/PDF link; the Paper button stays hidden while empty
},
```

The link is used by the **Code** button in the top bar, the **Code on GitHub** button in the hero, and the footer.

### 2. MIDI files

Put each file at the path below. File names are case-sensitive on GitHub Pages.

**Transfer examples** (six methods per example; the page shows one row per file, in this order):

| File | Row on the page |
|---|---|
| `source.mid` | Source (original passage) |
| `cf2get.mid` | CF²-GET (ours) |
| `midi-vae.mid` | MIDI-VAE |
| `latent-midi-vae.mid` | Latent MIDI-VAE |
| `ddim-inversion.mid` | DDIM Inversion |
| `cf2get-wo-hsic.mid` | CF²-GET w/o HSIC (shown only when the visitor turns on the ablation toggle; optional) |

…inside each of these folders:

```
samples/transfer/01_classical-to-jazz/
samples/transfer/02_jazz-to-rock/
samples/transfer/03_rock-to-latin/
samples/transfer/04_pop-to-classical/
samples/transfer/05_country-to-pop/
samples/transfer/06_latin-to-country/
```

Any slot you leave empty shows “Sample coming soon” on the public site, so you can publish before every file is ready.

## Changing the examples

Everything is listed in `config.js`:

- **Different genre pairs.** Edit `source` / `target` of an entry in `transferExamples` (genre keys: `classical`, `jazz`, `rock`, `pop`, `country`, `latin`). Rename its folder to match if you like; the folder name is just the `folder` value.
- **More examples.** Copy a line in `transferExamples`, point `folder` at a new folder, and add the files.
- **Fewer examples.** Delete the line.
- **A caption for one example.** Fill in its `note`.
- **Files named `.midi` or differently.** Change `file` in the `methods` list.
- **Hide the ablation.** Delete the `wo-hsic` line from `methods`; the toggle disappears with it.

## Previewing locally

Browsers refuse to load MIDI files into a page opened straight from disk, so start a small server in this folder:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>. While you preview on `localhost`, every empty slot shows the exact path it expects, and a **Sample checklist** in the bottom-right corner counts the files that are in place and lists the missing ones. The checklist never appears on the public site. (Add `?dev` to the public URL if you want to see it there.)

## Publishing on GitHub Pages

The paper points to `https://zhaoziyuan78.github.io/CF2-GET/`, which is the Pages site of the `CF2-GET` repository.

The MIDI examples are XMIDI excerpts and outputs derived from them. The code's MIT License does not cover them, so keep them out of the `main` branch and publish the page from its own branch:

1. In the repo, create an empty branch for the site: `git switch --orphan gh-pages`.
2. Copy the contents of this folder to the root of that branch (keep `.nojekyll`), commit and push.
3. On GitHub: **Settings → Pages → Build and deployment → Deploy from a branch**, choose `gh-pages` and `/ (root)`.
4. After a minute the page is live at the URL above.

## How playback works

- Piano rolls are drawn directly from the MIDI files. All rows of one example share the same bar grid and pitch range, so the melody can be compared row by row, even if two files were written at different tempi. Drum hits appear as ticks along the bottom lane.
- Audio uses the SGM+ General MIDI soundfont through Magenta.js, loaded from a CDN on first use, so multi-instrument files (including drums on channel 10) play with their own instruments. If the soundfont cannot load, the page falls back to a simple built-in synthesizer and says so.
- Each row also has a download button for the MIDI file.

## Files

```
index.html          page structure and text
config.js           links, example list, method list, BibTeX  ← the file you edit
assets/style.css    styles (light and dark)
assets/app.js       example cards, piano rolls, player
assets/midi.js      MIDI parser and audio engine
assets/hero.js      the animated flow figure at the top
assets/favicon.svg
samples/            your MIDI files
.nojekyll           tells GitHub Pages to serve files as they are
```
