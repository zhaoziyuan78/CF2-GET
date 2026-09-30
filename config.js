/* ==========================================================================
   CF²-GET demo page · configuration
   --------------------------------------------------------------------------
   This is the only file you need to edit.

   1. Paste the GitHub repository URL into `links.code` below.
   2. Drop the MIDI files into the folders under `samples/` using the file
      names listed in README.md. The page finds them automatically.

   Anything left empty shows a tidy "coming soon" state on the public site.
   When you preview locally (http://localhost), every empty slot shows the
   exact path it expects, and a checklist in the bottom-right corner counts
   what is still missing.
   ========================================================================== */

window.CF2GET_CONFIG = {

  /* ---- 1. Links ---------------------------------------------------------- */
  links: {
    code: "https://github.com/zhaoziyuan78/CF2-GET",   // ← GitHub repo, e.g. "https://github.com/zhaoziyuan78/CF2-GET"
    paper: ""   // optional: arXiv / PDF link. The Paper button stays hidden while empty.
  },

  /* ---- 2. Transfer examples: our method vs. baselines -------------------- */
  // Each example lives in its own folder and holds one MIDI file per method
  // (file names are set in `methods` below). Add, remove or reorder freely.
  // Genre keys: classical · jazz · rock · pop · country · latin
  transferExamples: [
    { folder: "samples/transfer/01_classical-to-jazz", source: "classical", target: "jazz",      note: "" },
    { folder: "samples/transfer/02_jazz-to-rock",      source: "jazz",      target: "rock",      note: "" },
    { folder: "samples/transfer/03_rock-to-latin",     source: "rock",      target: "latin",     note: "" },
    { folder: "samples/transfer/04_pop-to-classical",  source: "pop",       target: "classical", note: "" },
    { folder: "samples/transfer/05_country-to-pop",    source: "country",   target: "pop",       note: "" },
    { folder: "samples/transfer/06_latin-to-country",  source: "latin",     target: "country",   note: "" }
  ],

  // One row per method in every transfer example, top to bottom.
  // `ablation: true` rows appear only when the visitor switches on the ablation toggle.
  methods: [
    { key: "source",          file: "source.mid",          label: "Source",           detail: "Original" },
    { key: "cf2get",          file: "cf2get.mid",          label: "CF²-GET",          detail: "Flow inversion", ours: true },
    { key: "midi-vae",        file: "midi-vae.mid",        label: "MIDI-VAE",         detail: "Symbolic VAE" },
    { key: "latent-midi-vae", file: "latent-midi-vae.mid", label: "Latent MIDI-VAE",  detail: "Latent VAE swap" },
    { key: "ddim",            file: "ddim-inversion.mid",  label: "DDIM Inversion",   detail: "Diffusion inversion" },
    { key: "wo-hsic",         file: "cf2get-wo-hsic.mid",  label: "CF²-GET w/o HSIC", detail: "Ablation", ablation: true }
  ],

  /* ---- 3. Citation -------------------------------------------------------- */
  bibtex:
`@inproceedings{zhao2027cf2get,
  title     = {{CF$^2$-GET}: Conditional Flow for Counterfactual Symbolic Music Genre Transfer},
  author    = {Zhao, Ziyuan and Yu, Ruiji and Hou, Dylan Xinming and Xia, Gus},
  booktitle = {Submitted to the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}`,

  /* ---- 4. Playback (normally no need to touch) ---------------------------- */
  // General MIDI soundfont used for playback. If it cannot be reached, the
  // page falls back to a built-in synthesizer so the samples still play.
  soundfont: "https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus"
};
