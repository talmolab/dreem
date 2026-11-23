# Docs landing home page

Goal (Elise, 2026-11-20): make https://dreem.sleap.ai/ more user friendly, like https://sleap.ai/.

Choices:
- A landing-style home page inside the existing MkDocs Material site, using a custom `home.html`
  template. The deploy pipeline, mike versions and PR previews stay as they are.
- Sections: a hero (name, one-line description, buttons, demo videos with tracked identities).
  The rest of the home page text stays.
- Top navigation tabs: Home | Get started | Guides | Examples | API.

How the site is published: `.github/workflows/docs.yml` runs `mike deploy dev` on each push to
main, and `mike deploy <tag> latest` when a release is published. Only a release (or a manual
`mike deploy ... latest --push`) changes https://dreem.sleap.ai/latest/.

## To do

- [x] Branch `elise/docs-landing-home` from `origin/main`
- [x] Hero clips (`docs/assets/hero/`, 0.9 MB with posters): the visualizer's demo labels
      (`docs/assets/demos/*.slp`) rendered with `dreem.io.visualize.render_slp_video`, no ID labels.
      Animals: first 16 s, 512 px. Microscopy: all 63 s at 4x speed (the cells move a few pixels
      per second), trails of 150 frames
- [x] `overrides/home.html` (hero; content block without Material's fallback h1) +
      `docs/css/home.css` (loaded on the home page only); `docs/index.md` uses the template and
      hides the sidebar and table of contents; its old h1 is dropped (the hero holds the name)
- [x] Group `nav` into tabs (Home, Get Started, Guides, Examples, API, Changelog); enable
      `navigation.tabs`
- [x] Add `overrides/**` to the docs workflows' path filters
- [x] `mkdocs build`: same 386 warnings as `origin/main` (all pre-existing); hero links, clips
      and posters resolve; one h1 on the home page; tabs mark the active section
- [x] Docs caught up with code merged since 0.5.1 (audit 2026-11-20: every `dreem` command and
      option against `docs/cli.md`, default config keys against `docs/`): `dreem system` (#168);
      `--max-detection-overlap`, `--dilation-radius`, `--mult-thresh` on track/eval (#149); the CSV
      `flagged` column and CTC `.flagged_frames.txt` (#178); the `output_format` config key
      (`docs/configs/inference.md`). Build warnings unchanged (386, as main)
- [x] Get Started (2026-11-23, Elise: "more emphasis on what the user must do, and easier to
      follow ui"): Quickstart rewritten as six guided steps (install, try the sample, prepare
      your data, choose parameters, track, check results) with a step strip, "You should see"
      and "If it fails" boxes, and a checklist of the files a user must bring; Installation
      restyled with tabs (uv / uv tool / pip; NVIDIA / Apple / CPU). Quickstart first in the tab;
      Choosing Parameters listed under both Get Started and Guides (the Guides tab is the one
      highlighted on that page). The sample commands were run here on a CPU: download 6 s,
      `dreem track` 14 min 20 s, 2 tracks over 2,560 frames; the crop stays 70 (Elise), with a
      note that the helper suggests 104 for these 102 px flies
- [x] Contact (2026-11-23, Elise: Salk email only, on the home page and in the footer, a little
      more highlighted): the home page's Get Help section is a "Questions or feedback?" box with
      an "Email edavis@salk.edu" button (primary) and an "Open an issue" button; an envelope icon
      (mailto, title "Email edavis@salk.edu") follows the GitHub and X icons in every page's
      footer. Build warnings unchanged (386)
- [ ] Look at the page in a browser (light/dark, narrow screen): not possible on this machine
      (no browser; installing one was blocked). Via the PR preview at dreem.sleap.ai/pr/<n>/,
      or `uv run mkdocs serve`
- [ ] Elise reviews; commit and PR only with her approval
- Not confirmed: whether the demo `.slp` tracks are DREEM's output (no provenance in the files);
  the captions say "tracked" without naming the tracker
