# Docs parameter helper

Goal (Elise, 2026-11-21): "make parameter choosing easier for users ... maybe they upload a video
from their dataset and it will give them recommendations, or ... visual videos showing what type
of dataset each parameter choice would be best for".

Choices (Elise):
- Both: a helper that reads the user's files in the browser, plus a gallery of the training datasets.
- Inputs: detections `.slp`, their video, and CTC mask TIFFs.
- Settings: crop size, max tracks, max distance, anchor and orientation, max gap, model.
- Crop rule: **cover the whole animal** (median anchor-centred keypoint span, rounded up to a
  multiple of 8), although the pretrained models were trained with smaller crops (below).
- Same branch as the landing page (`elise/docs-landing-home`), one docs PR.

## Files

- `docs/parameters.md`: the page (Guides > Choosing Parameters): the helper, how the suggestions
  are made, and the gallery.
- `docs/js/param-helper/stats.js`: the rules, pure functions (no DOM).
- `docs/js/param-helper/app.js`: the page wiring: sleap-io.js 0.1.9 for `.slp` (as the Live
  Visualizer), geotiff 2.1.3 for TIFF masks (first 300 frames), the frame view with crop squares
  and a drag-to-measure box, the movement histogram, the command.
- `docs/css/param-helper.css`; `mkdocs.yml` nav; a tip linking the page in `docs/usage.md`.
- Here: `reference.py` (an independent Python version of the rules that also exports test
  fixtures) and `stats.test.mjs` (Node tests).

## How DREEM uses the settings (checked in the code, 2026-11-23)

- Crops are cut in native pixels around the anchor and never resized (`get_bbox`, `crop_bbox`).
- `anchors: centroid` = mean of the visible keypoints; a missing anchor node falls back to it.
- `max_center_dist` is in pixels; `max_tracks` caps how many tracks are ever created (then
  instances are forced onto existing tracks); `max_gap` ends a track after that many missing frames.

## Calibration (training labels, up to 4 videos / 3,000 frames each)

| Dataset | Per frame | Size (median, px) | Moves (99%, px) | Closest spacing (5%, px) | Trained crop |
|---|---|---|---|---|---|
| four / two / three / eight flies | 4 / 2 / 3 / 8 | 100 / 103 / 107 / 101 | 8 / 19 / 33 / 5 | 84 / 78 / 57 / 66 | 64 / 64 / 64 / 128 |
| btc | 2 | 273 | 35 | 79 | 64 |
| mice_hc | 2 | 257 | 73 | 85 | 64 |
| slap | 1-2 | 307 | 19 | 111 | 90 |
| zebrafish (centroids) | 10 | - | 28 | 56 | 64 |
| lysosomes / dynamicnuclearnet / phase contrast / DIC (centroids) | varies | - | 9 / 6 / 5 / 2 | 8 / 22 / 14 / 35 | 22 / 32 / 42 / 100 |

## Tests

```bash
F=$(mktemp -d)
uv run python docs_parameter_helper/reference.py $F demo_flies docs/assets/demos/animals.slp
uv run python docs_parameter_helper/reference.py $F demo_flies_untracked docs/assets/demos/animals.slp --untracked
uv run python docs_parameter_helper/reference.py $F demo_cells docs/assets/demos/microscopy.slp
FIXTURES=$F node --test docs_parameter_helper/
```

2026-11-23: 12/12 pass (4 unit tests; the 3 demo fixtures plus one file each of mice_hc, btc,
slap, zebrafish and DynamicNuclearNet). The real sleap-io.js 0.1.9 in Node reads both demo files
into the same structure as the Python export and gives the same suggestions.

## To do

- [x] Rules in `stats.js`, checked against `reference.py`
- [x] Page, styles, nav, gallery, link from the Usage guide
- [x] `mkdocs build`: same 386 warnings as main; page elements, images and CDN URLs resolve
- [ ] Try it in a browser: both demos, your own `.slp` + video, drawing a box, a CTC mask stack,
      light/dark mode, a narrow window. Not possible on the remote machine (no browser)
- [ ] The TIFF path (`loadMasks`, geotiff) has only been checked against geotiff's documented API
- [ ] Elise reviews the rules and the gallery wording; commit and PR only with her approval
