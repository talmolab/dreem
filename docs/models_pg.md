# Pretrained Models

DREEM provides pretrained models for multi-object tracking across two domains: animals and microscopy. Both models are available on Hugging Face and can be used directly with the `dreem` CLI using shortnames.

## Quick Usage

Use pretrained models by passing a shortname to `--checkpoint`. The model is automatically downloaded from HuggingFace and cached locally (in `~/.cache/huggingface/hub/`):

```bash
# Animal tracking (mice, flies, zebrafish)
dreem track ./data --checkpoint animals --output ./results --crop-size 70

# Microscopy tracking (cells, organelles)
dreem track ./data --checkpoint microscopy --output ./results --crop-size 128
```

---

## Summary

| Shortname | Domain | Training Data | Hugging Face |
|-----------|--------|---------------|--------------|
| `animals` | Animals | ~1M frames across multiple species | [talmolab/dreem-animals-pretrained](https://huggingface.co/talmolab/dreem-animals-pretrained) |
| `microscopy` | Microscopy | ~100K frames across cells and organelles | [talmolab/dreem-microscopy-pretrained](https://huggingface.co/talmolab/dreem-microscopy-pretrained) |

---

## Animals

A general-purpose animal identity tracking model trained on ~1 million frames of proofread, identity-tracked public data spanning multiple species and scales, from fruit flies to mice. Note that you do not need the config to train the model; the CLI has many options that work without having to use a config file. See [CLI documentation](cli.md) for available options.

| | |
|---|---|
| **Shortname** | `animals` |
| **Input format** | Videos with detection labels in `.slp` format |
| **Training data** | ~1M frames ([datasets](datasets.md)) |
| **Hardware** | 4x A40 GPUs |
| **Metrics** | CLEARMOT ([py-motmetrics](https://github.com/cheind/py-motmetrics)) |
| **Training config** | [animals-pretrained-config.yaml](https://huggingface.co/talmolab/dreem-animals-pretrained/blob/main/animals-pretrained-config.yaml) |
| **Download** | [animals-pretrained.ckpt](https://huggingface.co/talmolab/dreem-animals-pretrained/blob/main/animals-pretrained.ckpt) |

```bash
dreem track ./data --checkpoint animals --output ./results --crop-size 70
```

---

## Microscopy

A general-purpose microscopy identity tracking model trained on ~100K frames of proofread, identity-tracked public data spanning diverse biological scales, from organelles to cell nuclei.

| | |
|---|---|
| **Shortname** | `microscopy` |
| **Input format** | Videos with detection labels in `.slp` or Cell Tracking Challenge format |
| **Training data** | ~100K frames ([datasets](datasets.md)) |
| **Hardware** | 4x A40 GPUs |
| **Metrics** | CLEARMOT ([py-motmetrics](https://github.com/cheind/py-motmetrics)), CTC ([py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics)) |
| **Config** | [microscopy-pretrained-config.yaml](https://huggingface.co/talmolab/dreem-microscopy-pretrained/blob/main/microscopy-pretrained-config.yaml) |
| **Download** | [pretrained-microscopy.ckpt](https://huggingface.co/talmolab/dreem-microscopy-pretrained) |

```bash
dreem track ./data --checkpoint microscopy --output ./results --crop-size 128
```
