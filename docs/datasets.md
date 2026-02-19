# Datasets

Here we provide information and download links for datasets used for training and evaluating DREEM across a range of tracking scenarios. Datasets are organized into two categories: **Animals** (behavioral tracking of whole organisms) and **Microscopy** (tracking of cells and subcellular structures). All datasets, including metadata, is available on [Google Drive](https://drive.google.com/drive/u/0/folders/0ACXbRsrn_M0JUk9PVA).

!!! hint

    Need a quick testing clip?
    Animals: [A short, simple video with four flies.](https://drive.google.com/file/d/1XLMmp8M5Xf9GYfq6BKZpge11or7EQWBh)
    Microscopy: [A short clip with slow moving cell nuclei.](https://drive.google.com/file/d/1eRbhxa1Iw-1ONhMp1kwD5J3KIsytG8CO)
---

## Summary

### Animals

| Dataset | Subject | Animals | Videos | Frames |
|---------|---------|---------|--------|--------|
| [`mice_btc`](#mice_btc) | Mice | 2 | 25 | ~1.05M |
| [`mice_hc`](#mice_hc) | Mice | 2 | 36 | ~31K |
| [`slap2m`](#slap2m) | Mice | 1–4 | 17 | ~294K |
| [`flies13`](#flies13) | Fruit Flies | 2–8 | 58 | ~94K |
| [`zebrafish10`](#zebrafish10) | Zebrafish | 10 | 18 | 27K |

### Microscopy

| Dataset | Subject | Objects per Frame | Videos | Frames |
|---------|---------|-------------------|--------|--------|
| [`lysosomes`](#lysosomes) | Lysosomes | 1–8 | 10 | ~3.9K |
| [`dynamicnuclearnet`](#dynamicnuclearnet) | Cell Nuclei | 3–249 | 130 | ~6.7K |
| [`motchallenge_dic`](#motchallenge_dic) | Cells (DIC) | 0–117 | 37 | ~80K |
| [`mouse_c2c12`](#mouse_c2c12) | Mouse C2C12 Cells | 2–95 | 19 | ~20K |

---

## Animals

### `mice_btc`
![mice_btc](assets/images/example.mice_hc.jpg)

| Name          | `mice_btc` |
|---------------|------------|
| Description   | Pairs of mice (*Mus musculus*) continuously monitored in a behavioral tracking chamber. Long-duration recordings captured simultaneously from multiple camera angles at high frame rates. |
| Videos        | 25 (19 train / 3 val / 3 test) |
| Image size    | 768 x 1024 x 1 |
| Num Animals   | 2 |
| Frames        | ~1.05M |
| Download      | [Google Drive](https://drive.google.com/drive/folders/12x5Fgs9I9MujuqaB1cVG-PDgzXuaRdTQ) |

---

### `mice_hc`
![mice_hc](assets/images/example.mice_of.jpg)

| Name          | `mice_hc` |
|---------------|-----------|
| Description   | Pairs of mice (*Mus musculus*) in a home cage setting, imaged from above. Short clips extracted from longer social interaction recordings. Animals can be low contrast against the bedding background. |
| Videos        | 36 (32 train / 3 val / 1 test) |
| Image size    | 1024 x 1280 x 1 |
| Num Animals   | 2 |
| Frames        | ~31K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1vTTM8LNT4cYqG8HpVmqJeQ67VgsXdzVY) |

---

### `slap2m`
![slap2m](assets/images/example.gerbils.jpg)

| Name          | `slap2m` |
|---------------|----------|
| Description   | Mice (*Mus musculus*) tracked using the SLAP2 two-photon imaging rig. Variable group sizes from single animals up to groups of 4, with long-duration continuous recordings. |
| Videos        | 17 (11 train / 2 val / 4 test) |
| Image size    | 1024 x 1280 x 1 |
| Num Animals   | 1–4 |
| Frames        | ~294K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1Pgfg-W9uv9Xe39jK2Aj2rUpqERlycTU0) |

---

### `flies13`
![flies13](assets/images/example.flies13.jpg)

| Name          | `flies13` |
|---------------|-----------|
| Description   | Groups of 2, 4, or 8 freely interacting fruit flies (*Drosophila melanogaster*) in circular arenas. Contains three sub-conditions (two-flies, four-flies, eight-flies) representing different group sizes. Annotated with a 13-point skeleton with identity. |
| Videos        | 58 (46 train / 7 val / 5 test) |
| Image size    | 1024 x 1024 x 1 |
| Num Animals   | 2–8 |
| Frames        | ~94K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1JpyktfS6Pr3yAYNS7QZ3wqKWO0araMmI) |

---

### `zebrafish10`
![zebrafish10](assets/images/example.bees.jpg)

| Name          | `zebrafish10` |
|---------------|---------------|
| Description   | Groups of 10 zebrafish (*Danio rerio*) freely swimming, imaged at very high spatial resolution. All clips contain exactly 10 tracked individuals throughout. |
| Videos        | 18 (6 train / 6 val / 6 test) |
| Image size    | 3712 x 3712 x 1 |
| Num Animals   | 10 |
| Frames        | 27K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1hLxTgByAxkQA8gFDCpCALSgKb-O4VQcy) |

---

## Microscopy

### `lysosomes`
![lysosomes](assets/images/example.fly32.jpg)

| Name          | `lysosomes` |
|---------------|-------------|
| Description   | Lysosomal organelles in live cells imaged with Airyscan confocal microscopy. Small fields of view with a variable number of organelles per video. Organelles exhibit rapid, non-linear motion. |
| Videos        | 10 (3 train / 1 val / 6 test) |
| Image size    | 250 x 250 x 1 |
| Num Objects   | 1–8 per frame |
| Frames        | ~3.9K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1OVNifmkSE3O_KVWy1LxI2kEUCDpCWdca) |

---

### `dynamicnuclearnet`
![dynamicnuclearnet](assets/images/example.mice_of.jpg)

| Name          | `dynamicnuclearnet` |
|---------------|---------------------|
| Description   | Fluorescently labeled cell nuclei from the DynamicNuclearNet tracking benchmark. Spans a wide range of cell densities and mitotic activity across multiple experimental conditions. |
| Videos        | 130 (91 train / 27 val / 12 test) |
| Image size    | 592 x 608 x 1 |
| Num Objects   | 3–249 per frame |
| Frames        | ~6.7K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1XGG8Z56P0BeG80GYdOSrh5tUYrTnQykA) |

---

### `motchallenge`
![motchallenge](assets/images/example.bees.jpg)

| Name          | `motchallenge` |
|---------------|--------------------|
| Description   | Cell tracking sequences from the MOT Challenge benchmark imaged with Differential Interference Contrast (DIC) microscopy. Spans multiple mammalian cell lines with varying morphology and density, including dividing and migrating cells. |
| Videos        | 37 (27 train / 3 val / 7 test) |
| Image size    | 320 x 400 x 1 |
| Num Objects   | 0–117 per frame |
| Frames        | ~80K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1Ua3ZC2gYbV2HukJaXfQI2TmUJTdFQxR5) |

---

### `phase_contrast`
![phase_contrast](assets/images/example.gerbils.jpg)

| Name          | `phase_contrast` |
|---------------|---------------|
| Description   | Mouse C2C12 myoblast cells imaged with phase contrast microscopy over long recording periods. High-resolution images with many cells per frame undergoing mitosis and migration. |
| Videos        | 19 (14 train / 1 val / 4 test) |
| Image size    | 1040 x 1392 x 1 |
| Num Objects   | 2–95 per frame |
| Frames        | ~20K |
| Download      | [Google Drive](https://drive.google.com/drive/folders/1sTTNtdOeYHkchptevMVAhoimd84UgoPm) |
