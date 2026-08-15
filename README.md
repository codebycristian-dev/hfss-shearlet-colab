# HFSS Shearlet Colab

Reproducible processing pipeline for the 2.45 GHz HFSS cylindrical phantom dataset.

The repository code is public. No GitHub account, token, or other GitHub credentials are required to run the Colab notebook. The HFSS dataset is **not public** and must currently be supplied separately from Google Drive or by manual upload.

## Open in Colab

[![Open Gate 2 development notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codebycristian-dev/hfss-shearlet-colab/blob/gate2-shearlet/notebook/HFSS_Shearlet_2p45GHz.ipynb)

During Gate 2 validation the notebook explicitly clones `REPO_REF = "gate2-shearlet"`. **Immediately before the validated branch is merged to `main`, change `REPO_REF` and this badge back to `main`.**

1. Open the notebook using the badge above.
2. Set `DATA_SOURCE` to `"DRIVE"` or `"UPLOAD"` in the data-source cell.
3. For `"DRIVE"`, place the dataset ZIP in Google Drive and update `DATA_ZIP` if necessary. For `"UPLOAD"`, select the dataset ZIP when prompted.
4. Run all cells. The notebook clones the public repository directly and does not request GitHub credentials.

## Gate 1 — HFSS reconstruction
Gate 1 implements:

1. `.fld` parsing.
2. Real/imaginary pairing.
3. Complex electric-field reconstruction.
4. `|E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)`.
5. Physical solid masks for all 10 XZ and 10 YZ cuts.
6. 40 numeric single-channel electric-field magnitude matrices in V/m (20 cuts × 2 modes).
7. 40 `physical_shared` PNGs using the global in-solid maximum.
8. 40 `presentation_shared` PNGs using one global 99.5th-percentile limit.
9. 40 publication-ready `article_shared` PNGs plus shared-scale mosaics under `outputs/03_article_figures`.
10. Per-cut validation metrics and run-scale metadata under `outputs/04_metrics`.

Gate 1 remains the authoritative HFSS reconstruction stage and is not changed by Gate 2.

## Gate 2 — three-level discrete Shearlet analysis

Gate 2 reads only `outputs/02_numeric/**.npz` and transforms the
`unmasked_intensity_V_per_m` arrays with pyShearLab-MIND 0.0.3. It uses exactly
three scales (`shearLevels=[1,1,2]`), reflect-pads without resizing, RMS-normalizes
the directional coefficients, and excludes the low-pass filter from the three
scale-response maps. Level 1 is the coarser spatial-scale band, Level 2 is the
intermediate band, and Level 3 is the finer band; exact physical frequency cutoffs
have not been calibrated. The phantom mask is applied only after decomposition for the
full-phantom and physical 1.5-mm-interior analyses.

“Coefficient energy” means a discrete sum of squared RMS-normalized Shearlet
coefficients. It is not electromagnetic energy and is not Poynting intensity.

Run Gate 2 after Gate 1 with:

```python
from src.shearlet_analysis import run_shearlet_pipeline
run_shearlet_pipeline("outputs")
```

Artifacts are written under `outputs/05_shearlet`: 40 scientific NPZ files
containing exactly three full-resolution numeric maps each, 120 shared-per-level
PNG maps, level and directional metric CSV files, article figures, and complete
run metadata. The Colab notebook runs both gates and creates one downloadable ZIP.

## Data
The HFSS dataset is not included in this public repository. Do not commit it to GitHub. During development, place the original dataset ZIP in Google Drive or upload it manually in Colab.

Expected phantom data:
- Mode 1: T01–T10 + A01–A10, real and imaginary.
- Mode 2: T01–T10 + A01–A10, real and imaginary.
- Frequency: 2.45 GHz.
- Coordinates: mm.
- Electric field: V/m.
- Mode 1 polarization: `parallel_y` (electric field predominantly parallel to the cylinder y-axis).
- Mode 2 polarization: `perpendicular_y` (electric field predominantly perpendicular to the cylinder y-axis).

## Scientific rule
The future shearlet transform must receive the numeric **single-channel 2D electric-field magnitude matrix**, never the RGB pixels of a plotted PNG. Here, $|E|$ is electric-field magnitude in V/m; it is not Poynting intensity.

Numeric matrices are stored under `outputs/02_numeric/Mode{1,2}/{XZ,YZ}` as
compressed NumPy files. Each contains the masked and pre-mask electric-field magnitude matrices,
the solid mask, and physical coordinate axes. PNGs under `outputs/01_intensity`
are rendered visualization artifacts only—not scientific transform inputs. Neither
visualization scale clips or normalizes the saved scientific matrices or their metrics.
Run metadata records the actual input ZIP filename and computed SHA-256; no expected
dataset hash is hard-coded.

`outputs/01_intensity/article_shared` is for publication-quality visualization only.
It uses a shared global-percentile `cividis` scale and a neutral light-gray masked
exterior; it does not use per-image normalization or image enhancement. Scientific
analysis and any future Shearlet input continue to use only the numeric single-channel
NPZ matrices under `outputs/02_numeric`.
