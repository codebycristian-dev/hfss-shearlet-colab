# HFSS Shearlet Colab

Reproducible processing pipeline for the 2.45 GHz HFSS cylindrical phantom dataset.

## Gate 1 scope
This first gate intentionally implements only:

1. `.fld` parsing.
2. Real/imaginary pairing.
3. Complex electric-field reconstruction.
4. `|E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)`.
5. Physical solid masks for all 10 XZ and 10 YZ cuts.
6. 40 numeric single-channel electric-field magnitude matrices in V/m (20 cuts × 2 modes).
7. 40 `physical_shared` PNGs using the global in-solid maximum.
8. 40 `presentation_shared` PNGs using one global 99.5th-percentile limit.
9. Per-cut validation metrics and run-scale metadata under `outputs/04_metrics`.

**Shearlet processing is intentionally excluded until Gate 1 is audited.**

## Data
Do not commit the HFSS dataset to GitHub. During development, place the original dataset ZIP in Google Drive or upload it in Colab.

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
