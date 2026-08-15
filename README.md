# HFSS Shearlet Colab

Reproducible processing pipeline for the 2.45 GHz HFSS cylindrical phantom dataset.

## Gate 1 scope
This first gate intentionally implements only:

1. `.fld` parsing.
2. Real/imaginary pairing.
3. Complex electric-field reconstruction.
4. `|E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)`.
5. Physical solid masks for all 10 XZ and 10 YZ cuts.
6. 40 single-channel intensity visualizations (20 cuts × 2 modes).
7. Per-cut validation metrics.

**Shearlet processing is intentionally excluded until Gate 1 is audited.**

## Data
Do not commit the HFSS dataset to GitHub. During development, place the original dataset ZIP in Google Drive or upload it in Colab.

Expected phantom data:
- Mode 1: T01–T10 + A01–A10, real and imaginary.
- Mode 2: T01–T10 + A01–A10, real and imaginary.
- Frequency: 2.45 GHz.
- Coordinates: mm.
- Electric field: V/m.

## Scientific rule
The future shearlet transform must receive the numeric **single-channel 2D intensity matrix**, never the RGB pixels of a plotted PNG.
