# Codex Task 01 — HFSS data ingestion and 40 intensity maps

Implement and harden Gate 1 for the repository. Do **not** implement Shearlet yet.

## Scientific context
The dataset contains Ansys HFSS frequency-domain complex electric-field exports at 2.45 GHz for a cylindrical muscle-equivalent phantom. For every cut, HFSS exported separate files for:
- `Real(<Ex,Ey,Ez>)`
- `Imag(<Ex,Ey,Ez>)`

Reconstruct:
`E = Re(E) + 1j*Im(E)`

The scalar field used for the exploratory presentation is:
`|E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)` in V/m.

There are exactly:
- 10 transverse XZ cuts T01–T10 at y = [-90,-70,-50,-30,-10,+10,+30,+50,+70,+90] mm.
- 10 longitudinal YZ cuts A01–A10 at x = [-45,-35,-25,-15,-5,+5,+15,+25,+35,+45] mm.
- 2 Floquet modes (M1/M2).
Therefore Gate 1 must produce exactly 40 phantom intensity images.

## Geometry
Cylinder axis: global y.
Radius: 50 mm.
Length: 200 mm, so |y| <= 100 mm.

XZ mask:
`x^2 + z^2 <= 50^2`.

YZ mask at fixed x:
`|y| <= 100` and `|z| <= sqrt(50^2 - x^2)`.

Outside-solid pixels must be masked/set to zero only after the original scientific matrix has been reconstructed.

## Requirements
1. Make the parser robust to whitespace and HFSS scientific notation.
2. Never assume HFSS row serialization order when reshaping; map coordinates explicitly.
3. Validate Real/Imag coordinate equality.
4. Fail fast on NaN/Inf.
5. Validate exactly 40 pairs before generating outputs.
6. Preserve numeric single-channel matrices separately from PNG visualization.
7. PNGs are presentation artifacts only; future transforms must never read RGB PNG pixels.
8. Generate `outputs/04_metrics/field_metrics.csv`.
9. Add tests for parser, pairing, geometry masks, reshaping, expected dataset discovery, and output count.
10. Make `notebook/HFSS_Shearlet_2p45GHz.ipynb` executable in Google Colab with:
   - Google Drive data option.
   - manual ZIP upload fallback.
   - Run All workflow.
   - a final Gate 1 assertion that exactly 40 intensity PNGs were generated.
11. Do not silently normalize each image independently. Preserve physical V/m amplitudes. If presentation normalization is added, label it explicitly and keep raw metrics.
12. Do not implement Shearlet in this task.

## Acceptance criteria
- `pytest -q` passes.
- Colab notebook can Run All after dataset path/upload is provided.
- Exactly 40 PNG intensity maps are generated.
- Exactly 40 metric rows are produced.
- No RGB conversion is used as scientific input.
- M1/M2 and XZ/YZ directory structure is preserved.
- Any data inconsistency produces a clear error rather than continuing.

When done, summarize files changed, tests run, and any assumptions.
