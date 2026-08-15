# HFSS Shearlet Colab

Public, reproducible processing pipeline for the validated 2.45 GHz HFSS cylindrical-phantom analysis. Gate 1 reconstructs the electric field, and Gate 2 performs the three-level discrete Shearlet analysis.

The repository code is public and the Colab workflow requires no GitHub account, token, or other GitHub credentials. The HFSS dataset is **not public** and must currently be supplied separately through Google Drive or manual upload. It is not included in this repository.

## Open in Colab

[![Open notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codebycristian-dev/hfss-shearlet-colab/blob/main/notebook/HFSS_Shearlet_2p45GHz.ipynb)

The notebook performs an unauthenticated shallow clone of `https://github.com/codebycristian-dev/hfss-shearlet-colab.git` from `main`.

## Validated method

The complete production flow is:

```text
HFSS Real/Imag .fld
    -> complex Ex, Ey, Ez
    -> Electric-field magnitude |E| [V/m]
    -> 40 numeric single-channel matrices
    -> unmasked numeric HFSS matrix
    -> reflect padding to 512x512
    -> 3-level Discrete Shearlet Transform
    -> RMS-normalized directional coefficients
    -> Level 1 / Level 2 / Level 3 aggregate responses
    -> full_phantom versus interior_1p5mm sensitivity analysis
    -> metrics and publication figures
```

Gate 1 is complete and production-validated on all 40 HFSS cuts. It pairs real and imaginary exports, reconstructs the complex components, and computes:

```text
|E| = sqrt(|Ex|^2 + |Ey|^2 + |Ez|^2)
```

The analyzed physical quantity is **Electric-field magnitude |E| [V/m]**. Physical V/m amplitudes are preserved; no independent per-image normalization is applied to scientific arrays.

Gate 2 is complete and production-validated with pyShearLab-MIND 0.0.3. It reads `unmasked_intensity_V_per_m` from the Gate 1 scientific NPZ files, reflect-pads the numeric matrix to 512x512, and performs no spatial resizing or interpolation. The system uses exactly three levels with `nScales=3`, `shearLevels=[1,1,2]`, and `full=0`: one low-pass plus 8, 8, and 16 directional filters at Levels 1, 2, and 3, respectively, for 33 filters total. The low-pass is excluded from the three aggregate response maps. Raw coefficients are retained for inverse reconstruction; RMS-normalized coefficients are used for comparative response and coefficient-energy analysis.

Scale interpretation is limited to the validated ordering:

- Level 1 = coarser spatial-scale band
- Level 2 = intermediate spatial-scale band
- Level 3 = finer spatial-scale band

No unvalidated physical spatial-frequency cutoff values are assigned. The `full_phantom` ROI is the Gate 1 solid mask. The `interior_1p5mm` ROI contains solid pixels with Euclidean distance greater than 1.5 mm from the exterior, using the physical axis spacings.

User-facing Gate 2 quantities are described as **RMS-normalized Shearlet response**, **aggregate Shearlet response magnitude**, and **Shearlet coefficient-energy fraction**.

> Shearlet coefficient energy is a signal-processing quantity based on squared RMS-normalized Shearlet coefficients. It is not electromagnetic energy and is not Poynting intensity.

## Scientific input and visualization rules

Scientific analysis uses numeric, single-channel NPZ arrays. PNG files are visualization artifacts only. PNG/RGB pixels are **never** transform input. No per-image normalization is used, and no spatial resizing or interpolation is used before Shearlet analysis.

Gate 1 stores both the masked `intensity_V_per_m` visualization matrix and the pre-mask `unmasked_intensity_V_per_m` scientific transform input, together with the solid mask, physical coordinate axes, cut metadata, and unchanged machine polarization metadata:

- Mode 1 — parallel to y: `parallel_y`
- Mode 2 — perpendicular to y: `perpendicular_y`

Gate 1 visualizations use shared scales. The `article_shared` images use one global-percentile `cividis` scale and a neutral masked exterior. Visualization clipping does not modify saved scientific matrices or metrics.

## Production outputs

The combined workflow writes:

```text
outputs/
├── 01_intensity/
│   ├── physical_shared/       # exactly 40 PNGs
│   ├── presentation_shared/   # exactly 40 PNGs
│   └── article_shared/        # exactly 40 PNGs
├── 02_numeric/                    # exactly 40 Gate 1 scientific NPZ matrices
├── 03_article_figures/            # Gate 1 publication PNG/PDF figures
├── 04_metrics/
│   ├── field_metrics.csv          # exactly 40 rows
│   └── run_metadata.json
└── 05_shearlet/
    ├── numeric/                   # exactly 40 scientific Shearlet NPZ artifacts,
    │                              # containing exactly 120 numeric level maps
    ├── level_maps/                # exactly 120 rendered PNG level maps
    ├── article_figures/           # Gate 2 publication PNG/PDF figures
    └── metrics/
        ├── shearlet_level_metrics.csv       # exactly 120 rows
        ├── shearlet_direction_metrics.csv   # exactly 120 rows
        └── shearlet_run_metadata.json
```

Metadata records the dataset filename and SHA-256, Git commit SHA, repository ref, software versions, full filter indices and configuration, filter counts, padding, ROI definitions, visualization percentiles, and scale interpretation.

## Reproducibility

1. Open the notebook in Colab using the badge above.
2. Select **Run all**.
3. Supply the original HFSS dataset ZIP through the supported Google Drive or manual-upload mechanism.
4. The notebook validates provenance and processes all 40 cuts.
5. Gate 1 and Gate 2 outputs are validated and packaged automatically into one downloadable ZIP.

The notebook recomputes the dataset SHA-256 and Git commit SHA for each run. Known NumPy deprecation warnings emitted by the pinned third-party pyShearLab-MIND package are non-fatal; the package source is not patched and warnings are not globally suppressed.
