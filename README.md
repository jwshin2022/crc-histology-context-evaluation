# Bias and local contrast in colorectal gene expression prediction from neighboring histology

This software reproduces the manuscript's donor and section error decomposition, training-mean comparison, fixed three-seed summaries, direction-specific contrasts, shuffled-neighbor evaluation and spatial-block intervals from saved predictions and measured marker scores.

## Run

Install NumPy as recorded in requirements.txt, then run `python evaluate.py`. The script writes six CSV files under reproduced/. Compare these with expected_results/. Reproduction takes seconds on a CPU. No account, GPU, absolute local path or model download is required.

## Data and scope

The input archive contains two RNA-derived marker scores at conventional Visium measurements, image-grid coordinates, evaluation eligibility, training scores, fixed normalizers and saved predictions. The six assessment captures belong to three donors. Seeds are training realizations on these same donors. Absent molecular measurements are marked by eligible=false and must never be evaluated as observed zeros. Coordinates are micrometers; spot capture diameter is 55 µm and center spacing is 100 µm.

The original collection is Valdeolivas et al., npj Precision Oncology 8, 10 (2024), doi:10.1038/s41698-023-00488-4. Original data are archived at https://doi.org/10.5281/zenodo.7760264 under CC BY 4.0. The processed measurement subset retains attribution under CC BY 4.0. Data in this package are previously released de-identified research measurements.

The bootstrap uses 2,000 paired resamples of spatial blocks within each observed section. Widths 400, 800, 1200 µm and two grid origins are supplied. These are approximate conditional intervals for fixed models on observed sections. They do not estimate patient-population uncertainty or establish calibrated coverage. Cross-block dependence can remain.

## Software scope

This release reproduces the evaluation of saved predictions. Original-model training requires the raw histology, cached Phikon features and upstream environments. It does not claim to reproduce original-model training from this small archive. The shuffled run used a fixed within-capture permutation of neighbor feature identities in training and inference, seed 20261019, the original training/selection donors and 15 epochs.

## Licenses

Original code in this release is MIT licensed. Processed public data retain CC BY 4.0. Upstream model source and pretrained weights are not redistributed here. Upstream DeepSpot is MIT; COAST and HEXST source have CC BY-NC 4.0 terms. Consult their original repositories before reuse.

Authors: Jee-Woo Shin and Chung-Pyo Hong, Division of Computer Engineering, Hoseo University, Asan, Republic of Korea.

CSV ordering is not meaningful; compare rows after matching all identifier columns.

training_source/ contains the actual matched-model source adaptations for algorithm inspection; its README states the original-layout and feature dependencies.
