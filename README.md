<!-- markdownlint-disable MD013 -->

# Yarlung Zangbo River routing model uncertainty analysis

Manuscript source plus data-processing and plotting scripts for an academic paper. The manuscript is authored in Markdown and rendered to DOCX via Pandoc.

The published paper is available at <https://doi.org/10.1029/2024WR038564>.

## Repository structure

- `main.md`: Manuscript source file in Markdown format.
- `si.md`: Supporting Information source file in Markdown format.
- `references.bib`: Bibliography file in BibTeX format.
- `fig/`: Directory containing figure image files.
- `data/`: Directory containing processed data files used in the analysis.
- `scripts/`: Directory containing data-processing and plotting scripts (e.g., Python, R, or shell scripts).

## Scripts

Key Python scripts for data processing and model execution include:

- `wrfout_surface_vars_extract.py`: Extracts surface variables from WRF output files (`wrfout`).
- `wrfout_collect.py`: Merges multiple extracted surface variable files into a single file along an ensemble axis.
- `gauge_network.py`: Builds the gauge network topography based on the river network topography and gauge locations. It identifies upstream and downstream gauges, corresponding river reaches, and the reaches connecting a gauge to its upstream counterpart.
- `lateralflow_create.py`: Generates the lateral flow data used to drive the river routing model.
- `routing_common.py`: Contains core functions for the routing model.
- `routing.py`: Implements the simple Muskingum routing model.
- `routing_calibration.py`: Calibrates celerity parameters for each gauge, processing from upstream to downstream.
- `plot_calibration.py`: Plots calibration results and generates the `celerity_measurement.csv` file.
- `routing_ensemble_generate.py`: Uses the `celerity_measurement.csv` to generate an ensemble of routing parameters in a template directory.
- `routing_ensemble.py`: Executes the ensemble of routing simulations.

## Manuscript editing

- Keep section structure and citation keys stable unless a change is requested.
- Citations should use BibTeX keys (e.g., `@SomeKey`) and entries live in `references.bib`.
- Meaning-preserving edits:
  - Polish: grammar/typos/clarity without changing meaning.
  - Rephrase: polish + modest restructuring for flow (still meaning-preserving).

## Automation and reproducibility guidelines

If you use AI tools or automated scripts to modify

### When to ask

Clarify first if it’s unclear whether `submission*/` directories should be updated (otherwise treat them as archival).
