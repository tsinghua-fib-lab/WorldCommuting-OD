# WorldCommuting-OD: Origin-Destination Matrix Generation

A deep learning framework for generating origin-destination (OD) matrices using diffusion models and the graph transformer. This project focuses on predicting commuting patterns and population flows between geographic regions using satellite imagery, demographic data, POI data, and spatial relationships.

## Project Overview

This project implements a diffusion-based graph generative model to generate OD matrices for cities worldwide. It combines satellite imagery features, demographic information, and spatial distance data to generate realistic commuting patterns between different geographic regions.

## Directory Structure

```
GeneratingCodeData/
├── code/                   # Main source code directory
│   ├── main.py             # Main execution script and entry point
│   ├── model.py            # Neural network architectures and diffusion models
│   ├── train.py            # Training loop and optimization logic
│   ├── eval.py             # Evaluation and generation scripts
│   ├── data_load.py        # Data loading and preprocessing utilities
│   └── utils/              # Utility functions and helper modules
│       ├── tool.py         # General utility functions and data processing tools
│       ├── metrics.py      # Evaluation metrics and performance calculations
│       ├── MyLogger.py     # Logging and experiment tracking
│       └── procedure.py    # Procedural utilities and workflow functions
├── data/                   # Data storage directory
│   ├── cities_global_1625_shp/    # Global city shapefiles
│   ├── global_cities/             # Global city datasets
│   ├── GHSL_744_shp/              # GHSL (Global Human Settlement Layer) shapefiles
│   └── GHSL_744/                  # GHSL raster data
└── exp/                   # Experiment configuration and outputs
    ├── config/            # Configuration files for experiments
    └── model/             # Saved model checkpoints and outputs
```

## Usage

1. Configure experiment parameters in `exp/config/`
2. Prepare data in the required format under `data/`
3. Run training: `python code/main.py`

## Dependencies

- PyTorch
- DGL (Deep Graph Library)
- NumPy
- GeoPandas
- matplotlib
- Scikit-learn
- tqdm