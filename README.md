# Tomato Irrigation Data Mining Project

## Overview
This project performs clustering and outlier detection on a tomato irrigation dataset using nutrient levels and pH vs. planting days.

## Structure
- `src/`: Python script for preprocessing, visualization, and KMeans clustering
- `data/`: Small dataset (`tomato_irrigation_dataset.csv`)
- `notebooks/`: Jupyter Notebook version for step-by-step analysis
- `run_all.sh`: Shell script to execute the project
- `requirements.txt`: Python dependencies (for pip)
- `environment.yml`: Conda environment definition (recommended)
- `output/`: Folder containing saved plots (when running the script)

## Setup

### 🔹 Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate tomato-env
```

### 🔹 Option 2: Using virtualenv + pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Project

```bash
bash run_all.sh
```

Or run directly:

```bash
python src/main.py #better
```

> All plots will be saved in the `output/` folder as PNG files.

## Run in Notebook

```bash
jupyter notebook notebooks/analysis.ipynb #better
```

## Software
- Python 3.10
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn,ipython

## Expected Runtime
~1–2 minutes depending on system
