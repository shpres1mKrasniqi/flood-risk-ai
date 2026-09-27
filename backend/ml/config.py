"""Central configuration for the ML component: paths, dataset schema and class labels."""

import os
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent       
REPO_ROOT = ML_ROOT.parents[1]                    

DATASET_PATH = Path(
    os.getenv("FLOOD_DATASET_PATH", REPO_ROOT / "data" / "raw" / "DataSeti_Finali.xlsx")
)

EXPERIMENTS_DIR = ML_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"

ID_COLUMN = "municipality"
TARGET = "risk"

NUMERIC_FEATURES: list[str] = [
    "elevation",
    "distance_from_river",
    "rainfall",
    "max_water_level",
    "min_water_level",
    "min_slope",
    "max_slope",
]
CATEGORICAL_FEATURES: list[str] = ["soil_type"]
FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

OPTIONAL_COLUMNS: list[str] = ["data_year"]

UNITS: dict[str, str] = {
    "elevation": "m",
    "distance_from_river": "km",
    "rainfall": "mm",
    "max_water_level": "mm",
    "min_water_level": "mm",
    "min_slope": "%",
    "max_slope": "%",
}

HEADER_PREFIXES: dict[str, list[str]] = {
    "municipality": ["qyteti", "municipality"],
    "elevation": ["lartesia mbidetare", "elevation"],
    "distance_from_river": ["distanca nga lumi", "distance_from_river"],
    "rainfall": ["sasia e reshjeve", "rainfall"],
    "soil_type": ["lloji kryesore i tokes", "soil_type"],
    "max_water_level": ["niveli max i ujit", "max_water_level"],
    "min_water_level": ["niveli min i ujit", "min_water_level"],
    "min_slope": ["pjerrtesia min", "min_slope"],
    "max_slope": ["pjerrtesia max", "max_slope"],
    "risk": ["rreziku", "risk"],
    "data_year": ["viti", "data_year"],
}

RISK_LABELS: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}