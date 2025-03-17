import os
from pathlib import Path
import logging


list_of_files=[

    ".github/workflows",
    "preprocessing/__init__.py",
    "preprocessing/imputer.py",
    "preprocessing/scaler.py",
    "preprocessing/encoder.py",
    "preprocessing/dimensionality_reduction.py",
    "models/__init__.py",
    "models/classification/__init__.py",
    "models/classification/decision_tree.py",
    "models/classification/random_forest.py",
    "models/classification/logistics_regression.py",
    "models/classification/gradient_boosting.py",
    "models/classification/naive_bayes.py",
    "models/classification/gradient_boosting.py",
    "models/regression/linear_regression.py",
    "models/regression/decision_tree.py",
    "models/regression/random_forest.py",
    "models/regression/gradient_boosting.py",
    "vectorizer/vectorizer.py",
    "model_selection/split.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiment/experiments.ipynb"

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir,exist_ok= True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass


