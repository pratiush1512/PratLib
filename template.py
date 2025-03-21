import os
from pathlib import Path
import logging


list_of_files=[

    ".github/workflows",
    "pratlib/preprocessing/__init__.py",
    "pratlib/preprocessing/imputer.py",
    "pratlib/preprocessing/scaler.py",
    "pratlib/preprocessing/encoder.py",
    "pratlib/preprocessing/dimensionality_reduction.py",
    "pratlib/models/__init__.py",
    "pratlib/models/classification/__init__.py",
    "pratlib/models/classification/decision_tree.py",
    "pratlib/models/classification/random_forest.py",
    "pratlib/models/classification/logistics_regression.py",
    "pratlib/models/classification/gradient_boosting.py",
    "pratlib/models/classification/naive_bayes.py",
    "pratlib/models/classification/gradient_boosting.py",
    "pratlib/models/regression/linear_regression.py",
    "pratlib/models/regression/decision_tree.py",
    "pratlib/models/regression/random_forest.py",
    "pratlib/models/regression/gradient_boosting.py",
    "pratlib/vectorizer/vectorizer.py",
    "pratlib/model_selection/split.py",
    "pratlib/tests/unit/__init__.py",
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


