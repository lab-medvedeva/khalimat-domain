import subprocess
import pandas as pd
from pathlib import Path


def run_study(runs_listening_file, dir_path):
    full_path = Path(dir_path) / runs_listening_file
    runs = pd.read_csv(full_path)
    template = "python main.py -p 'Datasets/' -f {dataset}' -sn {study_name} -m {models} -sa {sampling} -b {butina} -rs {run_sampling} -ss {scaf_split}"
    for run in runs:
        subprocess.run(template.format())
