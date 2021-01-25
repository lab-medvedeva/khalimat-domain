import subprocess
import pandas as pd
from pathlib import Path


def run_study(runs_listening_file, dir_path):
    full_path = Path(dir_path) / runs_listening_file
    runs = pd.read_csv(full_path)
    for run in runs.index:
        st_name, model, sampling = str(runs.iloc[:, 0].loc[run]), str(runs.iloc[:, 1].loc[run]), str(runs.iloc[:, 2].loc[run])
        ds, split = str(runs.iloc[:, 3].loc[run]), str(runs.iloc[:, 4].loc[run])
        if sampling == 'No sampling' and split == 'train_test_split':
            subprocess.Popen(['python', 'main.py', '-p', 'Datasets/', '-f', '{}'.format(ds), '-sn', '{}'.format(st_name),
                              '-m', '{}'.format(model), '-sa', 'No sampling'])
        if sampling != 'No sampling' and split == 'train_test_split':
            subprocess.Popen(
                ['python', 'main.py', '-p', 'Datasets/', '-f', '{}'.format(ds), '-sn', '{}'.format(st_name),
                 '-m', '{}'.format(model), '-sa', '{}'.format(sampling), 'rs'])
        if sampling != 'No sampling' and split == 'split_with_butina':
            subprocess.Popen(
                ['python', 'main.py', '-p', 'Datasets/', '-f', '{}'.format(ds), '-sn', '{}'.format(st_name),
                 '-m', '{}'.format(model), '-sa', '{}'.format(sampling), '-b', '-rs'])
        if sampling != 'No sampling' and split == 'split_with_scaffold_splitter':
            subprocess.Popen(
                ['python', 'main.py', '-p', 'Datasets/', '-f', '{}'.format(ds), '-sn', '{}'.format(st_name),
                 '-m', '{}'.format(model), '-sa', '{}'.format(sampling), '-rs', '-ss'])

if __name__ == "__main__":
    run_study('Runs.csv', '/home/kmurtazalieva/Downloads/scams/Datasets')