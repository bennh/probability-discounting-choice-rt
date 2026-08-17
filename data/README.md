# Local data layout

Place the 49 participant MATLAB files in `data/raw/`. The default loader
expects each file to contain:

- `data_train`: run A trial matrix;
- `data_test`: run B trial matrix;
- `data_labels`: source column labels.

The default matrix positions are defined in `config/analysis.yaml`. Confirm
them against the actual files before running the preparation script.

Neither raw nor processed participant data should be committed. The `.gitkeep`
files preserve the directory structure only.

