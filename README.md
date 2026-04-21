# DDoS Detection in SDN using ACO and Random Forest

This project detects DDoS attacks in Software Defined Networks using:

- Ant Colony Optimization for feature selection
- Random Forest for binary classification
- Accuracy, precision, recall, and F1-score for evaluation
- Matplotlib graphs for feature importance and accuracy comparison

The code is written simply and includes comments so it can be explained in a college viva.

## Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Dataset

Use a CSV dataset such as CICIDS2017 or NSL-KDD.

The target column should contain labels for normal and attack traffic. Common normal labels such as `BENIGN`, `Normal`, and `0` are treated as normal. All other labels are treated as attack.

## Run

```bash
python ddos_aco_rf.py --dataset your_dataset.csv --target Label
```

If your target column is named `Label`, `label`, `class`, `target`, or `attack`, you can omit `--target`:

```bash
python ddos_aco_rf.py --dataset your_dataset.csv
```

## Quick Demo

This repository includes a small sample CSV only to verify that the program runs end to end:

```bash
python ddos_aco_rf.py --dataset sample_sdn_ddos.csv --target Label
```

For project results, use a real dataset such as CICIDS2017 or NSL-KDD.

## Useful Options

```bash
python ddos_aco_rf.py --dataset your_dataset.csv --target Label --ants 12 --iterations 10
```

- `--ants`: number of ants used by ACO
- `--iterations`: number of ACO iterations
- `--min-features`: minimum selected features per ant
- `--max-features`: maximum selected features per ant
- `--no-normalize`: disable StandardScaler normalization

## Output

The program prints:

- Total samples and features
- Number of normal and attack samples
- Best selected features from ACO
- Accuracy, precision, recall, and F1-score

The program saves:

- `outputs/feature_importance.png`
- `outputs/accuracy_comparison.png`

## Viva Explanation

1. The CSV dataset is loaded using pandas.
2. Missing and infinite values are cleaned.
3. Categorical columns are converted to numeric columns using one-hot encoding.
4. Labels are converted into binary values: `0` for Normal and `1` for Attack.
5. Ant Colony Optimization searches for a good subset of features.
6. Each ant selects features using pheromone values and feature importance values.
7. A Random Forest model scores each selected subset using validation F1-score.
8. Good feature subsets increase pheromone on their selected features.
9. The final Random Forest model is trained using the best selected features.
10. The model is evaluated using accuracy, precision, recall, and F1-score.
