"""
DDoS Detection in Software Defined Networks using
Ant Colony Optimization (ACO) and Random Forest.

This script is intentionally simple and well-commented so that it can be
explained in a college viva.

Required libraries:
    pandas, numpy, scikit-learn, matplotlib

Example:
    python ddos_aco_rf.py --dataset CICIDS2017.csv --target Label

If --target is not given, the script tries to find a common target column
name such as Label, label, class, target, or attack.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Keep Matplotlib cache inside the project folder. This avoids permission
# problems on locked-down college lab machines or sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_DIR, ".matplotlib-cache"))

import matplotlib.pyplot as plt


def find_target_column(data: pd.DataFrame, user_target: str = None) -> str:
    """Find the target column in the dataset."""
    if user_target:
        if user_target not in data.columns:
            raise ValueError(f"Target column '{user_target}' was not found in the CSV file.")
        return user_target

    possible_names = [
        "Label",
        "label",
        "Class",
        "class",
        "Target",
        "target",
        "Attack",
        "attack",
        "y",
    ]

    for name in possible_names:
        if name in data.columns:
            return name

    raise ValueError(
        "Could not automatically find the target column. "
        "Please pass it using --target column_name."
    )


def convert_labels_to_binary(labels: pd.Series) -> pd.Series:
    """
    Convert labels to binary values:
        0 = Normal
        1 = Attack

    Common normal labels such as BENIGN, Normal, normal, 0, and false are
    treated as Normal. All other labels are treated as Attack.
    """
    normal_values = {"normal", "benign", "0", "false", "non-attack", "non_attack"}

    def map_label(value):
        value_text = str(value).strip().lower().strip(" .")
        return 0 if value_text in normal_values else 1

    return labels.apply(map_label)


def load_and_preprocess_data(
    csv_path: str,
    target_column: str = None,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load the CSV dataset and perform basic preprocessing.

    Steps:
        1. Replace infinite values with missing values.
        2. Drop completely empty rows and columns.
        3. Separate features and target.
        4. Convert target labels to Attack/Normal binary labels.
        5. Fill missing numeric values with median.
        6. Fill missing categorical values with mode.
        7. Convert categorical features to numeric using one-hot encoding.
        8. Normalize feature values using StandardScaler.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file was not found: {csv_path}")

    data = pd.read_csv(csv_path, low_memory=False)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0, how="all")
    data = data.dropna(axis=1, how="all")

    target_column = find_target_column(data, target_column)

    y = convert_labels_to_binary(data[target_column])
    x = data.drop(columns=[target_column])

    # Remove duplicate column names if the dataset contains them.
    x = x.loc[:, ~x.columns.duplicated()]

    numeric_columns = x.select_dtypes(include=[np.number]).columns
    categorical_columns = x.select_dtypes(exclude=[np.number]).columns

    for column in numeric_columns:
        median_value = x[column].median()
        x[column] = x[column].fillna(median_value)

    for column in categorical_columns:
        mode_values = x[column].mode()
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        x[column] = x[column].fillna(fill_value)

    # Convert text/categorical features into numeric columns.
    x = pd.get_dummies(x, columns=list(categorical_columns), drop_first=True)

    # Some CSVs store numeric-looking values as text. Convert whatever can be
    # converted, and fill any remaining missing values.
    for column in x.columns:
        x[column] = pd.to_numeric(x[column], errors="coerce")
        x[column] = x[column].fillna(x[column].median())

    feature_names = list(x.columns)

    if normalize:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x = pd.DataFrame(x_scaled, columns=feature_names)

    return x, y, feature_names


def evaluate_subset(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    selected_indices: List[int],
) -> float:
    """
    Train a small Random Forest on the selected feature subset and return F1.

    F1-score is useful for attack detection because it balances precision and
    recall. Accuracy alone can be misleading if the dataset is imbalanced.
    """
    if len(selected_indices) == 0:
        return 0.0

    model = RandomForestClassifier(
        n_estimators=60,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced",
    )

    model.fit(x_train.iloc[:, selected_indices], y_train)
    predictions = model.predict(x_valid.iloc[:, selected_indices])

    return f1_score(y_valid, predictions, zero_division=0)


def calculate_feature_heuristic(x_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """
    Create a heuristic value for each feature using Random Forest importance.

    In ACO, ants prefer features with:
        - high pheromone value from previous good solutions
        - high heuristic value from initial feature importance
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    importance = model.feature_importances_
    importance = importance + 1e-6
    return importance / importance.sum()


def ant_colony_feature_selection(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    n_ants: int = 12,
    n_iterations: int = 10,
    evaporation_rate: float = 0.25,
    alpha: float = 1.0,
    beta: float = 2.0,
    min_features: int = 5,
    max_features: int = None,
) -> Tuple[List[int], float]:
    """
    Select important features using a simple Ant Colony Optimization algorithm.

    ACO idea:
        1. Each ant builds one possible feature subset.
        2. The subset is scored using a Random Forest validation F1-score.
        3. Good subsets add pheromone to their selected features.
        4. Pheromone evaporates, so poor choices slowly lose influence.
        5. After several iterations, the best subset is returned.
    """
    n_features = len(feature_names)

    if n_features == 0:
        raise ValueError("No features are available for feature selection.")

    if max_features is None:
        max_features = max(min(20, n_features), min_features)

    min_features = max(1, min(min_features, n_features))
    max_features = max(min_features, min(max_features, n_features))

    x_subtrain, x_valid, y_subtrain, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    heuristic = calculate_feature_heuristic(x_subtrain, y_subtrain)
    pheromone = np.ones(n_features)

    best_subset = []
    best_score = 0.0

    rng = np.random.default_rng(RANDOM_STATE)

    for iteration in range(n_iterations):
        ant_solutions = []

        for _ in range(n_ants):
            desirability = (pheromone ** alpha) * (heuristic ** beta)
            selection_probability = desirability / desirability.sum()

            subset_size = int(rng.integers(min_features, max_features + 1))

            selected = rng.choice(
                n_features,
                size=subset_size,
                replace=False,
                p=selection_probability,
            )
            selected = sorted(selected.tolist())

            score = evaluate_subset(x_subtrain, x_valid, y_subtrain, y_valid, selected)
            ant_solutions.append((selected, score))

            if score > best_score:
                best_subset = selected
                best_score = score

        # Evaporation: old pheromone becomes weaker.
        pheromone = (1 - evaporation_rate) * pheromone

        # Deposit pheromone: better solutions add more pheromone.
        for selected, score in ant_solutions:
            for feature_index in selected:
                pheromone[feature_index] += score

        print(
            f"ACO iteration {iteration + 1}/{n_iterations} - "
            f"best validation F1: {best_score:.4f}, "
            f"selected features: {len(best_subset)}"
        )

    return best_subset, best_score


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train the final Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """Calculate Accuracy, Precision, Recall, and F1-score."""
    predictions = model.predict(x_test)

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
    }

    return results


def print_results(results: List[dict]) -> None:
    """Print evaluation results in a readable table."""
    print("\nModel Evaluation Results")
    print("-" * 78)
    print(f"{'Model':35s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-score':>10s}")
    print("-" * 78)

    for result in results:
        print(
            f"{result['Model']:35s} "
            f"{result['Accuracy']:10.4f} "
            f"{result['Precision']:10.4f} "
            f"{result['Recall']:10.4f} "
            f"{result['F1-score']:10.4f}"
        )


def plot_feature_importance(
    model: RandomForestClassifier,
    selected_feature_names: List[str],
    output_path: str,
    top_n: int = 20,
) -> None:
    """Plot the most important selected features."""
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1][:top_n]

    names = [selected_feature_names[i] for i in sorted_indices]
    values = importances[sorted_indices]

    plt.figure(figsize=(11, 6))
    plt.barh(names[::-1], values[::-1], color="#2f6f9f")
    plt.xlabel("Random Forest Importance")
    plt.ylabel("Selected Feature")
    plt.title("Top Feature Importances After ACO Feature Selection")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_comparison(
    baseline_accuracy: float,
    selected_accuracy: float,
    output_path: str,
) -> None:
    """Plot accuracy with and without ACO feature selection."""
    labels = ["Without ACO", "With ACO"]
    scores = [baseline_accuracy, selected_accuracy]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, scores, color=["#7a7a7a", "#2f9f6f"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DDoS Detection using Ant Colony Optimization and Random Forest"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the CSV dataset, for example CICIDS2017.csv or NSL-KDD.csv",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Name of the target/label column. If omitted, common names are detected.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable StandardScaler normalization.",
    )
    parser.add_argument("--ants", type=int, default=12, help="Number of ants used in ACO.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of ACO iterations.",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=5,
        help="Minimum number of features an ant can select.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features an ant can select.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where plots will be saved.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and preprocessing dataset...")
    x, y, feature_names = load_and_preprocess_data(
        csv_path=args.dataset,
        target_column=args.target,
        normalize=not args.no_normalize,
    )

    print(f"Total samples: {len(x)}")
    print(f"Total features after preprocessing: {len(feature_names)}")
    print(f"Normal samples: {(y == 0).sum()}")
    print(f"Attack samples: {(y == 1).sum()}")

    if y.nunique() < 2:
        raise ValueError(
            "The target column contains only one class after binary conversion. "
            "Please check the label column."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining baseline Random Forest with all features...")
    baseline_model = train_random_forest(x_train, y_train)
    baseline_results = evaluate_model(
        baseline_model,
        x_test,
        y_test,
        "Random Forest without ACO",
    )

    print("\nRunning Ant Colony Optimization for feature selection...")
    selected_indices, best_validation_score = ant_colony_feature_selection(
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        n_ants=args.ants,
        n_iterations=args.iterations,
        min_features=args.min_features,
        max_features=args.max_features,
    )

    selected_feature_names = [feature_names[i] for i in selected_indices]

    print("\nBest ACO validation F1-score:", f"{best_validation_score:.4f}")
    print("\nSelected Features")
    print("-" * 40)
    for number, feature in enumerate(selected_feature_names, start=1):
        print(f"{number:2d}. {feature}")

    print("\nTraining final Random Forest using selected features...")
    x_train_selected = x_train.iloc[:, selected_indices]
    x_test_selected = x_test.iloc[:, selected_indices]

    selected_model = train_random_forest(x_train_selected, y_train)
    selected_results = evaluate_model(
        selected_model,
        x_test_selected,
        y_test,
        "Random Forest with ACO",
    )

    print_results([baseline_results, selected_results])

    feature_plot_path = os.path.join(args.output_dir, "feature_importance.png")
    accuracy_plot_path = os.path.join(args.output_dir, "accuracy_comparison.png")

    plot_feature_importance(
        model=selected_model,
        selected_feature_names=selected_feature_names,
        output_path=feature_plot_path,
    )
    plot_accuracy_comparison(
        baseline_accuracy=baseline_results["Accuracy"],
        selected_accuracy=selected_results["Accuracy"],
        output_path=accuracy_plot_path,
    )

    print("\nPlots saved:")
    print(f"1. {feature_plot_path}")
    print(f"2. {accuracy_plot_path}")


if __name__ == "__main__":
    main()
