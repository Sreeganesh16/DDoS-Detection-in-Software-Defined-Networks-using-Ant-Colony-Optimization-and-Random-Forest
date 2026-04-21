"""
Visual ACO Simulation Demo for DDoS Feature Selection.

This file is for presentation/viva purposes. It shows how ants select
features, how pheromone values change, and how the best feature subset is
found before Random Forest classification.

Run:
    python aco_simulation_demo.py --dataset sample_sdn_ddos.csv --target Label

Save as GIF:
    python aco_simulation_demo.py --dataset sample_sdn_ddos.csv --target Label --save-gif outputs/aco_simulation.gif
"""

import argparse
import os
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_DIR, ".matplotlib-cache"))

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from ddos_aco_rf import (
    RANDOM_STATE,
    calculate_feature_heuristic,
    evaluate_subset,
    load_and_preprocess_data,
)


def shorten_name(name: str, max_length: int = 16) -> str:
    """Shorten long feature names so they fit in the simulation plot."""
    return name if len(name) <= max_length else name[: max_length - 3] + "..."


def build_simulation_frames(
    x_train,
    y_train,
    feature_names: List[str],
    n_ants: int,
    n_iterations: int,
    subset_size: int,
) -> List[Dict]:
    """
    Run a small ACO process and store each ant step for animation.

    Each frame contains:
        - current pheromone values
        - selected feature indices
        - current ant score
        - best score so far
    """
    x_subtrain, x_valid, y_subtrain, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    n_features = len(feature_names)
    heuristic = calculate_feature_heuristic(x_subtrain, y_subtrain)
    pheromone = np.ones(n_features)
    rng = np.random.default_rng(RANDOM_STATE)

    best_score = 0.0
    best_subset = []
    frames = []

    for iteration in range(1, n_iterations + 1):
        for ant in range(1, n_ants + 1):
            desirability = pheromone * (heuristic ** 2)
            probabilities = desirability / desirability.sum()

            selected = sorted(
                rng.choice(
                    n_features,
                    size=min(subset_size, n_features),
                    replace=False,
                    p=probabilities,
                ).tolist()
            )

            score = evaluate_subset(
                x_subtrain,
                x_valid,
                y_subtrain,
                y_valid,
                selected,
            )

            if score > best_score:
                best_score = score
                best_subset = selected

            # Simple pheromone update for visual clarity:
            # small evaporation every ant step, then reward selected features.
            pheromone = 0.90 * pheromone
            for index in selected:
                pheromone[index] += score

            frames.append(
                {
                    "iteration": iteration,
                    "ant": ant,
                    "selected": selected,
                    "score": score,
                    "best_score": best_score,
                    "best_subset": best_subset.copy(),
                    "pheromone": pheromone.copy(),
                }
            )

    return frames


def animate_aco(
    frames: List[Dict],
    feature_names: List[str],
    output_gif: str = None,
    interval: int = 1000,
) -> None:
    """Create and show/save the ACO simulation animation."""
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    x_positions = np.cos(angles)
    y_positions = np.sin(angles)
    short_names = [shorten_name(name) for name in feature_names]

    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 0.42])
    network_axis = fig.add_subplot(grid[:, 0])
    bar_axis = fig.add_subplot(grid[0, 1])
    text_axis = fig.add_subplot(grid[1, 1])

    def draw(frame_number: int):
        frame = frames[frame_number]
        selected = set(frame["selected"])
        best_subset = set(frame["best_subset"])
        pheromone = frame["pheromone"]
        normalized_pheromone = pheromone / pheromone.max()

        network_axis.clear()
        bar_axis.clear()
        text_axis.clear()

        network_axis.set_title("ACO Feature Selection Simulation")
        network_axis.set_xlim(-1.45, 1.45)
        network_axis.set_ylim(-1.45, 1.45)
        network_axis.set_aspect("equal")
        network_axis.axis("off")

        for i in range(n_features):
            color = "#2f9f6f" if i in selected else "#6c8ebf"
            edge_color = "#d98600" if i in best_subset else "#243447"
            size = 300 + 1700 * normalized_pheromone[i]

            network_axis.scatter(
                x_positions[i],
                y_positions[i],
                s=size,
                c=color,
                edgecolors=edge_color,
                linewidths=2,
                alpha=0.88,
            )
            network_axis.text(
                1.2 * x_positions[i],
                1.2 * y_positions[i],
                short_names[i],
                ha="center",
                va="center",
                fontsize=8,
            )

        network_axis.scatter(0, 0, s=520, c="#222222")
        network_axis.text(0, 0, f"Ant {frame['ant']}", color="white", ha="center", va="center")

        for index in selected:
            network_axis.plot(
                [0, x_positions[index]],
                [0, y_positions[index]],
                color="#2f9f6f",
                linewidth=2.2,
                alpha=0.8,
            )

        sorted_indices = np.argsort(pheromone)
        bar_colors = ["#2f9f6f" if i in selected else "#7a7a7a" for i in sorted_indices]

        bar_axis.barh(
            [short_names[i] for i in sorted_indices],
            pheromone[sorted_indices],
            color=bar_colors,
        )
        bar_axis.set_title("Pheromone Strength")
        bar_axis.set_xlabel("Pheromone")

        selected_names = ", ".join(short_names[i] for i in frame["selected"])
        best_names = ", ".join(short_names[i] for i in frame["best_subset"])

        text_axis.axis("off")
        text_axis.text(
            0,
            0.95,
            f"Iteration: {frame['iteration']}    Ant: {frame['ant']}",
            fontsize=11,
            weight="bold",
        )
        text_axis.text(0, 0.70, f"Current subset F1-score: {frame['score']:.4f}", fontsize=10)
        text_axis.text(0, 0.52, f"Best F1-score so far: {frame['best_score']:.4f}", fontsize=10)
        text_axis.text(0, 0.31, f"Current selected features:\n{selected_names}", fontsize=9)
        text_axis.text(0, 0.05, f"Best features so far:\n{best_names}", fontsize=9)

        fig.tight_layout()

    demo_animation = animation.FuncAnimation(
        fig,
        draw,
        frames=len(frames),
        interval=interval,
        repeat=True,
    )

    if output_gif:
        os.makedirs(os.path.dirname(output_gif) or ".", exist_ok=True)
        writer = animation.PillowWriter(fps=max(1, int(1000 / interval)))
        demo_animation.save(output_gif, writer=writer)
        print(f"ACO simulation GIF saved to: {output_gif}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual ACO simulation demo")
    parser.add_argument("--dataset", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--ants", type=int, default=5, help="Number of ants per iteration")
    parser.add_argument("--iterations", type=int, default=5, help="Number of ACO iterations")
    parser.add_argument(
        "--display-features",
        type=int,
        default=10,
        help="Number of top features to show in the visual demo",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=3,
        help="Number of features each ant selects",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Delay between animation frames in milliseconds",
    )
    parser.add_argument(
        "--save-gif",
        default=None,
        help="Optional path to save the simulation as a GIF",
    )

    args = parser.parse_args()

    print("Loading dataset for ACO simulation...")
    x, y, feature_names = load_and_preprocess_data(args.dataset, args.target)

    x_train, _, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    heuristic = calculate_feature_heuristic(x_train, y_train)
    top_indices = np.argsort(heuristic)[::-1][: min(args.display_features, len(feature_names))]

    # The visual demo uses only the top features so the animation stays readable.
    x_demo = x_train.iloc[:, top_indices]
    demo_feature_names = [feature_names[i] for i in top_indices]

    print(f"Showing {len(demo_feature_names)} features in the visual simulation.")
    print("Close the animation window when your demo is finished.")

    frames = build_simulation_frames(
        x_train=x_demo,
        y_train=y_train,
        feature_names=demo_feature_names,
        n_ants=args.ants,
        n_iterations=args.iterations,
        subset_size=args.subset_size,
    )

    animate_aco(
        frames=frames,
        feature_names=demo_feature_names,
        output_gif=args.save_gif,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
