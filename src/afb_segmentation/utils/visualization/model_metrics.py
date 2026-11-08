"""Plot YOLO training histories and evaluation metrics."""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_hist_models(models_dict: dict) -> None:
    """Plot recorded training-history series for each model.

    Parameters
    ----------
    models_dict : dict
        Mapping from model names to mappings containing a ``train_history``
        dictionary of metric series.

    Returns
    -------
    None
        Adds the plots to a Matplotlib figure.
    """
    fig = plt.figure(figsize=(15, 100), dpi=600)

    colors = plt.cm.Set3(np.linspace(0, 1, len(models_dict)))
    color_map = {name: colors[i] for i, name in enumerate(models_dict.keys())}

    for model_name in list(models_dict.keys()):
        train_hist = models_dict[model_name]["train_history"]
        counter = 1

        for plot_name in train_hist.keys():
            if plot_name in ["epoch", "time", "lr/pg0", "lr/pg1", "lr/pg2"]:
                continue

            epochs = list(range(len(train_hist[plot_name])))
            plt.subplot(21, 2, counter)
            plt.plot(
                epochs,
                list(train_hist[plot_name].values()),
                label=f"{model_name}",
                color=color_map[model_name],
                linewidth=2,
            )

            plt.title(plot_name, fontsize=12, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel(plot_name)
            plt.legend(loc="upper left")
            plt.grid(True, alpha=0.3)

            counter += 1


def plot_evaluation_metrics(modes_dict: dict) -> None:
    """Plot YOLO box and mask mAP metrics for each model.

    Parameters
    ----------
    modes_dict : dict
        Mapping from model names to dictionaries containing a ``results_test``
        mapping with box and mask mAP metrics.

    Returns
    -------
    None
        Displays a four-panel Matplotlib figure.
    """
    model_names = list(modes_dict.keys())
    mAP50b = [
        modes_dict[model_name]["results_test"]["metrics/mAP50(B)"][0]
        for model_name in model_names
    ]
    mAP50m = [
        modes_dict[model_name]["results_test"]["metrics/mAP50(M)"][0]
        for model_name in model_names
    ]
    mAP95b = [
        modes_dict[model_name]["results_test"]["metrics/mAP50-95(B)"][0]
        for model_name in model_names
    ]
    mAP95m = [
        modes_dict[model_name]["results_test"]["metrics/mAP50-95(M)"][0]
        for model_name in model_names
    ]

    fig, ax = plt.subplots(2, 2, figsize=(15, 10), dpi=600)

    ax[0, 0].bar(
        range(len(model_names)),
        mAP50m,
        color="skyblue",
        alpha=0.8,
        edgecolor="navy",
        linewidth=1,
    )
    ax[0, 0].set_title("mAP@50 (Masks)", fontsize=14, fontweight="bold")
    ax[0, 0].set_ylabel("mAP@50 Score", fontsize=12)
    ax[0, 0].set_xticks(range(len(model_names)))
    ax[0, 0].set_xticklabels(
        [name.replace("yolo11", "Y11").replace(" - ", "\n") for name in model_names],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax[0, 0].grid(axis="y", alpha=0.3)
    ax[0, 0].set_ylim(0, max(mAP50m) * 1.1)

    for i, v in enumerate(mAP50m):
        ax[0, 0].text(
            i,
            v + max(mAP50m) * 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax[0, 1].bar(
        range(len(model_names)),
        mAP95m,
        color="lightcoral",
        alpha=0.8,
        edgecolor="darkred",
        linewidth=1,
    )
    ax[0, 1].set_title("mAP@50-95 (Masks)", fontsize=14, fontweight="bold")
    ax[0, 1].set_ylabel("mAP@50-95 Score", fontsize=12)
    ax[0, 1].set_xticks(range(len(model_names)))
    ax[0, 1].set_xticklabels(
        [name.replace("yolo11", "Y11").replace(" - ", "\n") for name in model_names],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax[0, 1].grid(axis="y", alpha=0.3)
    ax[0, 1].set_ylim(0, max(mAP95m) * 1.1)

    for i, v in enumerate(mAP95m):
        ax[0, 1].text(
            i,
            v + max(mAP95m) * 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax[1, 0].bar(
        range(len(model_names)),
        mAP50b,
        color="lightgreen",
        alpha=0.8,
        edgecolor="darkgreen",
        linewidth=1,
    )
    ax[1, 0].set_title("mAP@50 (Bounding Boxes)", fontsize=14, fontweight="bold")
    ax[1, 0].set_ylabel("mAP@50 Score", fontsize=12)
    ax[1, 0].set_xlabel("Models", fontsize=12)
    ax[1, 0].set_xticks(range(len(model_names)))
    ax[1, 0].set_xticklabels(
        [name.replace("yolo11", "Y11").replace(" - ", "\n") for name in model_names],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax[1, 0].grid(axis="y", alpha=0.3)
    ax[1, 0].set_ylim(0, max(mAP50b) * 1.1)

    for i, v in enumerate(mAP50b):
        ax[1, 0].text(
            i,
            v + max(mAP50b) * 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax[1, 1].bar(
        range(len(model_names)),
        mAP95b,
        color="orange",
        alpha=0.8,
        edgecolor="darkorange",
        linewidth=1,
    )
    ax[1, 1].set_title("mAP@50-95 (Bounding Boxes)", fontsize=14, fontweight="bold")
    ax[1, 1].set_ylabel("mAP@50-95 Score", fontsize=12)
    ax[1, 1].set_xlabel("Models", fontsize=12)
    ax[1, 1].set_xticks(range(len(model_names)))
    ax[1, 1].set_xticklabels(
        [name.replace("yolo11", "Y11").replace(" - ", "\n") for name in model_names],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax[1, 1].grid(axis="y", alpha=0.3)
    ax[1, 1].set_ylim(0, max(mAP95b) * 1.1)

    for i, v in enumerate(mAP95b):
        ax[1, 1].text(
            i,
            v + max(mAP95b) * 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.suptitle(
        "Model Performance Comparison - Bounding Box Detection",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
