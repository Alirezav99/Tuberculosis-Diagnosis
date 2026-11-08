"""Plot histograms and descriptive statistics."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_professional_histogram(
    data,
    title="Histogram",
    xlabel="Value",
    ylabel="Frequency",
    bins=30,
    show_pmf=True,
    show_kde=True,
    show_mean=True,
    show_median=True,
    show_std=True,
    figsize=(8, 5),
    color="skyblue",
    pmf_color="orange",
    kde_color="red",
    grid=True,
    font_size=14,
):
    """Plot a histogram with density curves and summary statistic markers.

    Parameters
    ----------
    data : array-like
        One-dimensional values to plot.
    title : str, default='Histogram'
        Figure title.
    xlabel : str, default='Value'
        Horizontal axis label.
    ylabel : str, default='Frequency'
        Vertical axis label.
    bins : int, default=30
        Number of histogram bins.
    show_pmf : bool, default=True
        Whether to draw the histogram-based PMF/PDF curve.
    show_kde : bool, default=True
        Whether to draw the kernel density estimate.
    show_mean : bool, default=True
        Whether to draw the mean marker.
    show_median : bool, default=True
        Whether to draw the median marker.
    show_std : bool, default=True
        Whether to draw markers one standard deviation from the mean.
    figsize : tuple, default=(8, 5)
        Figure size in inches.
    color : str, default='skyblue'
        Histogram color.
    pmf_color : str, default='orange'
        PMF/PDF curve color.
    kde_color : str, default='red'
        KDE curve color.
    grid : bool, default=True
        Whether to show the plot grid.
    font_size : int, default=14
        Font size for labels and titles.

    Returns
    -------
    None
        Displays the figure with ``matplotlib.pyplot.show``.
    """
    plt.figure(figsize=figsize, dpi=300)

    sns.histplot(
        data,
        bins=bins,
        color=color,
        stat="density",
        edgecolor="black",
        alpha=0.7,
        label="Histogram",
    )

    if show_pmf:
        values, base = np.histogram(data, bins=bins, density=True)
        centers = (base[:-1] + base[1:]) / 2
        plt.plot(centers, values, color=pmf_color, lw=2, label="PMF/PDF")

    if show_kde:
        sns.kdeplot(data, color=kde_color, lw=2, label="KDE")

    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    if show_mean:
        plt.axvline(
            mean, color="blue", linestyle="--", lw=1.5, label=f"Mean: {mean:.2f}"
        )
    if show_median:
        plt.axvline(
            median, color="green", linestyle="--", lw=1.5, label=f"Median: {median:.2f}"
        )
    if show_std:
        plt.axvline(
            mean + std,
            color="purple",
            linestyle=":",
            lw=1,
            label=f"+1 Std: {mean+std:.2f}",
        )
        plt.axvline(
            mean - std,
            color="purple",
            linestyle=":",
            lw=1,
            label=f"-1 Std: {mean-std:.2f}",
        )

    plt.title(title, fontsize=font_size + 2, fontweight="bold")
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.legend(fontsize=font_size - 2)
    if grid:
        plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_histogram_bacill_area(area_of_bacills):
    """Plot a histogram and boxplot for bacillus-area values.

    Parameters
    ----------
    area_of_bacills : array-like
        Values used to calculate the mean, standard deviation, and median.

    Returns
    -------
    None
        Displays the histogram and boxplot with ``matplotlib.pyplot.show``.
    """
    mean_val = area_of_bacills.mean()
    std_val = area_of_bacills.std()
    median_val = np.median(area_of_bacills)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=600)

    axes[0].hist(area_of_bacills, bins=30, color="skyblue", edgecolor="black")
    axes[0].set_xlabel("Area of bacills")
    axes[0].set_ylabel("Count")
    axes[0].axvline(
        mean_val,
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        label=f"Mean: {mean_val:.1f}",
    )
    axes[0].axvline(
        median_val,
        color="green",
        linestyle="dashed",
        linewidth=1.5,
        label=f"Median: {median_val:.1f}",
    )
    axes[0].axvline(
        mean_val + std_val,
        color="orange",
        linestyle="dotted",
        linewidth=1,
        label=f"+1 Std: {mean_val + std_val:.1f}",
    )
    axes[0].axvline(
        mean_val - std_val,
        color="orange",
        linestyle="dotted",
        linewidth=1,
        label=f"-1 Std: {mean_val - std_val:.1f}",
    )
    axes[0].legend()

    axes[1].boxplot(area_of_bacills, vert=False)
    axes[1].set_title("Boxplot: Total White Pixels per Mask")
    axes[1].set_xlabel("Area of bacills")

    plt.tight_layout()
    plt.show()
