import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.colors import ListedColormap


def get_checkerboard_data(modalities, order=["non-brain", "structGS", "func"], split_str=" + "):
    order = np.asarray(order)
    split_modalities = [m.split(split_str) for m in modalities]
    checkerboard = []
    for i, line in enumerate(split_modalities):
        active_mods = np.zeros(len(order)).astype(int)
        for m in line:
            active_mods = active_mods | (m == order)
        checkerboard.append(active_mods * (i + 1))
    checkerboard = np.asarray(checkerboard)
    return checkerboard, order.tolist()


def get_boxplot_with_checkerboard(selected_data, target_order, modalities_ranked, mod_colors, figsize=(11, 2),
                                  violin_x=None, checker_order=["non-brain", "structGS", "func"], x="r2",
                                  xlabel="$R^2$"):
    # set up plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(target_order) + 1, width_ratios=[0.4] + np.ones(len(target_order)).tolist())
    fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    for i in range(2, gs._ncols):
        fig.add_subplot(gs[i], sharex=ax1, sharey=ax1)

    axes = fig.get_axes()

    # 0. checkerboard
    ax = axes[0]
    checkerboard, order = get_checkerboard_data(modalities_ranked, order=checker_order)
    cols = [(1, 1, 1)] + [mod_colors[m] for m in modalities_ranked]
    cmap = ListedColormap(cols)

    sns.heatmap(checkerboard, cbar=False, cmap=cmap, linewidths=.2, ax=ax)
    ax.set_xticklabels(order, rotation=45, ha="left", va="bottom", rotation_mode="default")
    #ax.xaxis.set_ticks_position('none')
    ax.xaxis.tick_top()
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel("modality")

    # 1. plot boxplots
    for ax, target in zip(axes[1:], target_order):
        selected_data_target = selected_data.query("target==@target")

        sns.boxplot(
            x=x,
            y="modality",
            data=selected_data_target,
            order=modalities_ranked,
            palette=mod_colors,
            ax=ax,
        )

        if violin_x:
            sns.violinplot(
                x=violin_x,
                y="modality",
                data=selected_data_target,
                order=modalities_ranked,
                color="grey",
                ax=ax,
            )

        ax.set_xlabel(xlabel)
        label_median(ax)
        ax.set_ylabel("")
        ax.set_title(target)
        ax.set_yticklabels([])
    plt.subplots_adjust(wspace=.05)
    return fig


def label_median(ax, max_median_line=True, font_size=10):
    # https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    lines = ax.get_lines()
    categories = ax.get_yticks()
    max_x = max([lines[1 + cat * 6].get_xdata()[0] for cat in categories])
    max_median = max([lines[4 + cat * 6].get_xdata()[0] for cat in categories])

    for cat in categories:
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        # x = round(lines[4+cat*6].get_xdata()[0],1)
        x = round(lines[4 + cat * 6].get_xdata()[0], 2)
        x_pos = lines[1 + cat * 6].get_xdata()[0]

        ax.text(
            x_pos + max_x * .05,
            cat - categories.max() * .01,
            f'{x}',
            ha='left',
            va='bottom',
            fontweight='bold',
            size=font_size,
            color='black',
        )
    if max_median_line:
        ax.axvline(max_median, color="k", ls="--")
