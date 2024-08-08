#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# plt.style.use(["science", "ieee"])

linewidth = 0.7
markersize = 8
markeredgewidth = 0.5
title_fontsize = 17
label_fontsize = 17
tick_fontsize = 17
legend_fontsize = 15


def plot_fpr_fnr(probs, labels, threshold_range, step):
    fpr_list = []
    fnr_list = []
    threshold_min, threshold_max = threshold_range
    thresholds = np.arange(threshold_min, threshold_max + step, step)
    for t in thresholds:
        predictions = np.where(probs >= t, 1, 0)
        fp = np.sum((predictions == 1) & (labels == 0))
        tp = np.sum((predictions == 1) & (labels == 1))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))

        fpr_list.append(fp / (fp + tn))
        fnr_list.append(fn / (fn + tp))

    fig1, ax1 = plt.subplots()
    ax1.plot(thresholds, fpr_list, linewidth=linewidth)
    ax1.set_xlabel("Threshold", fontsize=label_fontsize)
    ax1.set_ylabel("FPR", fontsize=label_fontsize)
    ax1.xaxis.set_major_locator(mticker.FixedLocator(np.arange(threshold_min, threshold_max + 0.1, 0.1)))
    ax1.set_ylim([0.0, 1.0])
    ax1.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    ax1.minorticks_off()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.plot(thresholds, fnr_list, linewidth=linewidth)
    ax2.set_xlabel("Threshold", fontsize=label_fontsize)
    ax2.set_ylabel("FNR", fontsize=label_fontsize)
    ax2.xaxis.set_major_locator(mticker.FixedLocator(np.arange(threshold_min, threshold_max + 0.1, 0.1)))
    ax2.set_ylim([0.0, 1.0])
    ax2.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    ax2.minorticks_off()
    fig2.tight_layout()
    return fig1, fig2


def plot_bar(bar_data, error_data, xlabel=None, ylabel=None, yticks=None, xticklabels=None,
             yticklabels=None, ylim=None,):
    num_bars = len(bar_data)

    fig, ax = plt.subplots()
    ax.bar(
        x=np.arange(num_bars), height=bar_data, yerr=error_data, linewidth=0.3,
        facecolor="lightgreen", edgecolor="k", ecolor="k", capsize=1.5,
        error_kw={"elinewidth": 0.3, "capthick": 0.3}
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(np.arange(num_bars), xticklabels, rotation=0)
    if yticks is not None:
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="x", which="major")
    ax.tick_params(axis="y", which="major")
    ax.minorticks_off()
    fig.tight_layout()
    return fig


def plot_record(d, keys, title=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None,
                xlim=None, ylim=None, use_marker=False, show_legend=True):

    assert "epoch" in d.keys()
    line_styles = ["-", "--", "-.", ":"]
    marker_styles = ["o", "d", "s", "P", "*", "x"]
    colors = ["k", "r", "b", "g", "y", "c", "m"]
    line_style_counter = 0
    marker_style_counter = 0
    color_counter = 0

    fig, ax = plt.subplots()
    for i in range(len(keys)):
        ax.plot(
            d["epoch"], d[keys[i]], label=keys[i].replace("_", " "), color=colors[color_counter],
            linestyle=line_styles[line_style_counter], linewidth=linewidth,
            marker=marker_styles[marker_style_counter] if use_marker else "", markersize=markersize, markerfacecolor="none",
            markeredgecolor=colors[color_counter], markeredgewidth=markeredgewidth
        )
        line_style_counter = 0 if line_style_counter + 1 == len(line_styles) else line_style_counter + 1
        marker_style_counter = 0 if marker_style_counter + 1 == len(marker_styles) else marker_style_counter + 1
        color_counter = 0 if color_counter + 1 == len(colors) else color_counter + 1

    ax.set_xlabel("\#Epoch", fontsize=label_fontsize)
    ax.set_ylabel("_".join(keys[0].replace("_", " ").split(" ")[1:]), fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if xticks is not None:
        ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    ax.minorticks_off()
    if show_legend:
        plt.legend(loc="best", fontsize=legend_fontsize, fancybox=False, frameon=True, framealpha=1.0, edgecolor="black")
    fig.tight_layout()
    return fig


def plot_heatmap(data, img_path, xticks=None, xticklabels=None, yticks=None, yticklabels=None, ticksize=6,
                 xlabel=None, ylabel=None, labelsize=12):

    grid_kws = {"width_ratios": [1, 0.08], "wspace": 0.08}
    data = pd.DataFrame(data=data)
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)
    ax = sns.heatmap(data, ax=ax, cbar_ax=cbar_ax, cmap="YlGnBu", annot=False, fmt="d", cbar=True)

    # make frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    for _, spine in cbar_ax.spines.items():
        spine.set_visible(True)

    # set ticks
    ax.tick_params(
        axis="x", which="both", direction="out", top=False, right=False,
        bottom=False, left=False, labelrotation=0, labelsize=ticksize
    )
    ax.tick_params(
        axis="y", which="both", direction="out", top=False, right=False,
        bottom=False, left=False, labelrotation=0, labelsize=ticksize
    )
    if xticks is not None and xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None and yticklabels is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    cbar_ax.tick_params(axis="both", which="both", direction="out", labelsize=ticksize)
    cbar_ax.minorticks_off()

    # set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    fig.tight_layout()
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close("all")


def plot_hist(data, img_path, bins, density=False, ticksize=8, xlabel=None, ylabel=None, labelsize=12):

    fig, ax = plt.subplots()
    ax.hist(data, bins=bins, density=density)

    # set ticks
    ax.tick_params(
        axis="x", which="both", direction="out", top=False, right=False,
        bottom=True, left=False, labelrotation=0, labelsize=ticksize
    )
    ax.tick_params(
        axis="y", which="both", direction="out", top=False, right=False,
        bottom=False, left=True, labelrotation=0, labelsize=ticksize
    )
    ax.minorticks_off()

    # set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)

    fig.tight_layout()
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close("all")




def plot_images(arrays, img_path):

    n, h, w, c = arrays.shape
    num_rows = 1 if n % 5 != 0 else int(n / 5)
    num_cols = n if num_rows == 1 else 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
    if n == 1:
        if c == 1:
            axes.imshow(arrays.squeeze(), cmap="gray")
        else:
            axes.imshow(arrays.squeeze())
        axes.axis("off")
        fig.tight_layout()
        plt.savefig(img_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close("all")
    else:
        counter = 0
        for i in range(num_rows):
            for j in range(num_cols):
                array = arrays[counter]
                if c == 1:
                    axes[i, j].imshow(array.squeeze(), cmap="gray")
                else:
                    axes[i, j].imshow(array)
                axes[i, j].axis("off")
                counter += 1
        fig.tight_layout()
        plt.savefig(img_path, bbox_inches="tight", dpi=300)
        plt.close("all")


def plot_image_grid(arrays_list, img_path):

    n_rows = len(arrays_list)
    n_cols, h, w, c = arrays_list[0].shape
    fig, axes = plt.subplots(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            if c == 1:
                axes[i, j].imshow(arrays_list[i][j].squeeze(), cmap="gray")
            else:
                axes[i, j].imshow(arrays_list[i][j])
            axes[i, j].axis("off")
    fig.tight_layout()
    plt.savefig(img_path, bbox_inches="tight", dpi=300)
    plt.close("all")


