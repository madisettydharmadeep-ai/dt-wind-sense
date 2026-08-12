"""
plot_graph.py  --  visualise a B-matrix causal graph

Usage:
    python plot_graph.py path/to/B_Matrix_6.xlsx
    python plot_graph.py path/to/B_Matrix_6.xlsx --threshold 0.05 --title "WT80 window 0"

The Excel file must be a square N x N matrix where columns are node (sensor) names
and each cell [i,j] is the causal weight from row-node i to column-node j.
This is the format saved by causal_29_12.py (both raw and scaled B-matrix files).
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_b_matrix(path, threshold=0.1, title=None, save=False):
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    df = pd.read_excel(path)
    node_names = list(df.columns)
    matrix = df.values  # shape (N, N)
    N = len(node_names)

    G = nx.DiGraph()
    for name in node_names:
        G.add_node(name)

    edge_weights = {}
    for i in range(N):
        for j in range(N):
            w = matrix[i, j]
            if i != j and abs(w) >= threshold:
                G.add_edge(node_names[i], node_names[j], weight=w)
                edge_weights[(node_names[i], node_names[j])] = w

    # circular layout so short names don't collide
    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(10, 8))

    # nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightsteelblue",
                           node_size=1800, alpha=0.9)
    # short labels (last part of sensor name to keep readable)
    short = {n: n.replace("wrot_", "").replace("_ValBl", "\nBl") for n in node_names}
    nx.draw_networkx_labels(G, pos, labels=short, ax=ax,
                            font_size=8, font_color="black", font_weight="bold")

    if G.number_of_edges() > 0:
        weights = np.array([abs(d["weight"]) for _, _, d in G.edges(data=True)])
        # scale line widths 0.5-4 relative to max weight
        w_max = weights.max() if weights.max() > 0 else 1.0
        widths = 0.5 + 3.5 * (weights / w_max)

        # colour: positive=blue, negative=red
        colors = ["royalblue" if d["weight"] >= 0 else "tomato"
                  for _, _, d in G.edges(data=True)]

        nx.draw_networkx_edges(G, pos, ax=ax, width=list(widths),
                               edge_color=colors, arrows=True,
                               arrowsize=20, connectionstyle="arc3,rad=0.15",
                               min_source_margin=30, min_target_margin=30)

        edge_labels = {e: f"{w:+.2f}" for e, w in edge_weights.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=7, rotate=False)
    else:
        ax.text(0.5, 0.5, f"No edges above threshold {threshold}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="grey")

    file_label = os.path.basename(path)
    ax.set_title(title if title else file_label, fontsize=12, pad=12)
    ax.axis("off")

    # legend
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color="royalblue", lw=2, label="positive causal weight"),
              Line2D([0], [0], color="tomato",    lw=2, label="negative causal weight")]
    ax.legend(handles=legend, loc="upper left", fontsize=8)

    plt.tight_layout()

    if save:
        out = os.path.splitext(path)[0] + "_plot.png"
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a causal B-matrix graph.")
    parser.add_argument("path", help="Path to the B-matrix Excel file (.xlsx)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Min |weight| to draw an edge (default: 0.1)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")
    parser.add_argument("--save", action="store_true",
                        help="Save PNG next to the input file instead of showing interactively")
    args = parser.parse_args()

    plot_b_matrix(args.path, threshold=args.threshold, title=args.title, save=args.save)
