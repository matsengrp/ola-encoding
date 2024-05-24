"""
Test how OLA distance changes under shuffling leaf labels
"""
from random import shuffle
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

from vector_encoding import (
    get_random_tree,
    ola_distance,
    get_names,
)
from tree_rearrangement import spr_neighbor

def shuffle_labels(tree, permutation):
    """
    returns a new tree whose labels are shuffled according to `permutation`

    Args:
        permutation (list): list of integers from 0 to `n_leaves` - 1, in shuffled order
    """
    new_tree = tree.copy()
    # create leaf shuffle dictionary from permutation
    leaf_names = sorted(tree.get_leaf_names())
    assert len(leaf_names) == len(permutation)
    leaf_shuffle = {}
    for i, name in enumerate(leaf_names):
        leaf_shuffle[name] = leaf_names[permutation[i]]
    # iterate through leaves of `new_tree`
    for leaf in new_tree:
        leaf.name = leaf_shuffle[leaf.name]
    return new_tree

def avg_ola_distance_shuffle(tree1, tree2, n_shuffles=10):
    names = tree1.get_leaf_names()
    n_leaves = len(names)

    dist = ola_distance(tree1, tree2)
    distances = [dist]
    for _ in range(n_shuffles - 1):
        perm = list(range(n_leaves))
        shuffle(perm)
        shuf_tree1 = shuffle_labels(tree1, perm)
        shuf_tree2 = shuffle_labels(tree2, perm)
        dist = ola_distance(shuf_tree1, shuf_tree2)
        distances.append(dist)

    return np.array(distances).mean(), np.array(distances).std()

def plot_dist_vs_shuffle_on_spr_walk(
    n_leaves,
    n_steps,
    n_walks=10,
    out_file="test.pdf",
):
    fig, ax = plt.subplots()
    ax.set_xlabel("OLA distance")
    ax.set_ylabel("shuffled OLA distance")

    for i in range(n_walks):
        tree1 = get_random_tree(n_leaves)
        tree2 = tree1

        # create random permutation for leaf labels
        perm = list(range(n_leaves))
        shuffle(perm)
        shuf_tree1 = shuffle_labels(tree1, perm)

        dist_pairs = []
        for _ in range(n_steps):
            tree2 = spr_neighbor(tree2)
            shuf_tree2 = shuffle_labels(tree2, perm)

            dist = ola_distance(tree1, tree2)
            shuf_dist = ola_distance(shuf_tree1, shuf_tree2)

            # print("original dist =", dist)
            # print("shuffled dist =", shuf_dist)
            dist_pairs.append((dist, shuf_dist))

        # plot pairs
        ax.plot(
            [x for (x, _) in dist_pairs], 
            [y for (_, y) in dist_pairs],
            c="lightgray",
            alpha=0.4,
        )
        ax.scatter(
            [x for (x, _) in dist_pairs], 
            [y for (_, y) in dist_pairs],
            c=np.arange(n_steps),
            cmap="turbo_r",
            alpha=0.6,
        )
        print(f"done plotting walk {i}")

    fig.savefig(out_file)

def plot_with_cmap(ax, xs, ys, **kwargs):
    N = len(xs)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", 
        plt.cm.viridis(np.linspace(0,1,N)),
    )
    ax.plot(xs, ys, **kwargs)
    pass

def main():
    pass

if __name__ == "__main__":
    plot_dist_vs_shuffle_on_spr_walk(n_leaves=500, n_steps=60)

