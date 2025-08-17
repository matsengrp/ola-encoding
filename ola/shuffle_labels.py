"""
Test how OLA distance changes under shuffling leaf labels
"""
from math import isqrt
from random import shuffle
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns

from ola_encoding import (
    to_vector,
    to_tree,
    ola_distance,
)
from utils import (
    get_random_tree,
)
from tree_rearrangement import spr_neighbor

def shuffle_labels(tree, permutation):
    """
    returns a new tree whose labels are shuffled according to `permutation`

    Args:
        tree: ete3 Tree object
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
    out_file="temp.pdf",
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

def plot_shuffled_dist_on_spr_walk(n_leaves, n_steps, out_file="temp.pdf"):
    """
    Choose a random starting tree with `n_leaves` leaves, run an SPR walk for `n_steps`
    steps, and plot the OLA distance from the starting tree using the original labeling,
    as well as using a shuffled labeling
    """
    # create starting tree
    tree_0 = get_random_tree(n_leaves)
    # create random leaf label shuffle
    perm = list(range(n_leaves))
    shuffle(perm)
    shuf_tree_0 = shuffle_labels(tree_0, perm)
    # initialize distance lists
    dists = [0]
    shuf_dists = [0]
    # create SPR walk
    tree = tree_0
    for _ in range(n_steps):
        tree = spr_neighbor(tree)
        dist = ola_distance(tree_0, tree)
        dists.append(dist)

        shuf_tree = shuffle_labels(tree, perm)
        shuf_dist = ola_distance(shuf_tree_0, shuf_tree)
        shuf_dists.append(shuf_dist)
    # plot distances
    fig, ax = plt.subplots()
    ax.plot(dists, alpha=0.5, marker="o")
    ax.plot(shuf_dists, alpha=0.5, marker="o")

    fig.savefig(out_file)

def plot_avg_ola_dist_on_spr_walk(n_leaves, n_steps, out_file="temp.pdf"):
    """
    Function does the following:
        1. Choose a random starting tree with `n_leaves` leaves, 
        2. run an SPR walk for `n_steps` steps, and 
        3. plot the average OLA distance from the starting tree, averaging over a random
            choice of leaf label shuffles
    """
    # create starting tree
    tree_0 = get_random_tree(n_leaves)
    # create and store shuffles
    perms = [list(range(n_leaves)) for _ in range(10)]
    for perm in perms:
        shuffle(perm)
    # initialize distance lists
    dists = [[0] for _ in range(10)]
    avg_dists = [0.]
    std_devs = [0.]
    # shuf_dists = [0]
    # create SPR walk
    tree = tree_0
    for _ in range(n_steps):
        tree = spr_neighbor(tree)
        # dist, std = avg_ola_distance_shuffle(tree_0, tree)
        for i, perm in enumerate(perms):
            shuf_tree_0 = shuffle_labels(tree_0, perm)
            shuf_tree = shuffle_labels(tree, perm)
            distance = ola_distance(shuf_tree_0, shuf_tree)
            dists[i].append(distance)
        shuf_dists = np.array([dist[-1] for dist in dists])
        avg_dists.append(shuf_dists.mean())
        std_devs.append(shuf_dists.std())

    # plot distances
    fig, ax = plt.subplots()
    for i in range(10):
        ax.plot(dists[i], alpha=0.5, color="C0")
    ax.plot(avg_dists, alpha=0.9, marker="o", color="C0")
    # plot standard deviations
    ax.plot(std_devs, alpha=0.5, marker="o", color="C1")
    ax.fill_between(range(n_steps + 1), std_devs, alpha=0.5, color="C1")

    ax.set_xlabel(f"SPR steps")
    ax.set_ylabel(f"OLA distance")

    fig.savefig(out_file)

def near_mid_far_test(seed=None, n_leaves=200, n_perms=10, output="temp.pdf"):
    """
    1. Choose a focal tree T_0
    2. Choose trees T_1, T_2, T_3, which have increasing distances from T_0
    3. Plot OLA-distances from T_0 to T_i, for random shufflings of leaf labels
    """
    # set random seed
    if seed is not None:
        random.seed(seed)

    tree_0 = get_random_tree(n_leaves)
    # [near, mid, far] = [5, isqrt(n_leaves), n_leaves // 2]
    spr_dists = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    spr_dist_trees = {}
    for dist in spr_dists:
        tree = to_tree(to_vector(tree_0))
        for _ in range(dist):
            tree = spr_neighbor(tree)
        spr_dist_trees[dist] = tree
    # # apply 5 SPR moves for "near" tree
    # tree_1 = to_tree(to_vector(tree_0))
    # for _ in range(near):
    #     tree_1 = spr_neighbor(tree_1)
    # # apply SPR moves for "mid" tree
    # tree_2 = to_tree(to_vector(tree_0))
    # for _ in range(mid):
    #     tree_2 = spr_neighbor(tree_2)
    # # apply SPR moves for "far" tree
    # tree_3 = to_tree(to_vector(tree_0))
    # for _ in range(far):
    #     tree_3 = spr_neighbor(tree_3)
    
    # plot OLA distances for random shuffles
    # create and store shuffles
    perms = [list(range(n_leaves)) for _ in range(n_perms)]
    for perm in perms:
        shuffle(perm)
    data = []
    for spr_dist in spr_dists:
        tree = spr_dist_trees[spr_dist]
        for perm in perms:
            stree_0 = shuffle_labels(tree_0, perm)
            stree = shuffle_labels(tree, perm)
            d = ola_distance(stree_0, stree)
            data.append([spr_dist, d])
    fig, ax = plt.subplots()

    df = pd.DataFrame(data, columns=("spr_moves", "ola_dist"))
    ax.set_xscale("log")
    bp = sns.boxplot(data=df, x="spr_moves", y="ola_dist", native_scale=True)

    bp.set_xticks(spr_dists)
    bp.set_xticklabels(spr_dists)

    ax.minorticks_off()
    ax.set_xlabel("SPR steps")
    ax.set_ylabel("OLA distance")
    sns.despine(fig, trim=True)
    fig.savefig(output)

def main():
    pass

if __name__ == "__main__":
    # plot_dist_vs_shuffle_on_spr_walk(n_leaves=500, n_steps=60)
    # plot_shuffled_dist_on_spr_walk(n_leaves=300, n_steps=30)
    # plot_avg_ola_dist_on_spr_walk(n_leaves=500, n_steps=100)

    near_mid_far_test(seed=168, n_leaves=200)

