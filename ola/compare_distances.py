
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from pathlib import Path

from subprocess import run
from ete3 import Tree
from ola_encoding import (
    to_tree, 
    to_vector,
    hamming_dist,
    ola_distance,
    ola_neighbor,
)
from utils import (
    get_random_tree,
    get_random_yule_tree,
    get_all_vectors, 
)
from tree_rearrangement import (
    nni_neighbor,
    spr_neighbor,
    rspr_distance,
)

def plot_avg_spr_distance_vs_tree_size(
    distribution="uniform",
    seed=None,
    output="temp.pdf"
):
    """
    create a scatterplot of expected distance from a fixed tree to a random SPR-neighbor 
    tree, using randomly seleted SPR neighbors
    args:
        distrubtion: "uniform" or "yule"
    """
    # set random seed
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots()

    xs = []
    ys = []
    for n in [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        for _ in range(10):
            if distribution == "uniform":
                tree = get_random_tree(n_leaves=n)
            elif distribution == "yule":
                tree = get_random_yule_tree(n_leaves=n)
            else:
                raise ValueError
            dists = []
            n_nbhrs = max(10, n // 10)
            for _ in range(n_nbhrs):
                tree_n = spr_neighbor(tree)
                dists.append(ola_distance(tree, tree_n))
            xs.append(n)
            ys.append(np.average(dists))
    
    ax.scatter(xs, ys, alpha=0.6)

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("OLA distance")
    
    fig.savefig(output)

def plot_avg_nni_distance_vs_tree_size(
    seed=None,
    output="temp.pdf"
):
    """
    create a boxplot of expected distance from a fixed tree to a random NNI-neighbor 
    tree, using randomly seleted NNI neighbors
    """
    # set random seed
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots()

    data = []

    for n in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        for _ in range(10):
            tree = get_random_tree(n_leaves=n)
            dists = []
            n_nbhrs = max(10, n // 10)
            for _ in range(n_nbhrs):
                tree_n = nni_neighbor(tree)
                dists.append(ola_distance(tree, tree_n))
            data.append([n, np.average(dists)])
    
    df = pd.DataFrame(data, columns=("n_leaves", "dist"))
    sns.boxplot(data=df, x="n_leaves", y="dist", native_scale=False, ax=ax)

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("OLA distance")
    
    sns.despine(fig, trim=True)
    fig.savefig(output)

def plot_height_vs_spr_nbhr_max_ola_distance(
    seed=None,
    output="temp.pdf"
):
    # set random seed
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots()

    for n in [20, 50, 100, 200]:
        xs = []
        ys = []
        for _ in range(30):
            tree = get_random_tree(n_leaves=n)
            dists = []
            n_nbhrs = 30
            for _ in range(n_nbhrs):
                tree_n = spr_neighbor(tree)
                dists.append(ola_distance(tree, tree_n))            
            # calculate tree height
            height = tree.get_farthest_node()[1]
            xs.append(height)
            ys.append(np.max(dists))
    
        ax.scatter(xs, ys, alpha=0.8, label=n)

    ax.set_xlabel("Tree height")
    ax.set_ylabel("OLA distance")
    ax.legend()

    # add linear plot for upper bound
    xs = range(2,16)
    ys = [2*x for x in xs]
    sns.lineplot(x=xs, y=ys, color="grey", linestyle="--")

    sns.despine(fig, trim=True)
    fig.savefig(output)

def plot_height_vs_spr_nbhr_avg_ola_distance(
    seed=None,
    output="temp.pdf"
):
    # set random seed
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots()

    for n in [20, 50, 100, 200]:
        xs = []
        ys = []
        for _ in range(30):
            tree = get_random_tree(n_leaves=n)
            dists = []
            n_nbhrs = max(20, n // 5)
            for _ in range(n_nbhrs):
                tree_n = spr_neighbor(tree)
                dists.append(ola_distance(tree, tree_n))            
            # calculate tree height
            height = tree.get_farthest_node()[1]
            xs.append(height)
            ys.append(np.average(dists))
    
        ax.scatter(xs, ys, alpha=0.8, label=n)

    ax.set_xlabel("Tree height")
    ax.set_ylabel("OLA distance")
    ax.legend()

    sns.despine(fig)
    fig.savefig(output)

def plot_ola_neighbor_spr_distance(
    seed=None,
    output="temp.pdf"
):
    """
    Generate pairs of trees which are OLA neighbors, i.e. their OLA distance = 1, and 
    plot their SPR distance vs their size
    """
    # set random seed
    if seed is not None:
        random.seed(seed)
    
    fig, ax = plt.subplots()

    tree_sizes = [5, 10, 20, 50, 100, 200, 500]
    if Path("temp_ola_nbhr").is_file():
        df = pd.read_csv("temp_ola_nbhr")
    else:
        data = []

        for n in tree_sizes:
            print(f"starting on {n} leaves")
            # option 1: take averages
            for i in range(50):
                tree = get_random_tree(n_leaves=n)
                dists = []
                n_nbhrs = max(10, n // 10)
                for _ in range(n_nbhrs):
                    tree_n = ola_neighbor(tree)
                    # compute rSPR distance using Whidden's C program
                    dists.append(rspr_distance(tree, tree_n))
                avg = np.average(dists)
                max_dist = np.max(dists)
                data.append([n, avg])
                print(f"{n} leaves; trial {i}; computed average = {avg}; max = {max_dist}")

            # option 2: no averages
            # for i in range(10):
            #     tree = get_random_tree(n_leaves=n)
            #     n_nbhrs = 10 # max(10, n // 10)
            #     for _ in range(n_nbhrs):
            #         tree_n = ola_neighbor(tree)
            #         # compute rSPR distance using Whidden's C program
            #         dist = rspr_distance(tree, tree_n)
            #         data.append([n, dist])
            #     print(f"{n} leaves; trial {i}; latest distance = {data[-1]}")

        df = pd.DataFrame(data, columns=("n_leaves", "dist"))
        # save data to csv file
        df.to_csv("temp_ola_nbhr")
    ax.set_xscale("log")
    bp = sns.boxplot(
        data=df, x="n_leaves", y="dist", ax=ax,
        native_scale=True,
        # whis=[0, 100],
        # notch=True,
        width=0.8,
        # alpha=0.6,
    )
    bp.set_xticks(tree_sizes)
    bp.set_xticklabels(tree_sizes)
    ax.minorticks_off()
    # ax.set_xticks(tree_sizes, minor=False)
    # ax.set_xticklabels([str(n) for n in tree_sizes])

    ax.set_xlabel("Number of leaves")
    ax.set_ylabel("SPR distance")

    sns.despine(fig, trim=True)
    fig.savefig(output)

def scatterplot_ola_neighbor_spr_distance(
    seed=None,
    output="temp.pdf"
):
    """
    Generate pairs of trees which are OLA neighbors, i.e. their OLA distance = 1, and 
    plot their SPR distance vs their size
    """
    # set random seed
    if seed is not None:
        random.seed(seed)
    
    fig, ax = plt.subplots()

    data = []

    for n in [5, 10, 20, 50, 100, 200, 500]:
        print(f"starting on {n} leaves")
        for i in range(10):
            tree = get_random_tree(n_leaves=n)
            n_nbhrs = 10
            tree_n = ola_neighbor(tree)
            for j in range(n_nbhrs):
                # compute rSPR distance using Whidden's C program
                dist = rspr_distance(tree, tree_n)
                ola_dist = ola_distance(tree, tree_n)
                data.append([n, dist, ola_dist])
                tree_n = ola_neighbor(tree_n)
            print(f"{n} leaves; trial {i}; latest distance = {data[-1]}")

    df = pd.DataFrame(data, columns=("n_leaves", "dist", "ola_dist"))
    sns.scatterplot(
        data=df, x="ola_dist", y="dist", hue="n_leaves",
        palette="husl", alpha=0.5,
        ax=ax,
    )

    ax.set_xlabel("OLA distance")
    ax.set_ylabel("SPR distance")

    sns.despine(fig, trim=True)
    fig.savefig(output)

def plot_random_spr_walks(
    n_leaves=30, n_steps=10, n_runs=2, seed=None, output="temp.pdf"
):
    """
    This function does the following:
        1. Generate a random SPR-walk in the space of trees with the specified number of
            leaves `n_leaves`
        2. Compute the OLA-distance from the i-th tree in the walk to the 0-th tree
        3. Make a plot of the i-th walk index vs the OLA-distance
        4. Repeat steps 1.-3. `n_runs`-many times
        5. Save the plot
    """
    # set random seed
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots()

    for _ in range(n_runs):
        ys = ola_dists_random_spr_walk(n_leaves, n_steps)

        ax.plot(
            ys,
            alpha=0.5,
            rasterized=True,
            marker="o",
            color="C0",
        )
    # ax.set_aspect("equal")

    # fig.suptitle(f"trees with {n_leaves} leaves")
    ax.set_xlabel("SPR steps")
    ax.set_ylabel("OLA distance")

    sns.despine(fig)
    fig.savefig(output)

def plot_random_nni_walks(n_leaves=30, n_steps=10, n_runs=2, seed=None, output="temp.pdf"):
    """
    This function does the following:
        1. Generate a random NNI-walk in the space of trees with the specified number of
            leaves `n_leaves`
        2. Compute the OLA-distance from the i-th tree in the walk to the 0-th tree
        3. Make a plot of the i-th walk index vs the OLA-distance
        4. Repeat steps 1.-3. `n_runs`-many times
        5. Save the plot
    """
    # set random seed
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots()

    for _ in range(n_runs):
        ys = ola_dists_random_nni_walk(n_leaves, n_steps)

        ax.plot(
            ys,
            alpha=0.5,
            rasterized=True,
            marker="o",
            color="C0",
        )
    # ax.set_aspect("equal")

    # fig.suptitle(f"trees with {n_leaves} leaves")
    ax.set_xlabel("NNI steps")
    ax.set_ylabel("OLA distance")

    sns.despine(fig, trim=False)
    fig.savefig(output)

def ola_dists_random_nni_walk(n_leaves=30, n_steps=10):
    """
    This function does the following:
        1. Generate a random NNI-walk in the space of trees with the specified number of
            leaves `n_leaves`
        2. Compute the OLA-distance from the i-th tree in the walk to the 0-th tree
        3. Returns the list of distances
    """
    # create starting tree
    start_tree = get_random_tree(n_leaves=n_leaves)
    tree = start_tree

    # initialize list of distances
    dists = [0]
    for _ in range(n_steps):
        tree = nni_neighbor(tree)
        dist = ola_distance(start_tree, tree)
        dists.append(dist)
    return dists

def ola_dists_random_spr_walk(n_leaves=30, n_steps=10):
    """
    This function does the following:
        1. Generate a random SPR-walk in the space of trees with the specified number of
            leaves `n_leaves`
        2. Compute the OLA-distance from the i-th tree in the walk to the 0-th tree
        3. Returns the list of distances
    """
    # create starting tree
    start_tree = get_random_tree(n_leaves=n_leaves)
    tree = start_tree

    # initialize list of distances
    dists = [0]
    for _ in range(n_steps):
        tree = spr_neighbor(tree)
        dist = ola_distance(start_tree, tree)
        dists.append(dist)
    return dists

def plot_random_spr_walk_vs_spr_distance(
    n_leaves=30, n_steps=10, n_runs=2, output="test.pdf"
):

    fig, ax = plt.subplots()

    for _ in range(n_runs):
        command = (
            f"random_spr_walk/random_spr_walk -ntax {n_leaves} -niterations {n_steps-1} "
            "-sfreq 1 > temp.log"
        )
        run(
            command, 
            shell=True
        )
        remove_line_numbering(file="temp.log")
        run(
            f"cat temp.log | rspr/rspr -pairwise 0 1 > path_rspr_dists.log",
            shell=True
        )
        # xs = range(n_steps)
        ys = np.genfromtxt("path_rspr_dists.log", delimiter=',').flat
        ax.plot(
            # xs,
            ys,
            alpha=0.8,
            rasterized=True
        )
    # ax.set_aspect("equal")

    fig.suptitle(f"random SPR walk on trees, with {n_leaves} leaves and {n_steps} steps")
    ax.set_xlabel('SPR steps')
    ax.set_ylabel('SPR distance')

    fig.savefig(output)

def make_scatterplot_from_lists(file1, file2, output="test.pdf"):
    """
    Example usage:
        make_scatterplot_from_lists(
            "path_rspr_dists.log",
            "path_vec_dists.log"
        )
    Args:
        file1 and file2: csv files that are "list-shaped"
    """
    # file listing rSPR distances
    arr1 = np.genfromtxt(file1, delimiter=',')
    # file listing vector-encoding distances
    arr2 = np.genfromtxt(file2, delimiter=',')
    xs = arr1.flat
    ys = arr2.flat

    fig, ax = plt.subplots()
    # make scatterplot
    ax.scatter(
        xs, 
        ys,
        alpha=0.2,
        rasterized=True,
    )
    ax.set_aspect('equal')

    n_steps = len(ys)
    fig.suptitle(f"random walk on trees, with 30 leaves and {n_steps} steps")
    ax.set_xlabel('rSPR distance')
    ax.set_ylabel('vector-encoding distance')

    fig.savefig(output)

def make_scatterplot_plus_histograms_from_lists(file1, file2, output="test.pdf"):
    """
    Args:
        file1 and file2: csv files that are "list-shaped"
    """
    # file listing rSPR distances
    arr1 = np.genfromtxt(file1, delimiter=',')
    # file listing vector-encoding distances
    arr2 = np.genfromtxt(file2, delimiter=',')
    xs = arr1.flat
    ys = arr2.flat

    fig, axs = plt.subplots(2, 2)
    axs[0][1].axis("off")
    # make scatterplot
    axs[1][0].scatter(
        xs, 
        ys,
        alpha=0.2,
        rasterized=True,
    )
    # show frequencies of x-values
    axs[0][0].bar(*np.unique(xs, return_counts=True))
    # show frequencies of y-values
    axs[1][1].barh(*np.unique(ys, return_counts=True))

    n_steps = len(ys)
    fig.suptitle(f"random walk on trees, with 30 leaves and {n_steps} steps")
    axs[1][0].set_xlabel('rSPR distance')
    axs[1][0].set_ylabel('vector-encoding distance')

    fig.savefig(output)

def make_scatterplot_from_matrices(file1, file2, output="test.pdf"):
    """
    Args:
        file1 and file2: csv files that are "matrix-shaped"
    """
    arr1 = np.genfromtxt(file1, delimiter=',')
    arr2 = np.genfromtxt(file2, delimiter=',')
    # why 944 ??
    xs = arr1[944].flat
    ys = arr2[944].flat
    
    fig, axs = plt.subplots(2, 2)
    axs[0][1].axis("off")
    # make scatterplot
    axs[1][0].scatter(
        xs, 
        ys,
        alpha=0.05,
        rasterized=True)
    # show frequencies of x-values
    axs[0][0].bar(*np.unique(xs, return_counts=True))
    # show frequencies of y-values
    axs[1][1].barh(*np.unique(ys, return_counts=True))

    fig.suptitle("rSPR distance vs. vector distance, at tree [0,1,2,3,4]")
    axs[1][0].set_xlabel('rSPR distance')
    axs[1][0].set_ylabel('vector-encoding distance')

    fig.savefig(output)

def make_animation_from_matrices(
        file1, file2, title_names, texts,
        output="test.gif"):
    """
    Example usage:
        make_animation_from_matrices(
            "all_rspr_distances_6.log",
            "all_vec_distances_6.log",
            list(make_all_vectors(6)),
            [to_tree(x).get_ascii(show_internal=False) for x in make_all_vectors(6)]
        )
    Args:
        title_names: a list of strings to use in title per frame
    """
    arr1 = np.genfromtxt(file1, delimiter=',')
    arr2 = np.genfromtxt(file2, delimiter=',')
    xs = arr1[0]
    ys = arr2[0]
    n_pts = len(xs)
    
    fig, (ax, ax_text) = plt.subplots(1, 2)
    fig.set_size_inches(8, 3)
    plt.subplots_adjust(bottom=0.15)

    sc = ax.scatter(
        xs, ys,
        alpha=0.01,
        rasterized=True,
    )

    # fig.suptitle("rSPR distance vs. vector distance, all trees on 6 leaves")
    ax.set_xlabel('rSPR distance')
    ax.set_ylabel('vector-encoding distance')

    ax_text.axis("off")
    tx = ax_text.text(0, 0, "placeholder")

    def animate(i):
    
        xs = arr1[i] + np.random.uniform(-0.05, 0.05, n_pts)
        ys = arr2[i] + np.random.uniform(-0.05, 0.05, n_pts)
        sc.set_offsets(
            list(zip(xs, ys))
        )
        vec = title_names[i]
        st = fig.suptitle(
            "distances from tree " + str(i) + ": code " + str(vec))
        print("animating step", i)
        tx.set_text(texts[i])
        return sc, st, tx

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=400,
        save_count=945
    )
    ani.save(output)

"""
Writing data to file
"""

def write_random_sample_newicks(n=30, num_trees=100, file="test.log"):
    """
    Samples a random selection of trees on n leaves, writes their newick
    strings to a file 
    """
    with open(file, 'w') as fh:
        for _ in range(num_trees):
            tree = get_random_tree(n)
            newick = tree.write(format=9)
            print(newick, file=fh)

def write_pairwise_vec_distances(n=4, file="test.log"):
    """
    Writes a matrix of vector distances to an output file,
    where rows and columns are indexed by integer vectors
    encoding trees on n leaves
    """
    with open(file, 'w') as fh:

        for vec1 in get_all_vectors(n):
            dists = []
            for vec2 in get_all_vectors(n):
                dist = hamming_dist(vec1, vec2)
                dists.append(dist)
            line = ",".join(str(d) for d in dists)
            print(line, file=fh)

def write_random_tree_pair(n=15, file="test.log"):
    tree1 = Tree(); tree1.populate(n)
    tree2 = Tree(); tree2.populate(n)
    print("first tree encoding:", to_vector(tree1))
    print("second tree encoding:", to_vector(tree2))
    print(ola_distance(tree1, tree2))
    with open(file, 'w') as fh:
        print(tree1.write(format=9), file=fh)
        print(tree2.write(format=9), file=fh)

def write_random_tree_path(n_leaves=30, n_steps=10, file="test.log"):
    """
    Create random OLA-walk on space of trees with `n_leaves`-many leaves, with specified
    number of steps, and save their newick strings to the specified file.
    """
    next_tree = Tree()
    next_tree.populate(n_leaves)
    with open(file, 'w') as fh:
        for _ in range(n_steps):
            next_tree = random_tree_neighbor(next_tree)
            print(next_tree.write(format=9), file=fh)

def read_trees_to_vector_distances(in_file="test.log", out_file=None):
    """
    Each line of `in_file` contains the newick string of a tree. This function computes
    the OLA-distance of each tree to the first tree, and writes these distances to the
    specified `out_file`.
    """
    dists = []
    with open(in_file, 'r') as fh:
        newicks = fh.read().splitlines()
    first_vec = to_vector(Tree(newicks[0]))
    for newick in newicks:
        other_vec = to_vector(Tree(newick))
        dists.append(hamming_dist(first_vec, other_vec))
    if out_file is None:
        return dists
    else:
        with open(out_file, 'w') as fh:
            fh.write(",".join(str(x) for x in dists))

def remove_line_numbering(file="test.log"):
    with open(file, 'r') as fh:
        lines = fh.read().splitlines()
    newicks = []
    for line in lines:
        if ':' in line:
            (_, newick) = line.split(': ')
        else: newick = line
        newicks.append(newick)
    with open(file, 'w') as fh:
        for newick in newicks:
            print(newick, file=fh)


if __name__ == "__main__":

    # plot_random_nni_walks(n_leaves=100, n_steps=50, n_runs=10, seed=168)

    # ola_distance_spr_walk_30_leaves.pdf
    # plot_random_spr_walks(n_leaves=30, n_steps=15, n_runs=10, seed=168)
    # ola_distance_spr_walk_100_leaves.pdf
    # plot_random_spr_walks(n_leaves=100, n_steps=50, n_runs=10, seed=168)
    # ola_distance_spr_walk_500_leaves.pdf
    # plot_random_spr_walks(n_leaves=500, n_steps=250, n_runs=10, seed=168)
    # ola_distance_spr_walk_500_leaves_short.pdf
    # plot_random_spr_walks(n_leaves=500, n_steps=50, n_runs=10, seed=168)
    # ola_distance_spr_walk_300_leaves.pdf
    # plot_random_spr_walks(n_leaves=300, n_steps=150, n_runs=10, seed=168)
    # ola_distance_spr_walk_300_leaves_short.pdf
    # plot_random_spr_walks(n_leaves=300, n_steps=50, n_runs=10, seed=168)
    # ola_distance_spr_walk_1000_leaves_short.pdf
    # plot_random_spr_walks(n_leaves=1000, n_steps=50, n_runs=10, seed=168)
    # ola_distance_spr_walk_3000_leaves_short.pdf
    # plot_random_spr_walks(n_leaves=3000, n_steps=50, n_runs=10, seed=168)

    # plot_avg_nni_distance_vs_tree_size(seed=168)
    # nni_neighbors_distance_boxplot.pdf
    # plot_height_vs_spr_nbhr_avg_ola_distance(seed=168)
    # spr_neighbors_distance_vs_height.pdf
    plot_height_vs_spr_nbhr_max_ola_distance(seed=168)  # spr_neighbors_max_ola_distance_vs_height.pdf

    # plot_avg_spr_distance_vs_tree_size(distribution="yule")

    # plot_ola_neighbor_spr_distance(seed=168)  # spr_distance_w_averages.pdf
    # scatterplot_ola_neighbor_spr_distance(seed=168)
    pass
