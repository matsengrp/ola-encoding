"""
Test how OLA distance changes under shuffling leaf labels
"""
import json
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from subprocess import run

from ola_encoding import (
    to_vector,
    to_tree,
    ola_distance,
)
from utils import (
    get_random_tree,
)
from tree_rearrangement import spr_neighbor

def shuffle_labels(tree, permutation=None):
    """
    returns a new tree whose labels are shuffled according to `permutation`

    Args:
        tree: ete3 Tree object
        permutation (list or None): 
            list of integers from 0 to `n_leaves` - 1, in shuffled order
            if None, then return a random shuffling of labels
    """
    new_tree = tree.copy()
    # create leaf shuffle dictionary from permutation
    leaf_names = sorted(tree.get_leaf_names())
    n_leaves = len(leaf_names)

    # choose random permutation, if not provided in argument
    if permutation is None:
        permutation = list(range(n_leaves))
        random.shuffle(permutation)
    
    assert n_leaves == len(permutation)
    leaf_shuffle = {}
    for i, name in enumerate(leaf_names):
        leaf_shuffle[name] = leaf_names[permutation[i]]
    # iterate through leaves of `new_tree`
    for leaf in new_tree:
        leaf.name = leaf_shuffle[leaf.name]
    return new_tree

def avg_ola_distance_shuffle(tree1, tree2, n_shuffles=10, shuffles=None):
    """
    Args:
        tree1, tree2: ete3 Tree objects
        n_shuffles: integer
        shuffles: list of permutations
    """
    if shuffles is None:
        # generate random shuffles
        names = tree1.get_leaf_names()
        n_leaves = len(names)

        shuffles = []
        for _ in range(n_shuffles - 1):
            perm = list(range(n_leaves))
            random.shuffle(perm)
            shuffles.append(perm)

    # else: `shuffles` is list of leaf ordering given in argument 
    # dist = ola_distance(tree1, tree2)
    # distances = [dist]
    distances = []
    for perm in shuffles:
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
        random.shuffle(perm)
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
    random.shuffle(perm)
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

def plot_avg_ola_dist_on_spr_walk(n_leaves, n_steps, out_file="temp.pdf", seed=None):
    """
    Function does the following:
        1. Choose a random starting tree with `n_leaves` leaves, 
        2. run an SPR walk for `n_steps` steps, and 
        3. plot the average OLA distance from the starting tree, averaging over a random
            choice of leaf label shuffles
    """
    # set random seed
    if seed is not None:
        random.seed(seed)

    # create starting tree
    tree_0 = get_random_tree(n_leaves)
    # create and store shuffles
    perms = [list(range(n_leaves)) for _ in range(10)]
    for perm in perms:
        random.shuffle(perm)
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
    fig, ax = plt.subplots(2)
    for i in range(10):
        ax[0].plot(dists[i], alpha=0.5, color="C0")
    ax[0].plot(avg_dists, alpha=0.9, marker="o", color="C0")
    # plot standard deviations
    ax[1].plot(std_devs, alpha=0.5, marker="o", color="C1")
    ax[1].fill_between(range(n_steps + 1), std_devs, alpha=0.5, color="C1")

    ax[1].set_xlabel(f"SPR steps")
    ax[0].set_ylabel(f"OLA distance")
    ax[1].set_ylabel(f"std. deviation of OLA dist.")

    fig.savefig(out_file)

def correlation_spr_walk_ola_shuffle(
    n_leaves=100, n_steps=100, n_walks=50, 
    load_data=False, out_file="temp.pdf", seed=None
):
    """
    Generates a collection of SPR walks. For each walk, we compute the distance from the
    n-th tree to the starting tree, using RF-distance, OLA-distance, and OLA-distance
    averaged over a collection of 10 random leaf orders
    """
    # set random seed
    if seed is not None:
        random.seed(seed)

    if load_data:
        print("loading data from files")
        with open("temp_RF_dists.json", "r") as fh:
            rf_dists = json.load(fh)
        with open("temp_OLA_dists.json", "r") as fh:
            ola_dists = json.load(fh)
        with open("temp_OLA_avg_dists.json", "r") as fh:
            ola_avg_dists = json.load(fh)
    else:  # compute distances along SPR walks
        # generate random starting trees
        start_trees = [get_random_tree(n_leaves) for _ in range(n_walks)]
        # generate random SPR walk from each start tree
        spr_walks = [[tree] for tree in start_trees]
        for i in range(n_walks):
            tree = start_trees[i]
            for _ in range(n_steps):
                tree = spr_neighbor(tree)
                spr_walks[i].append(tree)
        ## print([[tree.write(format=9) for tree in walk] for walk in spr_walks])
        # write trees to files as newick strings
        for i, walk in enumerate(spr_walks):
            # create file
            with open(f"nwk/walk_{i}.nwk", "w") as fh:
                # write newick strings
                newicks = [tree.write() for tree in walk]
                fh.write("\n".join(newicks))

        # compute RF distances
        rf_dists = [[0] for _ in range(n_walks)]
        for i in range(n_walks):
            tree_0 = spr_walks[i][0]
            for j in range(n_steps):
                tree = spr_walks[i][j + 1]
                d = tree_0.robinson_foulds(tree)[0]
                rf_dists[i].append(d)
        print("RF:", rf_dists)
        # compute OLA distances
        ola_dists = [[0] for _ in range(n_walks)]
        for i in range(n_walks):
            tree_0 = spr_walks[i][0]
            for j in range(n_steps):
                tree = spr_walks[i][j + 1]
                d = ola_distance(tree_0, tree)
                ola_dists[i].append(d)
        print("OLA:", ola_dists)

        # generate random shuffles
        perms = [list(range(n_leaves)) for _ in range(10)]
        for perm in perms:
            random.shuffle(perm)
        # store shuffles in file
        perm_file = "hop/perm_leaf_orders.txt"
        perm_strings = [",".join([str(x) for x in perm]) for perm in perms]
        with open(perm_file, "w") as fh:
            fh.write("\n".join(perm_strings))
        for i in range(10):
            perm_file = f"hop/order_{i+1}.txt"
            with open(perm_file, "w") as fh:
                fh.write(perm_strings[i])
        print(f"done writing shuffles to {perm_file}")
        

        # compute average OLA 
        ola_stddev_dists = [[0] for _ in range(n_walks)]
        ola_avg_dists = [[0] for _ in range(n_walks)]
        for i in range(n_walks):
            tree_0 = spr_walks[i][0]
            for j in range(n_steps):
                tree = spr_walks[i][j + 1]
                d, std = avg_ola_distance_shuffle(tree_0, tree, shuffles=perms)
                ola_avg_dists[i].append(d)
                ola_stddev_dists[i].append(std)
            print(f"  finished OLA avg for walk {i}")
        print("avg OLA:", ola_avg_dists)

        # save data
        with open("temp_RF_dists.json", "w") as fh:
            json.dump(rf_dists, fh)
        with open("temp_OLA_dists.json", "w") as fh:
            json.dump(ola_dists, fh)
        with open("temp_OLA_avg_dists.json", "w") as fh:
            json.dump(ola_avg_dists, fh)
        with open("temp_OLA_stddev_dists.json", "w") as fh:
            json.dump(ola_stddev_dists, fh)
    

    # compute correlation coefficients
    # corr_data = {"RF": [], "OLA": [], "avg_OLA": []}
    data = []
    steps = np.array(range(n_steps + 1))
    for i in range(n_walks):
        for k in range(20, n_steps+1, 20):
            steps_to_k = steps[:k + 1]
            rf_d = rf_dists[i][:k + 1]
            ola_d = ola_dists[i][:k + 1]
            avg_ola_d = ola_avg_dists[i][:k + 1]
            rf_corr = np.corrcoef(steps_to_k, rf_d)[0, 1]
            ola_corr = np.corrcoef(steps_to_k, ola_d)[0, 1]
            avg_ola_corr = np.corrcoef(steps_to_k, avg_ola_d)[0, 1]

            # corr_data["RF"].append((k, rf_corr))
            # corr_data["OLA"].append((k, ola_corr))
            # corr_data["avg_OLA"].append((k, avg_ola_corr))

            data.append(["RF", k, rf_corr, i])
            data.append(["OLA", k, ola_corr, i])
            data.append(["mean OLA", k, avg_ola_corr, i])
    # print("correlations:\n", corr_data)

    df = pd.DataFrame(data, columns=("dist_type", "step_limit", "corr", "walk_id"))

    fig, ax = plt.subplots()

    # palette = [sns.color_palette()[i] for i in [0, 5, 8]]
    # palette = [sns.color_palette("Paired")[i] for i in [1, 3, 2]]
    palette = [sns.color_palette("Paired")[i] for i in [1, 5, 4]]
    sns.boxplot(
        data=df, x="step_limit", y="corr", hue="dist_type", palette=palette, ax=ax
    )
    ax.set_xlabel("Number of SPR steps")
    ax.set_ylabel("Pearson correlation")
    ax.legend(
        bbox_to_anchor=(1.21, 0.03), loc="lower right", title="distance"
    )

    sns.despine(fig, offset=2, trim=True)
    # plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")

def newick_files_to_hop_dists(out_file="temp.json"):
    n_walks = 50
    hop_dists = [[0] for _ in range(n_walks)]
    for i in range(n_walks):
        # run CEDAR script to convert newick strings to HOP / LTS vectors
        cedar_script = "../../CEDAR/src/CEDAR.py"
        command = (
            f"python {cedar_script} fromNewick --input_file nwk/walk_{i}.nwk "
            f"--output_file hop/walk_{i}.vec --order_file hop/order.txt"
        )
        run(command, shell=True)

        # run CEDAR script to convert list of HOP strings to HOP similarities
        command = (
            f"python {cedar_script} HOP_sim --input_file hop/walk_{i}.vec "
            f"--output_file hop/sim_walk_{i}.txt --mode first"
        )
        run(command, shell=True)

        # extract HOP distances from HOP similarities
        with open(f"hop/sim_walk_{i}.txt", "r") as fh:
            lines = fh.readlines()
            hop_sims_i = [int(line.split(",")[2]) for line in lines[2:]]
            hop_dists_i = [100 - s for s in hop_sims_i]
            assert len(hop_dists_i) == 100
        first_dist = hop_dists_i[:5]
        print("walk ", i, ":", first_dist)
        hop_dists[i].extend(hop_dists_i)
    # save data
    with open(out_file, "w") as fh:
        json.dump(hop_dists, fh)

def newick_files_to_hop_shuffle_dists(out_file="temp.json"):
    n_walks = 50
    n_steps = 100
    hop_avg_dists = [[0] for _ in range(n_walks)]
    for i in range(n_walks):
        print("starting walk", i)
        # choose leaf order
        walk_i_hop_dists = []
        for j in range(1, 11):
            
            # run CEDAR script to convert newick strings to HOP / LTS vectors
            cedar_script = "../../CEDAR/src/CEDAR.py"
            command = (
                f"python {cedar_script} fromNewick --input_file nwk/walk_{i}.nwk "
                f"--output_file hop/walk_{i}_{j}.vec --order_file hop/order_{j}.txt"
            )
            run(command, shell=True)

            # run CEDAR script to convert list of HOP strings to HOP similarities
            command = (
                f"python {cedar_script} HOP_sim --input_file hop/walk_{i}_{j}.vec "
                f"--output_file hop/sim_walk_{i}_{j}.txt --mode first"
            )
            run(command, shell=True)

            # extract HOP distances from HOP similarities
            with open(f"hop/sim_walk_{i}_{j}.txt", "r") as fh:
                lines = fh.readlines()
            hop_sims_ij = [int(line.split(",")[2]) for line in lines[2:]]
            hop_dists_ij = [100 - s for s in hop_sims_ij]
            assert len(hop_dists_ij) == 100
            walk_i_hop_dists.append(hop_dists_ij)

            first_dist = hop_dists_ij[:5]
            print("walk", i, ", shuffle", j, ":", first_dist)
        # average the distances in `walk_i_hop_dists`
        dists_tr = [[row[k] for row in walk_i_hop_dists] for k in range(n_steps)]
        dists_tr_avg = [np.average(col) for col in dists_tr]
        hop_avg_dists[i].extend(dists_tr_avg)
    # save data
    with open(out_file, "w") as fh:
        json.dump(hop_avg_dists, fh)


# def correlation_hop_dist_spr_walk(n_steps=100, n_walks=50):
#     # compute correlation coefficients
#     data = []
#     steps = np.array(range(n_steps + 1))
#     for i in range(n_walks):
#         for k in range(5):
#             pass
#     pass

def spr_walk_ola_shuffle_stddev(out_file="temp.pdf"):
    # load data
    with open("temp_OLA_stddev_dists.json", "r") as fh:
        data = json.load(fh)
        print("data loaded")
    
    data_sparse = []
    for xs in data:
        # normalize by n_leaves, which is ~max. OLA distance
        data_sparse.append([x * 0.01 for x in xs[5::5]])

    fig, ax = plt.subplots()
    df = pd.DataFrame(data_sparse, columns=range(5, 101, 5))
    # print("dataframe:\n", df)

    color = sns.color_palette("pastel")[0]
    # palette = sns.color_palette("muted")[:1]
    # palette = sns.color_palette("muted")[9:10]
    sns.boxplot(data=df, color=color)
    ax.set_xlabel("SPR steps")
    ax.set_ylabel("Std. dev. of normalized OLA distance")

    sns.despine(fig, offset=10, trim=True)
    fig.savefig(out_file, bbox_inches="tight")

def spr_walk_ola_shuffle_avg(out_file="temp.pdf"):
    # load data
    with open("temp_OLA_avg_dists.json", "r") as fh:
        data = json.load(fh)
        print("data loaded")
    
    data_sparse = []
    for xs in data:
        # normalize by n_leaves, which is ~max. OLA distance
        data_sparse.append([x * 0.01 for x in xs[5::5]])

    fig, ax = plt.subplots()
    df = pd.DataFrame(data_sparse, columns=range(5, 101, 5))
    # print("dataframe:\n", df)

    color = sns.color_palette("pastel")[0]
    sns.boxplot(data=df, color=color)
    ax.set_xlabel("SPR steps")
    ax.set_ylabel("Mean normalized OLA distance")

    sns.despine(fig, offset=10, trim=True)
    fig.savefig(out_file, bbox_inches="tight")

def spr_walk_ola_noshuffle(out_file="temp.pdf"):
    # load data
    with open("temp_OLA_dists.json", "r") as fh:
        data =json.load(fh)
        print("data loaded")

    data_sparse = []
    for xs in data:
        # normalize by n_leaves, which is ~max. OLA distance
        data_sparse.append([x * 0.01 for x in xs[5::5]])

    fig, ax = plt.subplots()
    df = pd.DataFrame(data_sparse, columns=range(5, 101, 5))

    color = sns.color_palette("pastel")[0]
    sns.boxplot(data=df, color=color)
    ax.set_xlabel("SPR steps")
    ax.set_ylabel("Normalized OLA distance")

    sns.despine(fig, offset=10, trim=True)
    fig.savefig(out_file, bbox_inches="tight")

def table_correlation_spr_walk_ola_shuffle(
    n_leaves=100, n_steps=100, n_walks=50, 
    load_data=True, seed=None
):
    if not load_data:
        raise ValueError("not implemented")
    else:
        print("loading data from files")
        with open("temp_RF_dists.json", "r") as fh:
            rf_dists = json.load(fh)
        with open("temp_OLA_dists.json", "r") as fh:
            ola_dists = json.load(fh)
        with open("temp_OLA_avg_dists.json", "r") as fh:
            ola_avg_dists = json.load(fh)
        with open("temp_HOP_dists.json", "r") as fh:
            hop_dists = json.load(fh)
        with open("temp_HOP_avg_dists.json", "r") as fh:
            hop_avg_dists = json.load(fh)

    d_types = ["RF", "OLA", "avg_OLA", "HOP", "avg_HOP"]
    d_data = {
        "RF": rf_dists,
        "OLA": ola_dists,
        "avg_OLA": ola_avg_dists,
        "HOP": hop_dists,
        "avg_HOP": hop_avg_dists,
    }
    # compute correlation coefficients
    corr_means = {x: [] for x in d_types}
    corr_medians = {x: [] for x in d_types}
    corr_data = []

    steps = np.array(range(n_steps + 1))
    for k in range(10, n_steps+1, 10):
        k_data = {x: [] for x in d_types}
        steps_to_k = steps[k - 10:k + 1]
        print("current steps =", steps_to_k)
        for i in range(n_walks):
            for dist in d_types:
                dists_to_k = d_data[dist][i][k - 10:k + 1]
                if len(dists_to_k) != 11:
                    print(f"k={k}, i={i}, type={dist}, num dists={len(dists_to_k)}")
                corr = np.corrcoef(steps_to_k, dists_to_k)[0, 1]
                if np.isnan(corr):
                    print(f"warning occurred at k={k}, i={i}, type={dist},")
                    print("  data:", dists_to_k)
                    print("  correlation:", corr)
                    corr = 0.0
                k_data[dist].append(corr)
                corr_data.append((k, dist, corr, i))

        # average over all walks
        for d_type in d_types:
            corr_means[d_type].append(np.average(k_data[d_type]))
            corr_medians[d_type].append(np.median(k_data[d_type]))

    means_df = pd.DataFrame(corr_means)
    medians_df = pd.DataFrame(corr_medians)
    corr_df = pd.DataFrame(
        corr_data, columns=("steps", "dist_type", "corr", "walk_idx")
    )
    for df in [means_df, medians_df]:
        df["steps"] = [i for i in range(10, 101, 10)]

    # write data to csv's
    means_df.to_csv("temp_corr_means.csv", index=False)
    medians_df.to_csv("temp_corr_medians.csv", index=False)
    corr_df.to_csv("temp_corr_data.csv", index=False)
    # print("Means:\n", means_df)
    # print("Medians:\n", medians_df)
    with open("temp_SPR_walk_correlations.txt", "w") as fh:
        fh.write("\n".join(["Means:", str(means_df), "", "Medians:", str(medians_df)]))

def plot_correlation(show_means=False):
    corr_df = pd.read_csv("temp_corr_data.csv")

    fig, ax = plt.subplots()
    
    df = corr_df
    df["dist type"] = df["dist_type"].map({
        "RF": "RF",
        "OLA": "OLA",
        "HOP": "HOP",
        "avg_OLA": "mean OLA",
        "avg_HOP": "mean HOP",
    })

    palette = [sns.color_palette("Paired")[i] for i in [1, 5, 3]]
    if show_means:
        types = ["RF", "mean OLA", "mean HOP"]
    else:
        types = ["RF", "OLA", "HOP"]

    lp = sns.lineplot(
        data=df, x="steps", y="corr", hue="dist type", 
        hue_order=types,
        palette=palette,
        markers=True
    )
    spr_dists = [x for x in range(10, 101, 10)]
    lp.set_xticks(spr_dists)
    lp.set_xticklabels(spr_dists)
    # lp.set_yticks([y * 0.01 for y in range(90, 100, 2)])
    ax.legend().set_title(None)
    if True or show_means:
        sns.move_legend(lp, "upper right", bbox_to_anchor=(0.9, 1.1))

    ax.set_xlabel("SPR steps")
    ax.set_ylabel("Pearson correlation")
    sns.despine(fig, trim=True)
    fig.savefig("temp.pdf", bbox_inches="tight")

def near_mid_far_test(n_leaves=200, n_perms=10, output="temp.pdf", seed=None):
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
        random.shuffle(perm)
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


if __name__ == "__main__":
    # plot_dist_vs_shuffle_on_spr_walk(n_leaves=500, n_steps=60)
    # plot_shuffled_dist_on_spr_walk(n_leaves=300, n_steps=30)

    # plot_avg_ola_dist_on_spr_walk(n_leaves=100, n_steps=100, seed=168)
    # ola_dist_shuffling_on_spr_walk.pdf

    # near_mid_far_test(n_leaves=200, seed=168)

    # correlation_spr_walk_ola_shuffle(
    #     n_leaves=100, n_steps=100, n_walks=50, load_data=False, seed=168
    # )
    # corr_spr_steps_vs_ola_shuf_ola_50.pdf

    # table_correlation_spr_walk_ola_shuffle()
    plot_correlation(show_means=False)

    # spr_walk_ola_shuffle_stddev()
    # spr_walk_stddev_ola_shuffle.pdf
    # spr_walk_ola_shuffle_avg()
    # spr_walk_avg_ola_shuffle.pdf
    # spr_walk_ola_noshuffle()

    # newick_files_to_hop_dists()
    # newick_files_to_hop_shuffle_dists()
    pass
