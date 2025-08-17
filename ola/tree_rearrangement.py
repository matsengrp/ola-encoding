import random
import subprocess
from ete3 import Tree

def spr_neighbor(tree, max_tries=100):
    """
    return a random tree which differs from the input tree by one SPR 
    (subtree-prune-regraft) move
    """
    # check that input tree has at least 3 leaves
    n_leaves = len(tree.get_leaf_names())
    if n_leaves < 3:
        raise ValueError("Input tree must have at least 3 leaves")
    
    try:
        new_tree = tree.copy()
    except RecursionError:
        # is tree is too large, copy tree via Newick string
        newick = tree.write(format=9)
        new_tree = Tree(newick)
    node_list = list(new_tree.traverse())

    # choose prune-location and regraft-location
    prune_node_found = False
    while not prune_node_found:
        # choose regraft location on edge above random node
        regraft_node = random.choice(node_list)

        # choose prune location on edge above random node
        for _ in range(max_tries):
        # while not prune_node_found:
            prune_node = random.choice(node_list)
            # check that `prune_node` is not adjacent to `regraft_node`, to avoid 
            # identitcal tree after SPR move
            if prune_node.up == regraft_node.up:
                continue
            if prune_node.up == regraft_node:
                continue
            # check that `prune_node` is not an ancestor of `regraft_node`, to avoid
            # invalid SPR move
            mrca = new_tree.get_common_ancestor(prune_node, regraft_node)
            if mrca != prune_node:
                prune_node_found = True
                break
    if not prune_node_found:
        raise RuntimeError("didn't find prune and regraft locations")

    # add new node on the regraft edge, above `regraft_node`
    if not regraft_node.is_leaf():
        regraft_children = [x for x in regraft_node.children] # copy de-referenced list
        new_parent = Tree()
        for node in regraft_children:
            node.detach()
            new_parent.add_child(node)
        regraft_node.add_child(new_parent)
        new_parent = regraft_node
    else: # `regraft_node` is a leaf node
        new_parent = regraft_node.up.add_child()
        regraft_node.detach()
        new_parent.add_child(regraft_node)
    prune_parent = prune_node.up
    # prune subtree below `prune_node`
    prune_node.detach()
    if prune_parent.up is not None:
        prune_parent.delete()
    else: # `prune_parent` is root node
        new_tree = prune_parent.children[0]
        new_tree.up = None
    new_parent.add_child(prune_node)

    return new_tree

def nni_neighbor(tree):
    """
    return a random tree which differs from the input tree by one NNI (nearest-neighbor-
    interchange) move

    i.e. the tree
            /-A
       /(*)|
    --|     \-B
      |
       \-C
    gets changed to either 
            /-C
       /(*)|
    --|     \-B
      |
       \-A
    or
            /-A
       /(*)|
    --|     \-C
      |
       \-B
    """
    try:
        new_tree = tree.copy()
    except RecursionError:
        # if tree is too large, copy tree via Newick string
        newick = tree.write(format=9)
        new_tree = Tree(newick)
    node_list = list(new_tree.traverse())

    # choose an internal node
    int_node_found = False
    while not int_node_found:
        int_node = random.choice(node_list)
        if not int_node.is_leaf() and not int_node.is_root():
            int_node_found = True
    # choose child of internal node at random (1 of 2 choices)
    child_i = random.choice([0, 1])
    child_node = int_node.children[child_i]

    # switch chosen child with chosen node's sister
    sister = int_node.get_sisters()[0]
    int_node.up.remove_child(sister)
    int_node.remove_child(child_node)

    int_node.up.add_child(child_node)
    int_node.add_child(sister)

    return new_tree

# def all_spr_neighbors(tree):
#     """
#     returns an iterator of all SPR neighbors of the input tree
#     """
#     pass
    
# def all_nni_neighbors(tree):
#     """
#     returns an iterator of all NNI neighbors of the input tree
#     """
#     pass

def rspr_distance(tree1, tree2):
    """
    Compute unrooted SPR distance between two trees, using Whidden's C++ program
    """
    nwk1 = tree1.write(format=9)
    nwk2 = tree2.write(format=9)
    command = f"echo -e '{nwk1}\n{nwk2}' | ../../rspr/rspr -total -q"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        # command succeeded, print output
        if result.stdout:
            # print(result.stdout)
            stdout_lines = result.stdout.split("\n")
            dist_line = stdout_lines[-2]
            assert dist_line[:14] == "total distance"
            return int(dist_line.split("=")[1])
    else:
        print("tried command:", command)
        print("resulted in error:", result.stdout)
        raise ValueError(result.stdout)

