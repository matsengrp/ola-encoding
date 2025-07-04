import random
from ete3 import Tree

def spr_neighbor(tree):
    """
    return a random tree which differs from the input tree by one SPR 
    (subtree-prune-regraft) move
    """
    try:
        new_tree = tree.copy()
    except RecursionError:
        # copy tree via Newick string
        newick = tree.write(format=9)
        new_tree = Tree(newick)
    node_list = list(new_tree.traverse())

    # choose regraft location on edge above random node
    regraft_node = random.choice(node_list)

    # choose prune location on edge above random node
    prune_node_found = False
    while not prune_node_found:
        prune_node = random.choice(node_list)
        # check that `prune_node` is not adjacent to `regraft_node`, to avoid identitcal
        # tree after SPR move
        if prune_node.up == regraft_node.up:
            continue
        if prune_node.up == regraft_node:
            continue
        # check that `prune_node` is not an ancestor of `regraft_node`, to avoid invalid
        # SPR move
        mrca = new_tree.get_common_ancestor(prune_node, regraft_node)
        if mrca != prune_node:
            prune_node_found = True

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

def all_spr_neighbors(tree):
    """
    returns an iterator of all SPR neighbors of the input tree
    """
    pass
    
def all_nni_neighbors(tree):
    """
    returns an iterator of all NNI neighbors of the input tree
    """
    pass
