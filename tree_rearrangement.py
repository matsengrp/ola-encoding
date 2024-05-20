import random
from ete3 import Tree

def spr_neighbor(tree):
    """
    return a tree which differs from the input tree by one SPR (subtree-prune-regraft)
    move
    """
    new_tree = tree.copy()
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
    return a tree which differs from the input tree by one NNI (nearest-neighbor-
    interchange) move
    """
    pass
