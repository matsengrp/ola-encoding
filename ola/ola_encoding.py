"""
package to encode a phylogenetic tree on n leaves as an integer vector

The encoding works via applying a canonical internal-node labelling to the tree; the
vector then records where leaves are added iteratively
"""

import warnings
from random import randrange
from ete3 import Tree

def to_vector(tree, alph_leaf_names=False):
    """
    OLA encoding algorithm
    Args:
        tree: ete3.Tree object. tree is assumed to be rooted and bifurcating, and have 
            consecutive integers 0, 1, 2, ... as leaf names.
        alph_leaf_names: Bool. if `true`, input tree can have arbitrary distinct leaf
            names.
    Returns:
        a list of integers, which encodes the input tree
    """
    if not tree.is_root():
        raise ValueError("input tree should be rooted")

    n_leaves = len(tree.get_leaf_names())
    # handle small cases, <= 2 leaves
    if n_leaves < 1:
        raise ValueError("input tree should have at least 1 leaf")
    if n_leaves == 1:
        return []
    elif n_leaves == 2:
        return [0]
    
    # tree has 3 or more leaves
    # make copy of input tree
    tree_copy = Tree(
        tree.write(
            features=["label"], 
            format_root_node=True,
            format=9))

    if alph_leaf_names:
        # sort leaf names in alphabetical order
        sorted_leaf_names = sorted(tree_copy.get_leaf_names())
        leaf_to_idx = {}
        for (idx, name) in enumerate(sorted_leaf_names):
            leaf_to_idx[name] = idx
        # rename leaf name to index
        for leaf in tree_copy:
            idx = leaf_to_idx[leaf.name]
            leaf.name = str(idx)

    # from here, assume leaf names are "0", "1", "2", ..., "n-1"
    # construct list of leaves, ordered by label, for `tree_copy`
    label_to_leaf = [None for _ in range(n_leaves)]
    for leaf in tree_copy:  # iterates through leaves
        idx = int(leaf.name)
        label_to_leaf[idx] = leaf

    # perform internal node labelling
    for node in tree_copy.traverse(strategy='postorder'):
        if node.is_leaf():
            idx = int(node.name)
            node.label = idx
            node.clade_founder = idx
        else:
            # `clade_founder` = minimum label of descendant subclade 
            children_mins = [child.clade_founder for child in node.children]
            node.clade_founder = min(children_mins)
            node.clade_splitter = max(children_mins)
            node.label = - node.clade_splitter
    
    # add "grandparent-root" node so that every node in `tree_copy` has a parent
    tree_grandparent = Tree()
    tree_grandparent.add_child(tree_copy)
    # initialize encoding vector
    vector = []
    # fill in vector via tree deconstruction, removing one leaf at a time
    for i in range(n_leaves - 1):
        idx = n_leaves - 1 - i
        # find sister node of leaf idx
        leaf = label_to_leaf[idx]
        sister = leaf.get_sisters()[0]
        # assign vector entry (to be reversed later)
        vector.append(int(sister.label))
        # delete node, and its parent, from tree_copy
        leaf.up.up.add_child(sister)
        leaf.up.detach()
    vector.reverse()
    # print(f"running ola encoding with {n_leaves} leaves")
    return vector

def to_tree(vector, names=None):
    """
    OLA decoding algorithm
    Args:
        vector: list of integers with i-th entry in range {-i, ..., i}
        names: list of strings to be used as names of leaf nodes, strings
            should be distinct. Default choice is ["0", "1", "2", ...]
    Returns:
        ete3.Tree object encoded by vector
    """
    # check input vector is "valid"
    for i, vi in enumerate(vector):
        if not isinstance(vi, int):
            raise ValueError(
                f"input vector should should have int entries; given input={vector}"
            )
        if abs(vi) > i:
            raise ValueError(
                f"input vector entry vector[{i}] should be between -{i} and {i} "
                f"(inclusive), but given input has vector[{i}]={vi}"
            )
    n_leaves = len(vector) + 1
    if names is None:
        # generate default names ['0', '1', '2', ...]
        names = default_names(n_leaves)
    else:
        # ensure that `names` is a list of strings
        names = list(names) # names.copy()
        for i, name in enumerate(names):
            names[i] = str(name)
    # ensure that enough names are provided
    if len(names) < n_leaves:
        raise ValueError(
            "must provide at least n + 1 names, where n is the "
            "length of the encoding vector")
    if len(set(names)) < len(names):
        warnings.warn("leaf names provided are not distinct")

    # initialize label-to-node list, to avoid cost of tree search
    label_to_node = [None for _ in range(2 * n_leaves - 1)]

    # initialize tree; note `tree` has an extra "grand-root" node
    tree = Tree()
    child_0 = tree.add_child(name=names[0])
    child_0.label = 0
    label_to_node[0] = child_0

    # build tree iteratively, one leaf at a time
    for i in range(1, n_leaves):
        idx = vector[i - 1]
        # get node with label=idx, using stored list
        subtree = label_to_node[idx]
        # attach i-th leaf as sister of `subtree`, subdividing its parent-edge
        # to subdivide parent edge: add new node as sister of subtree-root
        new_node = subtree.add_sister(name="int-node-" + str(i))
        new_node.label = -i
        label_to_node[-i] = new_node
        # to subdivide parent edge: then move current subtree to lie below new node
        subtree.detach()
        new_node.add_child(subtree)
        # add new leaf under new (internal-)node
        child = new_node.add_child(name=names[i])
        child.label = i
        label_to_node[i] = child

    # return tree with "grand-root" removed
    final_tree = tree.children[0]
    # `.up = None` needed so that result is considered rooted
    final_tree.up = None
    return final_tree

def default_names(n, type="num"):
    """
    generate default names ['0', '1', '2', ...] used in `to_tree` function. If type 
    parameters "alpha" is selected, then output will be ['aa', 'ab', 'ac', ...]
    Args:
        n = number of names to output
        type = "num" or "alph"
    """
    if type == "num":
        names = [str(i) for i in range(n)]
    elif type == "alph":
        names = [chr(97 + i % 26) for i in range (n)]
        period = 26
        while period < n:
            names = [names[i // period][-1] + names[i] for i in range(n)]
            period = 26 * period
    else:
        raise ValueError("type parameter must be \"num\" or \"alph\"")
    return names


"""
Multifurcating versions
"""
def to_vector_multifurcating(tree, debugging=False):
    """
    WARNING: not tested
    args:
        tree: ete3.Tree object. tree is assumed to be rooted 
            If input tree is not bifurcating, method will return a vector which
            encodes a bifurcating resolution of the input.

    """
    if not tree.is_root():
        raise ValueError("input tree should be rooted")

    n_leaves = len(tree.get_leaf_names())
    if n_leaves < 1:
        raise ValueError("input tree should have at least 1 leaf")
    # handle small cases, < 2 leaves
    if n_leaves == 1:
        return []
    elif n_leaves == 2:
        return [0]
    
    # if tree has 3 or more leaves
    sorted_leaves = sorted(tree.get_leaf_names())
    if debugging: print("sorted_leaves:", sorted_leaves)
    
    # perform internal node labelling
    for node in tree.traverse(strategy='postorder'):
        if node.is_leaf():
            idx = sorted_leaves.index(node.name)
            node.label = [idx]
            node.unmatched_labels = [idx]
        else:
            # edge label = union of umatched labels of children nodes, except
            # with minimum label removed
            children_labels = []
            for child in node.children:
                children_labels += child.unmatched_labels
            if debugging: print("current unmatched labels:", children_labels)
            min_label = min(children_labels)
            node.unmatched_labels = [min_label]
            children_labels.remove(min_label)
            node.label = sorted(children_labels)
            if debugging: print(
                    "at node above", node.get_leaf_names(),
                    "assigning edge label:", node.label)
    
    # initialize encoding vector
    vector = [None] * (n_leaves - 1)
    for node in tree.traverse(strategy='preorder'):
        if node.is_leaf():
            continue
        for i in node.label:
            ## debugging
            if debugging: print("determining vector entry at pos.", i)
            if i == 0:
                continue
            if i > min(node.label):
                vector[i - 1] = - min(node.label) - 0.5
                continue
            # determine i-th entry of encoding vector, which is the 
            # first-encountered edge label below edge (-i) which has 
            # absolute value < i
            # 
            # (this is the edge label where the i-th leaf is attached when 
            # the tree is built iteratively)
            subnode = node
            found_label = False
            while not found_label:
                children = subnode.children
                assert len(children) > 0, (
                    "ERROR: reached no children without finding vector entry"
                )
                for child in children:
                    if child.is_leaf():
                        label = child.label[0]
                        # check label-found condition
                        if label < i:
                            vector[i - 1] = label
                            found_label = True
                            if debugging: 
                                print(
                                    "at leaf", child.get_leaf_names(),
                                    "found vector entry=", 
                                    vector[i - 1], "at pos.", i)
                            break
                    # if `child` is not a leaf node
                    for label in child.label:
                        # check label-found condition
                        if abs(label) < i:
                            vector[i - 1] = - label
                            found_label = True
                            if debugging: 
                                print(
                                    "at node over", child.get_leaf_names(),
                                    "found vector entry=", vector[i - 1], "at pos.", i)
                            break
                    if found_label: break
                # if label still not found, move down tree
                for child in children:
                    # check which child has small-enough label
                    if abs(child.unmatched_labels[0]) < i:
                        subnode = child

    return vector

"""
OLA distance
"""

def hamming_dist(vec1, vec2):
    """
    Computes hamming distance between input vectors
    Args:
        two lists, assumed to have the same length
    """
    return sum(a != b for (a, b) in zip(vec1, vec2))

def hamming_dist_of_encodings(tree1, tree2):
    """
    Alias for `ola_distance`
    """
    return ola_distance(tree1, tree2)

def ola_distance(tree1, tree2):
    """
    Computes OLA distance between the input trees, which is the hamming distance between
    OLA vector encodings. Assumes that input trees have the same leaf set.
    """
    vec1 = to_vector(tree1)
    vec2 = to_vector(tree2)
    return hamming_dist(vec1, vec2)

def get_vector_neighborhood(start_vec):
    """
    Returns a list of vectors which have hamming distance 1 from start_vec
    """
    neighbors = []
    for i in range(len(start_vec)):
        for entry in range(-i, i + 1):
            if entry == start_vec[i]:
                continue
            new_vec = start_vec.copy()
            new_vec[i] = entry
            neighbors.append(new_vec)
    return neighbors

def get_tree_neighborhood(start_tree):
    """
    Returns a list of trees whose vector encodings have hamming distance 1 from
    the vector encoding of start_tree
    """
    start_vec = to_vector(start_tree)
    return [to_tree(vec) for vec in get_vector_neighborhood(start_vec)]

def random_vector_neighbor(start_vec):
    """
    Returns a "valid" integer vector which has hamming distance 1 
    from start_vec
    Idea of process:
        (i, j) is a randomly generated point in an n x n square, which lies
        outside the main diagonal. We then modify the vector entry at position
        max(i, j). If i = max(i, j) we modify entry to +j; if j = max(i, j) we
        modify entry to -i. However, we make a slight adjustment to guarantee
        modified entry is different from old entry 
    """
    n = len(start_vec)
    i = randrange(0, n)
    j = randrange(0, n - 1)
    # guarantee that i != j
    if j >= i:
        j += 1
    k = max(i, j)

    old_val = start_vec[k]
    # set new_val to i - j
    new_val = i - j
    if new_val > 0:
        # modify new_val to guarantee it is new
        if new_val <= old_val:
            new_val -= 1
    else: # new_val is negative
        # modify new_val to guarantee it is new
        if new_val >= old_val:
            new_val += 1
    new_vec = start_vec.copy()
    new_vec[k] = new_val
    return new_vec

def lazy_random_vector_neighbor(start_vec):
    """
    Like `get_random_vector_neighbor` but allows output to be same as `start_vec`
    """
    n = len(start_vec)
    i = randrange(0, n)
    j = randrange(0, n)
    k = max(i, j)

    # old_val = start_vec[k]
    new_val = i - j

    new_vec = start_vec.copy()
    new_vec[k] = new_val
    return new_vec

def random_tree_neighbor(start_tree):
    start_vec = to_vector(start_tree)
    new_vec = random_vector_neighbor(start_vec)
    return to_tree(new_vec)
    
def lazy_random_tree_neighbor(start_tree):
    """
    Like `get_random_tree_neighbor` but allows output to be same as `start_tree`
    """
    start_vec = to_vector(start_tree)
    new_vec = lazy_random_vector_neighbor(start_vec)
    return to_tree(new_vec)
    


if __name__ == "__main__":

    # tree1 = Tree("((0,2),(3,(4,(5,1))));")
    # tree2 = Tree("((0,2),(((3,4),5),1));")
    # print(to_vector(test_tree))
    # print(test_tree.get_ascii(attributes=['label', 'name']))

    # vector = get_random_vector(7)
    # print("random vector: \n", vector)
    # print(
    #     "tree form:\n", 
    #     to_tree(vector).get_ascii(attributes=['label', 'name']))

    pass

    