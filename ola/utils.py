from itertools import combinations_with_replacement
from random import randrange
from typing import (
    List
)

from ete3 import Tree
from ola.ola_encoding import (
    to_vector,
    to_tree,
    get_vector_neighborhood,
)

"""
Utility functions
"""

def test_vector_idempotent(n=30):
    vec = get_random_vector(n)
    assert vec == to_vector(to_tree(vec))
    print("Passed test with vector = ", vec)

def test_ete_vector_idempotent(n=30):
    t = Tree()
    t.populate(n)
    vec = to_vector(t)
    assert vec == to_vector(to_tree(vec))
    print("Passed test with tree = ", t.write(format=9), t)

def combine_tree_vectors(left_vec, right_vec):
    n_left = len(left_vec) + 1
    n_right = len(right_vec) + 1
    combined_names = [
        chr(97 + i // 26) + chr(97 + i % 26) for i in range(n_left + n_right)]
    left_tree = to_tree(
        left_vec,
        names=combined_names[:n_left])
    right_tree = to_tree(
        right_vec, 
        names=combined_names[n_left:])
    t = Tree()
    t.add_child(left_tree)
    t.add_child(right_tree)
    return to_vector(t)

def split_tree_children_vectors(vec: List[int]):
    t = to_tree(vec)
    left_child = t.children[0]
    right_child = t.children[1]
    # `.up = None` needed so that result is considered rooted
    left_child.up = None
    right_child.up = None
    return (to_vector(left_child), to_vector(right_child))

def get_root_label_from_vector(vec):
    t = to_tree(vec)
    return t.label

"""
Produce vectors or trees or vector-iterators or tree-iterators
"""

def get_random_vector(n=30):
    """
    Returns a uniformly random lenth-(n - 1) integer vector with the restriction that
    -i <= a_i <= i for all i.
    args:
        n: number of leaves
    """
    vec = [None for _ in range(n - 1)]
    for i in range(n - 1):
        vi = randrange(-i, i + 1)
        vec[i] = vi
    return vec

def get_random_tree(n_leaves=30):
    """
    args:
        n_leaves: number of leaves
    """
    vec = get_random_vector(n_leaves)
    return to_tree(vec)

def get_random_yule_tree(n_leaves=30):
    """
    A Yule tree is a tree constructed by adding a new leaf as the sister to a previous 
    leaf. In terms of the OLA encoding, this means the OLA entires are nonnegative
    args:
        n_leaves: number of leaves
    """
    vec = [None for _ in range(n_leaves - 1)]
    for i in range(n_leaves - 1):
        vi = randrange(0, i + 1)
        vec[i] = vi
    return to_tree(vec)

def get_all_vectors(n=4):
    """
    Returns an iterator which yields all integer vectors of length n - 1,
    which satisfy the constraint that the i-th entry is in the range [-i, i]
    """
    vec = [-i for i in range(n - 1)]
    max_reached = False
    while not max_reached:
        yield vec.copy()

        # increment vector
        vec[-1] += 1
        # carry terms
        for i in range(n-2, 0, -1):
            if vec[i] > i:
                # carry term
                vec[i] = -i
                vec[i - 1] += 1
        # check if max reached
        if vec[0] > 0:
            max_reached = True

def get_all_treeshape_vectors(n=4):
    """
    Returns an iterator which yields vector encodings of tree on n leaves,
    representing all possible tree shapes
    """
    if n == 0:
        return iter(())
    elif n == 1:
        yield []
    elif n == 2:
        yield [0]
    else:
        # n >= 3
        for k in range(1, (n+1) // 2):
            for leftvec in get_all_treeshape_vectors(k):
                for rightvec in get_all_treeshape_vectors(n - k):
                    yield combine_tree_vectors(leftvec, rightvec)
        if n % 2 == 0:
            # n is even
            k = n // 2
            for (leftvec, rightvec) in combinations_with_replacement(
                get_all_treeshape_vectors(k), 2
            ):
                yield combine_tree_vectors(leftvec, rightvec)

def get_all_treeshapes(n=4):
    """
    Returns an iterator which yields trees on n leaves, representing all 
    possible tree shapes
    """
    for vec in get_all_treeshape_vectors(n):
        yield to_tree(vec)

def write_newicks_of_neighborhood(start_vec, file="test.log"):
    nbhd_vecs = get_vector_neighborhood(start_vec)
    with open(file, 'w') as fh:
        for vec in nbhd_vecs:
            tree = to_tree(vec)
            newick = tree.write(format=9)
            print(newick, file=fh)

def write_all_newicks(n=4, file="test.log"):
    """
    Writes all trees on n leaves in newick format, to output file
    """
    all_vecs = get_all_vectors(n)
    with open(file, 'w') as fh:
        for vec in all_vecs:
            tree = to_tree(vec)
            newick = tree.write(format=9)
            print(newick, file=fh)

def gen_all_newicks(n=4):
    """
    Returns an iterator which yields newick strings
    """
    all_vecs = get_all_vectors(n)
    for vec in all_vecs:
        tree = to_tree(vec)
        newick = tree.write(format=9)
        yield newick

