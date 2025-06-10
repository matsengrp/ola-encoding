import random
from ete3 import Tree
from ola.ola_encoding import (
    to_vector, 
    to_tree, 
    random_vector_neighbor, 
    hamming_dist,
)
from ola.utils import get_random_vector
from ola.tree_rearrangement import nni_neighbor

def test_inverse():
    """
    check that the OLA encoding and decoding implementations are inverses
    """
    for _ in range(100):
        n_leaves = random.randrange(50, 500)
        v = get_random_vector(n_leaves)
        t = to_tree(v)
        v_prime = to_vector(t)
        assert v == v_prime

        t_prime = to_tree(v_prime)
        assert t.write(format=9) == t_prime.write(format=9)

def test_encoding():
    t = Tree("(0,1);")
    v = to_vector(t)
    assert v == [0]

    t = Tree("((0,(1,4)),(((2,3),6),(5,7)));")
    v = to_vector(t)
    assert v == [0, -1, 2, 1, -3, -3, 5]

def test_vector_neighbor():
    for _ in range(100):
        v = get_random_vector(30)
        v_n = random_vector_neighbor(v)
        assert 1 == hamming_dist(v, v_n)

def test_nni_move():
    """
    check that if two trees differ by an NNI move, then they have different OLA codes
    """
    for _ in range(100):
        v = get_random_vector(30)
        t = to_tree(v)
        t_prime = nni_neighbor(t)
        v_prime = to_vector(t_prime)
        assert v != v_prime
