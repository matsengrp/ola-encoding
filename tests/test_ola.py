import random
from ola.ola_encoding import to_vector, to_tree, get_random_vector

def test_inverse():
    for _ in range(100):
        n_leaves = random.randrange(50, 500)
        v = get_random_vector(n_leaves)
        t = to_tree(v)
        v_prime = to_vector(t)
        assert v == v_prime
