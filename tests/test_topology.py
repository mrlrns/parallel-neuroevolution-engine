"""Tests for creature topology generation and mutation invariants.

These cover the genome side of the project: bilateral-symmetry generation in
train.generer_topologie, and the mutation operators in individu.Individu that
must keep every genome internally consistent (index ranges, list lengths,
enough muscles vs. bones) after repeated mutation.
"""

import copy
import random

import pytest

from individu import Individu
from train import generer_topologie


def _mirror_index(x, y, i, axe=400.0, tol=0.1):
    """Find the index of node i's mirror across the central axis, or None."""
    if abs(y[i] - axe) < tol:
        return i
    target_y = 2 * axe - y[i]
    for j in range(len(x)):
        if abs(x[j] - x[i]) < tol and abs(y[j] - target_y) < tol:
            return j
    return None


@pytest.mark.parametrize("seed", range(10))
def test_generer_topologie_is_bilaterally_symmetric(seed):
    random.seed(seed)
    creature = generer_topologie(seed)
    for i in range(len(creature.x)):
        assert _mirror_index(creature.x, creature.y, i) is not None, (
            f"node {i} has no mirror (seed={seed})"
        )


@pytest.mark.parametrize("seed", range(10))
def test_generer_topologie_indices_in_range(seed):
    random.seed(seed)
    creature = generer_topologie(seed)
    n = len(creature.x)
    assert len(creature.y) == n
    assert len(creature.muscle1) == len(creature.muscle2) == len(creature.is_bone)
    for a, b in zip(creature.muscle1, creature.muscle2):
        assert 0 <= a < n
        assert 0 <= b < n
        assert a != b


@pytest.mark.parametrize("seed", range(10))
def test_generer_topologie_not_all_bone_and_min_20pct_bone(seed):
    random.seed(seed)
    creature = generer_topologie(seed)
    nb_os = sum(creature.is_bone)
    n = len(creature.is_bone)
    assert nb_os < n  # never 100% bone
    # The generator tops nb_os up to int(n * 0.2) when it falls short; integer
    # truncation means this can land just under the literal 20% fraction, so
    # compare against the same int() threshold the generator itself uses.
    assert nb_os >= int(n * 0.2)


def _make_simple_individu():
    # A small square with a diagonal: 4 nodes, symmetric around y=400.
    x = [0.0, 100.0, 0.0, 100.0]
    y = [350.0, 350.0, 450.0, 450.0]
    muscle1 = [0, 0, 1, 2]
    muscle2 = [1, 2, 3, 3]
    is_bone = [1.0, 0.0, 0.0, 1.0]
    return Individu(x, y, muscle1, muscle2, is_bone, family=0)


def test_cloner_is_a_deep_independent_copy():
    ind = _make_simple_individu()
    clone = ind.cloner()
    clone.x.append(999.0)
    clone.muscle1.append(0)
    assert 999.0 not in ind.x
    assert len(clone.x) != len(ind.x)


@pytest.mark.parametrize("seed", range(20))
def test_mutate_keeps_genome_internally_consistent(seed):
    random.seed(seed)
    creature = generer_topologie(seed)
    for _ in range(15):
        creature.mutate(
            rate_new_node=0.3, rate_mut_length=0.3, rate_change_bone=0.3, rate_pop_node=0.3
        )
        n = len(creature.x)
        assert len(creature.y) == n
        assert len(creature.muscle1) == len(creature.muscle2) == len(creature.is_bone)
        for a, b in zip(creature.muscle1, creature.muscle2):
            assert 0 <= a < n
            assert 0 <= b < n
        for i in range(n):
            assert _mirror_index(creature.x, creature.y, i) is not None


def test_mutate_never_removes_below_five_nodes_or_two_muscles():
    random.seed(1)
    creature = generer_topologie(1)
    for _ in range(50):
        before = copy.deepcopy(creature)
        creature.mutate(
            rate_new_node=0.0, rate_mut_length=0.0, rate_change_bone=0.0, rate_pop_node=1.0
        )
        nb_muscles = sum(1 for b in creature.is_bone if b == 0.0)
        if len(creature.x) != len(before.x):
            assert len(creature.x) >= 5
            assert nb_muscles >= 2
