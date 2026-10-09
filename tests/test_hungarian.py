"""Tests for the from-scratch Hungarian implementation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hungarian import linear_sum_assignment


def total(cost, rows, cols):
    return sum(cost[r][c] for r, c in zip(rows, cols))


def test_square_optimal():
    cost = [[4, 1, 3],
            [2, 0, 5],
            [3, 2, 2]]
    rows, cols = linear_sum_assignment(cost)
    assert sorted(rows) == [0, 1, 2] and sorted(cols) == [0, 1, 2]
    assert total(cost, rows, cols) == 5  # (0,1)+(1,0)+(2,2) = 1+2+2


def test_rectangular_more_cols():
    cost = [[10, 1],
            [1, 10]]
    rows, cols = linear_sum_assignment(cost)
    assert total(cost, rows, cols) == 2  # cross assignment


def test_rectangular_more_rows():
    cost = [[5, 9],
            [1, 2],
            [8, 3]]
    rows, cols = linear_sum_assignment(cost)
    # best: row1->col0 (1), row2->col1 (3) = 4; row0 unmatched
    assert total(cost, rows, cols) == 4
    assert len(rows) == 2


def test_empty():
    assert linear_sum_assignment([]) == ([], [])
    assert linear_sum_assignment([[]]) == ([], [])


def test_ties_still_valid_permutation():
    cost = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    rows, cols = linear_sum_assignment(cost)
    assert sorted(rows) == [0, 1, 2] and sorted(cols) == [0, 1, 2]
    assert total(cost, rows, cols) == 3
