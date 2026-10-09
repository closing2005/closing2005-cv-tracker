"""Hungarian (Kuhn-Munkres) algorithm for optimal linear assignment.

Implemented from scratch on purpose: zero SciPy dependency, O(n^3),
handles rectangular cost matrices by padding with dummy rows/columns.

Used by the tracker to associate predicted track boxes with new
detections at minimum total cost (1 - IoU).
"""

from __future__ import annotations


def linear_sum_assignment(cost):
    """Solve the assignment problem for a rectangular cost matrix.

    Args:
        cost: 2D list/array of shape (n_rows, n_cols), finite floats.

    Returns:
        (row_idx, col_idx): lists of matched row/column indices.
        Unmatched rows/columns (matched to padded dummies) are excluded.
    """
    n_rows = len(cost)
    n_cols = len(cost[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return [], []

    n = max(n_rows, n_cols)
    # Pad to square with a large cost so dummies are only used when needed.
    big = max(max(row) for row in cost) + 1.0 if n_rows and n_cols else 1.0
    a = [[big] * n for _ in range(n)]
    for i in range(n_rows):
        for j in range(n_cols):
            a[i][j] = float(cost[i][j])

    # --- Kuhn-Munkres for rectangular -> square, 1-indexed internals ---
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)      # p[j] = row assigned to column j
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    row_idx, col_idx = [], []
    for j in range(1, n + 1):
        i = p[j]
        if i <= n_rows and j <= n_cols and a[i - 1][j - 1] < big:
            row_idx.append(i - 1)
            col_idx.append(j - 1)
    return row_idx, col_idx
