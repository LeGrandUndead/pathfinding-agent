"""
=============================================================================
TIER 2 — Agent / Logic Layer
=============================================================================
Implements A* (informed) and BFS (uninformed) search algorithms.
Entirely decoupled from rendering — depends only on environment.py.

Key data structures
-------------------
OPEN LIST  : heapq min-heap     -> O(log n) insert/pop
CLOSED SET : Python hash set    -> O(1)     membership test

Metrics tracked (for the final report)
---------------------------------------
nodes_explored  : |CLOSED set|  -- space complexity proxy
path_cost       : g(goal)       -- solution quality / optimality check
=============================================================================
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math


# ── Heuristic functions ─────────────────────────────────────────────────────

def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    h(n) = sqrt((r1-r2)^2 + (c1-c2)^2)

    Used as the heuristic because the environment allows diagonal moves
    (cost ~1.414). Euclidean distance is admissible for 8-connected grids
    since the true path cost is always >= the straight-line distance.
    """
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    h(n) = |r1-r2| + |c1-c2|

    Admissible for 4-connected grids (no diagonals). Included for comparison
    and used in the test suite for the orthogonal-only scenario.
    """
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


# ── Core data structures ────────────────────────────────────────────────────

@dataclass(order=True)
class Node:
    """Priority-queue node. Only f drives heap ordering."""
    f: float
    state:  Tuple[int, int]      = field(compare=False)
    parent: Optional['Node']     = field(default=None, compare=False)
    g:      float                = field(default=0.0,  compare=False)
    action: str                  = field(default="",   compare=False)


@dataclass
class SearchResult:
    """Unified result returned by every search algorithm."""
    algorithm:      str
    success:        bool
    path:           List[Tuple[int, int]]  # ordered (row, col) cells
    nodes_explored: int                    # |CLOSED| -- space complexity
    path_cost:      float                  # g(goal)  -- time/quality metric
    explored_order: List[Tuple[int, int]]  # visit order (for animation)


# ── Agent class ─────────────────────────────────────────────────────────────

class PathfindingAgent:
    """
    Stateless search agent operating over a GridEnvironment.

    Agent-Environment interaction cycle
    ------------------------------------
    1. PERCEPTION : agent calls env.get_successors(current_state)
    2. EVALUATION : score each successor with f(n) = g(n) + h(n)
    3. ACTION     : select lowest-f node from OPEN LIST (min-heap)
    4. STATE      : new (row, col) becomes the current state
    """

    def __init__(self, env):
        self.env = env

    # ── A* Search (Informed) ────────────────────────────────────────────

    def astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchResult:
        """
        A* Search with Euclidean distance heuristic.

        Evaluation function  : f(n) = g(n) + h(n)
        g(n) = cost from start to n  (known, exact)
        h(n) = Euclidean distance to goal  (estimated, admissible)

        Completeness : YES (finite graph, closed set prevents cycles)
        Optimality   : YES (admissible heuristic guarantees shortest path)
        Time/Space   : O(b^d) where b = branching factor, d = solution depth
        """
        # ── Initialise ──────────────────────────────────────────────────
        frontier:       List[Node]           = [Node(euclidean_distance(start, goal), start)]
        explored_order: List[Tuple[int,int]] = []
        # best g-cost seen per state (lazy-deletion optimisation)
        g_best = {start: 0.0}
        # closed set (VISIT hash table)
        closed = set()

        # ── Generic Search Loop ─────────────────────────────────────────
        while frontier:
            current = heapq.heappop(frontier)          # SELECT: min f(n)

            # Lazy-deletion: skip stale entries
            if current.state in closed:
                continue

            # VISIT: add to CLOSED set
            closed.add(current.state)
            explored_order.append(current.state)

            # GOAL TEST
            if current.state == goal:
                return SearchResult(
                    algorithm      = "A* (Euclidean)",
                    success        = True,
                    path           = self._reconstruct(current),
                    nodes_explored = len(closed),
                    path_cost      = round(current.g, 3),
                    explored_order = explored_order,
                )

            # EXPAND: generate successors
            for nxt, act, cost in self.env.get_successors(current.state):
                if nxt in closed:
                    continue
                new_g = current.g + cost
                if new_g < g_best.get(nxt, float("inf")):
                    g_best[nxt] = new_g
                    f = new_g + euclidean_distance(nxt, goal)
                    heapq.heappush(frontier, Node(f, nxt, current, new_g, act))

        return SearchResult("A* (Euclidean)", False, [], len(closed), 0.0, explored_order)

    # ── BFS (Uninformed baseline) ───────────────────────────────────────

    def bfs(self, start: Tuple[int, int], goal: Tuple[int, int]) -> SearchResult:
        """
        Breadth-First Search — level-by-level FIFO expansion.

        No heuristic is used (f(n) = g(n) only).

        Completeness : YES
        Optimality   : YES (uniform cost = 1 per step, orthogonal only)
        Time/Space   : O(b^d) — typically much larger constant than A*
        """
        queue          = deque([Node(0, start)])
        visited        = {start}
        explored_order = []

        while queue:
            current = queue.popleft()

            if current.state == goal:
                path = self._reconstruct(current)
                return SearchResult(
                    algorithm      = "BFS (Uninformed)",
                    success        = True,
                    path           = path,
                    nodes_explored = len(visited),
                    path_cost      = round(current.g, 3),
                    explored_order = explored_order,
                )

            explored_order.append(current.state)
            for nxt, act, cost in self.env.get_successors(current.state):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(Node(0, nxt, current, current.g + cost, act))

        return SearchResult("BFS (Uninformed)", False, [], len(visited), 0.0, explored_order)

    # ── Path reconstruction ─────────────────────────────────────────────

    @staticmethod
    def _reconstruct(node: Node) -> List[Tuple[int, int]]:
        """Walk parent pointers from goal back to start, then reverse."""
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]
