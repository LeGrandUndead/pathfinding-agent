"""
=============================================================================
PATHFINDING AGENT - TIER 2: AGENT / LOGIC LAYER
=============================================================================
Contains the search algorithms entirely decoupled from rendering.

Implements:
  • A* Search   — Informed search with f(n) = g(n) + h(n)
  • BFS         — Uninformed baseline for comparison

Data Structures:
  • OPEN LIST   → heapq-based priority queue  (O(log n) insert/pop)
  • CLOSED SET  → Python set hash table       (O(1) membership test)

Metrics tracked (for final report):
  • nodes_explored   — space complexity proxy
  • path_cost        — total g(n) of solution
  • path_length      — number of steps
=============================================================================
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

from environment import GridEnvironment


# ──────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Unified result object returned by every search algorithm."""
    algorithm:        str
    success:          bool
    path:             List[Tuple[int, int]]   # ordered list of (row,col) cells
    actions:          List[str]               # action sequence
    path_cost:        int                     # g(goal)  — time complexity proxy
    nodes_explored:   int                     # |CLOSED|  — space complexity proxy
    nodes_generated:  int                     # |OPEN| enqueues
    explored_order:   List[Tuple[int, int]]   # visit order (for animation)
    open_snapshots:   List[Set]               # frontier at each step (animation)
    failure_reason:   str = ""

    def summary(self) -> str:
        status = "SUCCESS" if self.success else f"FAILURE ({self.failure_reason})"
        lines = [
            f"{'─'*50}",
            f"  Algorithm      : {self.algorithm}",
            f"  Status         : {status}",
            f"  Path length    : {len(self.path)} cells",
            f"  Path cost      : {self.path_cost}",
            f"  Nodes explored : {self.nodes_explored}",
            f"  Nodes generated: {self.nodes_generated}",
            f"{'─'*50}",
        ]
        return "\n".join(lines)


@dataclass(order=True)
class _PQNode:
    """Priority queue entry — only f_cost drives ordering."""
    f_cost: float
    g_cost: int        = field(compare=False)
    state:  Any        = field(compare=False)
    parent: Any        = field(compare=False)
    action: str        = field(compare=False)


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic functions
# ──────────────────────────────────────────────────────────────────────────────

def manhattan_distance(state: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    h(n) = |r1 - r2| + |c1 - c2|

    Admissibility proof:
      On a 4-connected grid with uniform step cost = 1, the real cost between
      any two cells is ≥ their Manhattan distance.  Therefore h(n) never
      overestimates → A* is guaranteed to find the optimal path.
    """
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


# ──────────────────────────────────────────────────────────────────────────────
# Tier-2 Agent class
# ──────────────────────────────────────────────────────────────────────────────

class PathfindingAgent:
    """
    Stateless search agent.  Both A* and BFS operate over the same
    GridEnvironment via the get_successors() interface — clean decoupling.

    Agent–Environment interaction cycle
    ────────────────────────────────────
      1. PERCEPTION  : agent calls env.get_successors(current_state)
      2. EVALUATION  : agent scores each successor with f(n) = g(n) + h(n)
      3. ACTION      : agent selects lowest-f successor from OPEN LIST
      4. STATE       : new (row, col) becomes current state
    """

    def __init__(self, env: GridEnvironment):
        self.env = env

    # ──────────────────────────────────────────────────────────────────────
    # A* Search
    # ──────────────────────────────────────────────────────────────────────

    def astar(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
    ) -> SearchResult:
        """
        Informed A* Search
        ──────────────────
        OPEN  LIST: min-heap keyed on f(n) = g(n) + h(n)
        CLOSED SET: hash set of visited states

        Completeness : YES (finite graph, no repeated states)
        Optimality   : YES (admissible + consistent heuristic)
        Time  O(b^d)  where b = branching factor, d = solution depth
        Space O(b^d)  all nodes kept in memory
        """

        # ── Initialise ──────────────────────────────────────────────────
        h_start  = manhattan_distance(start, goal)
        root     = _PQNode(f_cost=h_start, g_cost=0, state=start,
                           parent=None, action="START")

        open_heap: List[_PQNode]               = [root]
        # g_best[state] = lowest g-cost seen so far (lazy-deletion optimisation)
        g_best:    Dict[Tuple, int]            = {start: 0}
        closed:    Set[Tuple[int, int]]        = set()
        parent_map: Dict[Tuple, Optional[_PQNode]] = {start: None}

        # Animation data
        explored_order: List[Tuple[int, int]] = []
        open_snapshots: List[Set]             = []
        nodes_generated = 1

        # ── Generic Search Loop ─────────────────────────────────────────
        while open_heap:
            # SELECT: pop lowest f(n) — O(log n)
            current = heapq.heappop(open_heap)

            # Lazy-deletion: skip if we already found a cheaper path
            if current.state in closed:
                continue

            # VISIT: move to CLOSED (hash table)
            closed.add(current.state)
            explored_order.append(current.state)
            open_snapshots.append({n.state for n in open_heap})

            # GOAL TEST
            if current.state == goal:
                path, actions = self._reconstruct_path(current)
                return SearchResult(
                    algorithm       = "A* (Manhattan Distance)",
                    success         = True,
                    path            = path,
                    actions         = actions,
                    path_cost       = current.g_cost,
                    nodes_explored  = len(closed),
                    nodes_generated = nodes_generated,
                    explored_order  = explored_order,
                    open_snapshots  = open_snapshots,
                )

            # EXPAND: generate successors
            for next_state, action, step_cost in self.env.get_successors(current.state):
                if next_state in closed:
                    continue  # already on optimal path

                tentative_g = current.g_cost + step_cost

                # Only enqueue if this is the best known path to next_state
                if tentative_g < g_best.get(next_state, float("inf")):
                    g_best[next_state] = tentative_g
                    h = manhattan_distance(next_state, goal)
                    f = tentative_g + h
                    child = _PQNode(f_cost=f, g_cost=tentative_g,
                                    state=next_state,
                                    parent=current, action=action)
                    heapq.heappush(open_heap, child)
                    nodes_generated += 1

        # OPEN list exhausted — no path exists (dead-end / disconnected graph)
        return SearchResult(
            algorithm       = "A* (Manhattan Distance)",
            success         = False,
            path=[], actions=[],
            path_cost       = -1,
            nodes_explored  = len(closed),
            nodes_generated = nodes_generated,
            explored_order  = explored_order,
            open_snapshots  = open_snapshots,
            failure_reason  = "No path exists (goal unreachable)",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Breadth-First Search (uninformed baseline)
    # ──────────────────────────────────────────────────────────────────────

    def bfs(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
    ) -> SearchResult:
        """
        Uninformed BFS — explores by level (FIFO queue).

        Completeness : YES
        Optimality   : YES (uniform costs)
        Time  O(b^d)
        Space O(b^d)  ← often much larger constant than A*
        """
        queue:    deque  = deque()
        visited:  Set    = set()
        parent:   Dict   = {start: None}
        action_map: Dict = {start: "START"}

        queue.append((start, 0))
        visited.add(start)

        explored_order: List[Tuple[int, int]] = []
        open_snapshots: List[Set]             = []
        nodes_generated = 1
        g_cost = 0

        while queue:
            state, g_cost = queue.popleft()
            explored_order.append(state)
            open_snapshots.append({s for s, _ in queue})

            if state == goal:
                path, actions = self._reconstruct_bfs_path(state, parent, action_map)
                return SearchResult(
                    algorithm       = "BFS (Uninformed)",
                    success         = True,
                    path            = path,
                    actions         = actions,
                    path_cost       = len(path) - 1,
                    nodes_explored  = len(visited),
                    nodes_generated = nodes_generated,
                    explored_order  = explored_order,
                    open_snapshots  = open_snapshots,
                )

            for next_state, action, cost in self.env.get_successors(state):
                if next_state not in visited:
                    visited.add(next_state)
                    parent[next_state]     = state
                    action_map[next_state] = action
                    queue.append((next_state, g_cost + cost))
                    nodes_generated += 1

        return SearchResult(
            algorithm       = "BFS (Uninformed)",
            success         = False,
            path=[], actions=[],
            path_cost       = -1,
            nodes_explored  = len(visited),
            nodes_generated = nodes_generated,
            explored_order  = explored_order,
            open_snapshots  = open_snapshots,
            failure_reason  = "No path exists",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Path reconstruction helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reconstruct_path(node: _PQNode) -> Tuple[List, List]:
        """Walk parent pointers from goal back to start, then reverse."""
        path, actions = [], []
        current = node
        while current is not None:
            path.append(current.state)
            if current.action != "START":
                actions.append(current.action)
            current = current.parent
        path.reverse()
        actions.reverse()
        return path, actions

    @staticmethod
    def _reconstruct_bfs_path(goal, parent, action_map) -> Tuple[List, List]:
        path, actions = [], []
        state = goal
        while state is not None:
            path.append(state)
            act = action_map.get(state, "")
            if act and act != "START":
                actions.append(act)
            state = parent.get(state)
        path.reverse()
        actions.reverse()
        return path, actions
