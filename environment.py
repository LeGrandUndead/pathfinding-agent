"""
=============================================================================
TIER 1 — Environment Layer
=============================================================================
Represents a Discrete, Static, and Fully Observable grid world.
Each (row, col) cell is a node in the state-space graph.
Obstacles define impassable regions; valid transitions define the edges.
=============================================================================
"""

from typing import List, Tuple, Set


class GridEnvironment:
    """
    Models the agent's world as a 2-D grid (state-space graph).

    Properties
    ----------
    Discrete         : finite set of (row, col) cells
    Static           : obstacles do not change during the search
    Fully Observable : agent has complete world knowledge at all times

    Transition model
    ----------------
    Orthogonal moves : cost 1.0   (Up, Down, Left, Right)
    Diagonal   moves : cost 1.414 (NW, NE, SW, SE — Pythagorean distance)
    """

    ORTHO = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    DIAG  = {"NW": (-1,-1), "NE": (-1, 1), "SW": ( 1,-1), "SE": ( 1, 1)}

    def __init__(self, rows: int, cols: int, obstacles: Set[Tuple[int, int]]):
        self.rows = rows
        self.cols = cols
        # FIX: freeze the set so the environment stays truly static
        self.obstacles = frozenset(obstacles)

    # ── Perception interface ────────────────────────────────────────────

    def is_valid(self, state: Tuple[int, int]) -> bool:
        """Return True if cell is in-bounds and not an obstacle."""
        r, c = state
        return (0 <= r < self.rows) and (0 <= c < self.cols) and (state not in self.obstacles)

    def is_obstacle(self, state: Tuple[int, int]) -> bool:
        """Return True if this cell is an obstacle (wall)."""
        return state in self.obstacles

    # ── Transition model ────────────────────────────────────────────────

    def get_successors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], str, float]]:
        """
        Expand a node: return all (next_state, action, cost) triples.
        This is the Agent-Environment boundary:
            Perception -> Action -> New State
        """
        r, c = state
        successors = []
        for action, (dr, dc) in self.ORTHO.items():
            nxt = (r + dr, c + dc)
            if self.is_valid(nxt):
                successors.append((nxt, action, 1.0))
        for action, (dr, dc) in self.DIAG.items():
            nxt = (r + dr, c + dc)
            if self.is_valid(nxt):
                successors.append((nxt, action, 1.414))
        return successors

    def __repr__(self) -> str:
        return f"GridEnvironment({self.rows}x{self.cols}, {len(self.obstacles)} obstacles)"


# ── Scenario factory ────────────────────────────────────────────────────────

def make_scenario(name: str = "default"):
    """
    Return (env, start, goal) for a named benchmark scenario.

    Scenarios
    ---------
    "default"   : simple 10x10 with a few obstacle clusters
    "hard_maze" : 25x25 with three horizontal walls (S-shaped corridors)
    "dead_end"  : 5x5 that forces backtracking from a false path
    "maze"      : 12x12 multi-corridor maze
    """
    if name == "hard_maze":
        obs = set()
        for i in range(20):   obs.add((6,  i))
        for i in range(5, 25): obs.add((12, i))
        for i in range(20):   obs.add((18, i))
        return GridEnvironment(25, 25, obs), (0, 0), (24, 24)

    elif name == "dead_end":
        # Two horizontal walls with a capped corridor -- forces the agent to
        # abandon the first corridor it enters and navigate around the barriers.
        obs = {
            (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
            (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),
            (3,6),
        }
        return GridEnvironment(8, 8, obs), (0, 0), (7, 7)

    elif name == "maze":
        obs = {
            (0,2),(1,2),(2,2),(3,2),(4,2),
            (2,4),(2,5),(2,6),(2,7),
            (4,4),(5,4),(6,4),(7,4),(8,4),
            (6,0),(6,1),(6,2),(6,3),
            (6,6),(6,7),(6,8),(6,9),(6,10),
            (8,6),(9,6),(10,6),(10,7),(10,8),
            (4,8),(4,9),(4,10),(4,11),
            (8,2),(9,2),(10,2),(11,2),
        }
        return GridEnvironment(12, 12, obs), (0, 0), (11, 11)

    else:  # "default"
        obs = {(1,1),(1,2),(1,3),(3,3),(3,2),(3,1)}
        return GridEnvironment(10, 10, obs), (0, 0), (9, 9)
