
"""
=============================================================================
PATHFINDING AGENT - TIER 1: ENVIRONMENT LAYER
=============================================================================
Represents the Discrete, Static, and Fully Observable state space.
Defines the grid topology, obstacle placement, and valid state transitions.

Architecture Role:  Data Preparation & World Modeling
AI Concept:         State-Space Graph (each cell = a node in a DAG)
=============================================================================
"""

from typing import List, Tuple, Optional, Set


class GridEnvironment:
    """
    Encapsulates the world the agent perceives and acts within.

    Properties
    ----------
    - Discrete:         Finite set of (row, col) cell positions
    - Static:           Obstacles do not change during search
    - Fully Observable: Agent has complete knowledge of the grid
    """

    # Movement directions: (Δrow, Δcol) — 4-connected grid (no diagonals)
    ACTIONS = {
        "UP":    (-1,  0),
        "DOWN":  ( 1,  0),
        "LEFT":  ( 0, -1),
        "RIGHT": ( 0,  1),
    }
    STEP_COST = 1  # Uniform cost per transition (admissibility guarantee)

    def __init__(self, rows: int, cols: int, obstacles: Set[Tuple[int, int]]):
        """
        Parameters
        ----------
        rows      : number of grid rows
        cols      : number of grid columns
        obstacles : set of (row, col) cells that are impassable walls
        """
        self.rows      = rows
        self.cols      = cols
        self.obstacles = frozenset(obstacles)  # immutable — environment is static

        # Metrics tracked for the final report
        self._transition_calls = 0

    # ------------------------------------------------------------------
    # Perception Interface  (what the Agent "sees")
    # ------------------------------------------------------------------

    def is_valid(self, state: Tuple[int, int]) -> bool:
        """Return True if a cell is within bounds AND not an obstacle."""
        r, c = state
        return (0 <= r < self.rows) and (0 <= c < self.cols) and (state not in self.obstacles)

    def is_obstacle(self, state: Tuple[int, int]) -> bool:
        return state in self.obstacles

    # ------------------------------------------------------------------
    # Transition Model  (successor function)
    # ------------------------------------------------------------------

    def get_successors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], str, int]]:
        """
        Expand a node: return all reachable (next_state, action, cost) triples.

        This is the Agent–Environment interaction boundary:
          Perception → Action → New State
        """
        self._transition_calls += 1
        successors = []
        r, c = state
        for action, (dr, dc) in self.ACTIONS.items():
            next_state = (r + dr, c + dc)
            if self.is_valid(next_state):
                successors.append((next_state, action, self.STEP_COST))
        return successors

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_grid_snapshot(self) -> List[List[str]]:
        """Return a 2-D list of cell types for visualization."""
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for (r, c) in self.obstacles:
            grid[r][c] = "#"
        return grid

    def __repr__(self) -> str:
        return f"GridEnvironment({self.rows}×{self.cols}, {len(self.obstacles)} obstacles)"


# ------------------------------------------------------------------
# Factory helpers — pre-built scenario maps
# ------------------------------------------------------------------

def make_scenario(name: str = "default") -> Tuple[GridEnvironment, Tuple, Tuple]:
    """
    Return (env, start, goal) for a named scenario.
    Useful for reproducible benchmark runs.
    """
    scenarios = {
        "default": {
            "rows": 10, "cols": 10,
            "obstacles": {
                (1,1),(1,2),(1,3),(1,4),(1,5),
                (3,3),(3,4),(3,5),(3,6),(3,7),
                (5,0),(5,1),(5,2),(5,3),
                (5,6),(5,7),(5,8),(5,9),
                (7,2),(7,3),(7,4),(7,5),(7,6),
                (2,8),(4,1),(6,5),(8,8),
            },
            "start": (0, 0),
            "goal":  (9, 9),
        },
        "maze": {
            "rows": 12, "cols": 12,
            "obstacles": {
                (0,2),(1,2),(2,2),(3,2),(4,2),
                (2,4),(2,5),(2,6),(2,7),
                (4,4),(5,4),(6,4),(7,4),(8,4),
                (6,0),(6,1),(6,2),(6,3),
                (6,6),(6,7),(6,8),(6,9),(6,10),
                (8,6),(9,6),(10,6),(10,7),(10,8),
                (4,8),(4,9),(4,10),(4,11),
                (8,2),(9,2),(10,2),(11,2),
            },
            "start": (0, 0),
            "goal":  (11, 11),
        },
        "dead_end": {
            "rows": 8, "cols": 8,
            "obstacles": {
                # corridor that leads to a dead end
                (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),
                (4,1),(4,2),(4,3),(4,4),(4,5),(4,6),
                (3,6),
                # second passage
                (1,3),(1,4),(1,5),
            },
            "start": (0, 0),
            "goal":  (7, 7),
        },
    }
    cfg = scenarios[name]
    env = GridEnvironment(cfg["rows"], cfg["cols"], cfg["obstacles"])
    return env, cfg["start"], cfg["goal"]
