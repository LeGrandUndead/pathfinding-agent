"""
=============================================================================
PATHFINDING AGENT - TIER 3: INTERFACE / VISUALIZATION LAYER
=============================================================================
Renders the grid and animates the search progress in the terminal.
Completely decoupled from Tiers 1 & 2 — depends only on their public APIs.

Also contains the Benchmark harness that generates comparison data
for the A* vs BFS PowerPoint report.
=============================================================================
"""

import time
import sys
import os
from typing import List, Tuple, Optional

from environment import GridEnvironment, make_scenario
from agent import PathfindingAgent, SearchResult


# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour codes
# ──────────────────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BG_DARK = "\033[40m"


# Cell display legend
CELL = {
    "empty":    f"{C.GRAY}·{C.RESET}",
    "wall":     f"{C.RED}█{C.RESET}",
    "start":    f"{C.GREEN}{C.BOLD}S{C.RESET}",
    "goal":     f"{C.MAGENTA}{C.BOLD}G{C.RESET}",
    "explored": f"{C.BLUE}○{C.RESET}",
    "frontier": f"{C.CYAN}◌{C.RESET}",
    "path":     f"{C.YELLOW}{C.BOLD}★{C.RESET}",
    "agent":    f"{C.GREEN}{C.BOLD}@{C.RESET}",
}


# ──────────────────────────────────────────────────────────────────────────────
# Core renderer
# ──────────────────────────────────────────────────────────────────────────────

class GridRenderer:
    """Renders a GridEnvironment to the terminal with optional overlays."""

    def __init__(self, env: GridEnvironment,
                 start: Tuple[int, int],
                 goal:  Tuple[int, int]):
        self.env   = env
        self.start = start
        self.goal  = goal

    def render(
        self,
        explored:  Optional[set] = None,
        frontier:  Optional[set] = None,
        path:      Optional[List[Tuple[int, int]]] = None,
        agent_pos: Optional[Tuple[int, int]] = None,
        title: str = "",
        stats: str = "",
    ) -> None:
        """Print the current grid state to stdout."""
        explored = explored or set()
        frontier = frontier or set()
        path_set = set(path) if path else set()

        # Top border
        width = self.env.cols * 2 + 3
        if title:
            print(f"\n{C.BOLD}{C.WHITE}  {title}{C.RESET}")
        print("  ┌" + "─" * (self.env.cols * 2) + "┐")

        for r in range(self.env.rows):
            row_str = f"  │"
            for c in range(self.env.cols):
                cell = (r, c)
                if cell == agent_pos:
                    ch = CELL["agent"]
                elif cell == self.start:
                    ch = CELL["start"]
                elif cell == self.goal:
                    ch = CELL["goal"]
                elif cell in path_set:
                    ch = CELL["path"]
                elif self.env.is_obstacle(cell):
                    ch = CELL["wall"]
                elif cell in frontier:
                    ch = CELL["frontier"]
                elif cell in explored:
                    ch = CELL["explored"]
                else:
                    ch = CELL["empty"]
                row_str += ch + " "
            row_str += "│"
            print(row_str)

        print("  └" + "─" * (self.env.cols * 2) + "┘")

        # Legend
        legend = (
            f"  {CELL['start']}=Start  {CELL['goal']}=Goal  "
            f"{CELL['wall']}=Wall  {CELL['explored']}=Explored  "
            f"{CELL['frontier']}=Frontier  {CELL['path']}=Path"
        )
        print(legend)
        if stats:
            print(f"  {C.CYAN}{stats}{C.RESET}")

    def animate_search(
        self,
        result: SearchResult,
        step_delay: float = 0.05,
        final_delay: float = 1.5,
        skip_animation: bool = False,
    ) -> None:
        """
        Stream the exploration animation frame by frame.
        Each frame shows: explored cells, frontier, current agent position.
        """
        if skip_animation:
            # Jump straight to final result
            self.render(
                explored  = set(result.explored_order),
                path      = result.path,
                title     = f"{'✓' if result.success else '✗'} {result.algorithm} — FINAL",
                stats     = f"Explored: {result.nodes_explored}  |  Path cost: {result.path_cost}",
            )
            return

        print(f"\n{C.BOLD}▶ Animating {result.algorithm}...{C.RESET}")
        print(f"  (Press Ctrl+C to skip to result)\n")

        explored_so_far = set()
        try:
            for i, cell in enumerate(result.explored_order):
                explored_so_far.add(cell)
                frontier = result.open_snapshots[i] if i < len(result.open_snapshots) else set()

                # Clear previous frame
                if i > 0:
                    lines = self.env.rows + 4
                    print(f"\033[{lines}A\033[J", end="")

                self.render(
                    explored  = explored_so_far,
                    frontier  = frontier,
                    agent_pos = cell,
                    title     = f"▶ {result.algorithm} — Step {i+1}/{len(result.explored_order)}",
                    stats     = f"Exploring: {cell}  |  Frontier size: {len(frontier)}",
                )
                time.sleep(step_delay)

        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}  Animation skipped.{C.RESET}")

        # Final frame — show solution path
        time.sleep(0.3)
        lines = self.env.rows + 4
        print(f"\033[{lines}A\033[J", end="")
        status_icon = "✓" if result.success else "✗"
        self.render(
            explored  = set(result.explored_order),
            path      = result.path,
            title     = f"{status_icon} {result.algorithm} — RESULT",
            stats     = (
                f"Explored: {result.nodes_explored}  |  "
                f"Path cost: {result.path_cost}  |  "
                f"Path length: {len(result.path)}"
            ),
        )
        time.sleep(final_delay)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark harness (generates PowerPoint data)
# ──────────────────────────────────────────────────────────────────────────────

class Benchmark:
    """
    Run A* and BFS on multiple scenarios and print a comparison table.
    Data directly usable in the A* vs BFS PowerPoint slide.
    """

    SCENARIOS = ["default", "maze", "dead_end"]

    @staticmethod
    def run() -> None:
        print(f"\n{'═'*70}")
        print(f"  {C.BOLD}{C.WHITE}BENCHMARK: A* vs BFS — All Scenarios{C.RESET}")
        print(f"{'═'*70}")
        print(f"  {'Scenario':<12} {'Algorithm':<26} {'Explored':>9} {'Generated':>10} {'Cost':>6} {'Time(ms)':>9}")
        print(f"  {'─'*12} {'─'*26} {'─'*9} {'─'*10} {'─'*6} {'─'*9}")

        all_results = []

        for scenario_name in Benchmark.SCENARIOS:
            env, start, goal = make_scenario(scenario_name)
            agent = PathfindingAgent(env)

            for algo_name, search_fn in [("A* (Manhattan)", agent.astar),
                                          ("BFS (Uninformed)", agent.bfs)]:
                t0 = time.perf_counter()
                result = search_fn(start, goal)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                status = "✓" if result.success else "✗ (no path)"
                cost_str = str(result.path_cost) if result.success else "N/A"

                print(
                    f"  {scenario_name:<12} {algo_name:<26} "
                    f"{result.nodes_explored:>9} {result.nodes_generated:>10} "
                    f"{cost_str:>6} {elapsed_ms:>8.2f}ms"
                )
                all_results.append((scenario_name, result, elapsed_ms))

            print(f"  {'·'*70}")

        # Summary insights
        print(f"\n  {C.BOLD}KEY INSIGHTS FOR PRESENTATION:{C.RESET}")
        print(f"  • A* explores far fewer nodes than BFS on open maps.")
        print(f"  • Both guarantee optimal path cost (admissible heuristic).")
        print(f"  • On tight mazes, advantage of h(n) shrinks — BFS converges.")
        print(f"  • Dead-end scenario shows backtracking via CLOSED set avoidance.")
        print(f"{'═'*70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{C.BOLD}{C.CYAN}")
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║    PATHFINDING AGENT — Semester Project   ║")
    print("  ║    A* Search  vs  BFS  |  3-Tier Arch.   ║")
    print("  ╚═══════════════════════════════════════════╝")
    print(C.RESET)

    # ── Tier 1: Environment ──────────────────────────────────────────────
    env, start, goal = make_scenario("default")
    print(f"  Environment : {env}")
    print(f"  Start       : {start}   Goal: {goal}")

    # ── Tier 2: Agent ────────────────────────────────────────────────────
    agent    = PathfindingAgent(env)
    renderer = GridRenderer(env, start, goal)

    # Show initial grid
    renderer.render(title="Initial Grid State")

    # ── A* Search ────────────────────────────────────────────────────────
    print(f"\n{C.BOLD}Running A* Search...{C.RESET}")
    astar_result = agent.astar(start, goal)
    renderer.animate_search(astar_result, step_delay=0.04, skip_animation=False)
    print(astar_result.summary())

    # ── BFS Search ───────────────────────────────────────────────────────
    print(f"\n{C.BOLD}Running BFS (Uninformed baseline)...{C.RESET}")
    bfs_result = agent.bfs(start, goal)
    renderer.animate_search(bfs_result, step_delay=0.03, skip_animation=False)
    print(bfs_result.summary())

    # ── Head-to-head comparison ──────────────────────────────────────────
    print(f"\n{C.BOLD}{C.WHITE}{'─'*50}")
    print(f"  HEAD-TO-HEAD COMPARISON (same grid, same goal)")
    print(f"{'─'*50}{C.RESET}")
    print(f"  {'Metric':<22} {'A*':>12} {'BFS':>12}")
    print(f"  {'─'*22} {'─'*12} {'─'*12}")
    rows = [
        ("Nodes Explored",   astar_result.nodes_explored,   bfs_result.nodes_explored),
        ("Nodes Generated",  astar_result.nodes_generated,  bfs_result.nodes_generated),
        ("Path Cost",        astar_result.path_cost,        bfs_result.path_cost),
        ("Path Length",      len(astar_result.path),        len(bfs_result.path)),
    ]
    for label, av, bv in rows:
        winner = f"{C.GREEN}◀{C.RESET}" if av < bv else ("" if av == bv else f"{C.YELLOW}▶{C.RESET}")
        print(f"  {label:<22} {str(av):>12} {str(bv):>12}  {winner}")

    efficiency = (1 - astar_result.nodes_explored / max(bfs_result.nodes_explored, 1)) * 100
    print(f"\n  {C.GREEN}A* efficiency gain: {efficiency:.1f}% fewer nodes explored{C.RESET}")

    # ── Benchmark across all scenarios ───────────────────────────────────
    print(f"\n{C.BOLD}Running full benchmark suite...{C.RESET}")
    Benchmark.run()

    # ── Dead-end scenario analysis ───────────────────────────────────────
    print(f"{C.BOLD}Dead-End Scenario Analysis{C.RESET}")
    env_de, s_de, g_de = make_scenario("dead_end")
    agent_de  = PathfindingAgent(env_de)
    renderer_de = GridRenderer(env_de, s_de, g_de)
    r_de = agent_de.astar(s_de, g_de)
    renderer_de.animate_search(r_de, skip_animation=True)
    print(f"  Dead-end handling: agent explored {r_de.nodes_explored} cells before finding path.")
    print(f"  Backtracking is implicit — the CLOSED SET prevents revisiting failed states.\n")


if __name__ == "__main__":
    main()
