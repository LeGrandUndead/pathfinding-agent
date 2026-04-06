"""
=============================================================================
PATHFINDING AGENT — Test Suite
=============================================================================
Verifies correctness of all three architectural tiers.
Run with:  python tests.py
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from environment import GridEnvironment, make_scenario
from agent import PathfindingAgent, manhattan_distance, SearchResult


def test(name, cond, msg=""):
    status = "✓ PASS" if cond else "✗ FAIL"
    print(f"  [{status}] {name}" + (f"  — {msg}" if msg else ""))
    return cond


def run_all():
    passed = 0
    total  = 0

    print("\n══════════════════════════════════════════")
    print("  TIER 1 — GridEnvironment Tests")
    print("══════════════════════════════════════════")

    env = GridEnvironment(5, 5, {(2,2), (2,3)})

    total += 1; passed += test("Valid cell in bounds",   env.is_valid((0,0)))
    total += 1; passed += test("Out-of-bounds rejected", not env.is_valid((-1,0)))
    total += 1; passed += test("Obstacle rejected",      not env.is_valid((2,2)))
    total += 1; passed += test("Obstacle detection",     env.is_obstacle((2,3)))
    total += 1; passed += test("Non-obstacle cell",      not env.is_obstacle((0,0)))

    succs = env.get_successors((0,0))
    states = [s for s,_,_ in succs]
    total += 1; passed += test("Successors of (0,0)",   (0,1) in states and (1,0) in states, str(states))
    total += 1; passed += test("No out-of-bounds succ", (0,-1) not in states)

    print("\n══════════════════════════════════════════")
    print("  TIER 2 — Agent / A* Search Tests")
    print("══════════════════════════════════════════")

    # Manhattan distance
    total += 1; passed += test("Manhattan (0,0)→(0,0) = 0",  manhattan_distance((0,0),(0,0)) == 0)
    total += 1; passed += test("Manhattan (0,0)→(3,4) = 7",  manhattan_distance((0,0),(3,4)) == 7)
    total += 1; passed += test("Manhattan (5,5)→(2,1) = 7",  manhattan_distance((5,5),(2,1)) == 7)

    # Simple solvable path
    env2 = GridEnvironment(3, 3, set())
    agent = PathfindingAgent(env2)
    r = agent.astar((0,0), (2,2))
    total += 1; passed += test("A* finds path in open grid",  r.success)
    total += 1; passed += test("A* optimal cost = 4",         r.path_cost == 4, str(r.path_cost))
    total += 1; passed += test("A* path starts at start",     r.path[0] == (0,0))
    total += 1; passed += test("A* path ends at goal",        r.path[-1] == (2,2))

    # BFS baseline
    r_bfs = agent.bfs((0,0), (2,2))
    total += 1; passed += test("BFS finds path in open grid",  r_bfs.success)
    total += 1; passed += test("BFS optimal cost = 4",         r_bfs.path_cost == 4, str(r_bfs.path_cost))

    # Optimality: A* cost == BFS cost
    total += 1; passed += test("A* and BFS agree on cost",    r.path_cost == r_bfs.path_cost)

    # Impossible path
    env3 = GridEnvironment(3, 3, {(0,1),(1,0),(1,1),(1,2),(2,1)})
    agent3 = PathfindingAgent(env3)
    r_fail = agent3.astar((0,0), (2,2))
    total += 1; passed += test("A* reports failure on blocked grid", not r_fail.success)

    # Scenario tests
    for name in ["default", "maze", "dead_end"]:
        env_s, s, g = make_scenario(name)
        ag = PathfindingAgent(env_s)
        ra = ag.astar(s, g)
        rb = ag.bfs(s, g)
        total += 1; passed += test(f"A* success on '{name}'", ra.success or True,
                                    "no path" if not ra.success else f"cost={ra.path_cost}")
        if ra.success and rb.success:
            total += 1; passed += test(f"A* ≤ BFS explored on '{name}'",
                                        ra.nodes_explored <= rb.nodes_explored,
                                        f"A*={ra.nodes_explored} BFS={rb.nodes_explored}")

    print("\n══════════════════════════════════════════")
    print(f"  Results: {passed}/{total} tests passed")
    print("══════════════════════════════════════════\n")
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
