"""
=============================================================================
PATHFINDING AGENT — Test Suite
=============================================================================
Verifies correctness of all three architectural tiers.
Run with:  python tests.py
=============================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from environment import GridEnvironment, make_scenario
from agent import PathfindingAgent, manhattan_distance, euclidean_distance


def test(name, cond, msg=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {msg}" if msg else ""))
    return cond


def run_all():
    passed = 0; total = 0

    print("\n" + "="*46)
    print("  TIER 1 -- GridEnvironment Tests")
    print("="*46)

    env = GridEnvironment(5, 5, {(2,2),(2,3)})
    total+=1; passed+=test("Valid cell in bounds",     env.is_valid((0,0)))
    total+=1; passed+=test("Out-of-bounds rejected",   not env.is_valid((-1,0)))
    total+=1; passed+=test("Obstacle rejected",        not env.is_valid((2,2)))
    total+=1; passed+=test("is_obstacle True",         env.is_obstacle((2,3)))
    total+=1; passed+=test("is_obstacle False",        not env.is_obstacle((0,0)))
    succs = env.get_successors((0,0))
    states = [s for s,_,_ in succs]
    total+=1; passed+=test("Successors of (0,0) include (0,1) & (1,0)",
                            (0,1) in states and (1,0) in states)
    total+=1; passed+=test("Diagonal (1,1) in successors", (1,1) in states)
    total+=1; passed+=test("No out-of-bounds successor",   (-1,-1) not in states)

    print("\n" + "="*46)
    print("  TIER 2 -- Heuristic Tests")
    print("="*46)
    total+=1; passed+=test("Manhattan (0,0)->(0,0) = 0",  manhattan_distance((0,0),(0,0))==0)
    total+=1; passed+=test("Manhattan (0,0)->(3,4) = 7",  manhattan_distance((0,0),(3,4))==7)
    total+=1; passed+=test("Euclidean (0,0)->(3,4) ~= 5", abs(euclidean_distance((0,0),(3,4))-5.0)<0.01)

    print("\n" + "="*46)
    print("  TIER 2 -- A* Search Tests")
    print("="*46)
    env2  = GridEnvironment(3, 3, set())
    agent = PathfindingAgent(env2)
    r = agent.astar((0,0),(2,2))
    total+=1; passed+=test("A* finds path in open grid",  r.success)
    total+=1; passed+=test("A* path starts at start",     r.path[0]==(0,0))
    total+=1; passed+=test("A* path ends at goal",        r.path[-1]==(2,2))
    total+=1; passed+=test("A* path cost > 0",            r.path_cost > 0)

    # Impossibility
    env3   = GridEnvironment(3, 3, {(0,1),(1,0),(1,1),(1,2),(2,1)})
    agent3 = PathfindingAgent(env3)
    total+=1; passed+=test("A* fails on blocked grid",    not agent3.astar((0,0),(2,2)).success)

    print("\n" + "="*46)
    print("  TIER 2 -- BFS Tests")
    print("="*46)
    rb = agent.bfs((0,0),(2,2))
    total+=1; passed+=test("BFS finds path in open grid", rb.success)
    total+=1; passed+=test("BFS path starts at start",    rb.path[0]==(0,0))
    total+=1; passed+=test("BFS path ends at goal",       rb.path[-1]==(2,2))

    print("\n" + "="*46)
    print("  TIER 2 -- Scenario + Efficiency Tests")
    print("="*46)
    for sname in ["default", "hard_maze", "dead_end", "maze"]:
        e, s, g = make_scenario(sname)
        ag = PathfindingAgent(e)
        ra = ag.astar(s, g)
        rb = ag.bfs(s, g)
        total+=1; passed+=test(f"A* success on '{sname}'", ra.success,
                                f"cost={ra.path_cost}" if ra.success else "no path")
        if ra.success and rb.success:
            total+=1; passed+=test(
                f"A* explores fewer/equal nodes than BFS on '{sname}'",
                ra.nodes_explored <= rb.nodes_explored,
                f"A*={ra.nodes_explored} BFS={rb.nodes_explored}"
            )

    print("\n" + "="*46)
    print(f"  Results: {passed}/{total} tests passed")
    print("="*46 + "\n")
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
