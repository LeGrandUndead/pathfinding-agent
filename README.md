# Pathfinding Agent: A* vs BFS Comparison

**CMI Semester Project — AI Application**

---

## 1. Project Overview

This project implements a **Rational Agent** capable of navigating a discrete, static grid environment. It compares two fundamental search strategies:

- **A\* Search (Informed)** — uses the Euclidean distance heuristic to efficiently compute optimal paths on 8-connected grids (orthogonal + diagonal moves).
- **Breadth-First Search (BFS) (Uninformed)** — serves as a baseline to highlight the performance advantages of informed search.

All four benchmark scenarios confirm that A* returns the same optimal path as BFS while exploring significantly fewer nodes (−26% to −78% depending on the scenario).

---

## 2. Architecture (3-Tier Design)

The application is structured into three modular layers, each with a single responsibility:

| Tier | File | Responsibility |
|------|------|----------------|
| 1 | `environment.py` | World model — grid, obstacles, transition model |
| 2 | `agent.py` | Search logic — A* and BFS algorithms |
| 3 | `visualization.py` | GUI — Tkinter interface, animation, chart export |

### Tier 1: Environment (`environment.py`)

- **State Space:** grid represented as `(row, col)` coordinates
- **Transition Model:**
  - Orthogonal moves: Up, Down, Left, Right — cost `1.0`
  - Diagonal moves: NW, NE, SW, SE — cost `1.414` (√2)
  - Obstacles stored as `frozenset` — environment is immutable after construction
- **Public interface:** `is_valid(state)`, `is_obstacle(state)`, `get_successors(state)`

### Tier 2: Agent Logic (`agent.py`)

- **A\* Search** — evaluation function `f(n) = g(n) + h(n)`
  - OPEN LIST: `heapq` min-heap, O(log n) insert/pop
  - CLOSED SET: Python `set`, O(1) membership test
  - Lazy deletion to handle stale heap entries
  - Heuristic: Euclidean distance (admissible on 8-connected grids)
- **BFS** — FIFO `deque`, no heuristic, `f(n) = g(n)` only
- Both algorithms return a `SearchResult` dataclass with path, nodes explored, path cost, and exploration order

### Tier 3: Visualization (`visualization.py`)

- **Interactive Tkinter GUI** with scenario selector and animation
- **Side-by-side comparison panel** — run A* and BFS simultaneously
- **Speed slider** — control animation from 5 ms to 120 ms per frame
- **Export Benchmark PNG** — generates a full multi-panel chart summary (requires `matplotlib`)
- **Live statistics** panel — algorithm, nodes explored, path cost, time

---

## 3. Key AI Concepts Applied

### PEAS Framework

| Component | Definition |
|-----------|------------|
| **Performance** | Minimum path cost from start to goal |
| **Environment** | Discrete 2-D grid — static, deterministic, fully observable |
| **Actuators** | 8 movement directions (4 orthogonal + 4 diagonal) |
| **Sensors** | `is_valid(state)` — full world map known at all times |

### Heuristic Admissibility

- **Euclidean distance** is admissible on 8-connected grids: the straight-line distance never exceeds the true path cost when diagonal moves cost √2. This guarantees A* returns the optimal solution.
- **Manhattan distance** is included as a reference function and is admissible for 4-connected (orthogonal-only) grids.

### Search Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| `nodes_explored` | Size of CLOSED set — space complexity proxy |
| `path_cost` | `g(goal)` — solution quality / optimality check |

---

## 4. Benchmark Results

| Scenario | Grid | A* Nodes | BFS Nodes | Path Cost | A* Reduction |
|----------|------|----------|-----------|-----------|--------------|
| Default | 10×10 | 21 | 94 | 13.898 | **−78%** |
| Hard Maze | 25×25 | 354 | 481 | 67.108 | **−26%** |
| Dead-End | 8×8 | 32 | 51 | 12.242 | **−37%** |
| Maze | 12×12 | 32 | 104 | 20.140 | **−69%** |

Path costs are identical between A* and BFS in every scenario, confirming that the admissible heuristic does not sacrifice optimality for efficiency.

---

## 5. How to Run

### Requirements

```bash
pip install matplotlib   # optional — only needed for chart export
```

No other third-party dependencies. The core code uses only the Python standard library.

### Run the GUI

```bash
python visualization.py
```

### Run the Test Suite

```bash
python tests.py
# Expected: 27/27 tests passed
```

### GUI Controls

| Control | Description |
|---------|-------------|
| Scenario dropdown | Switch between `default`, `hard_maze`, `dead_end`, `maze` |
| **Run A\*** | Animate A* exploration and path on the selected scenario |
| **Run BFS** | Animate BFS exploration and path on the selected scenario |
| **Compare Side-by-Side** | Run both algorithms simultaneously in two panels |
| Speed slider | Adjust animation speed (5–120 ms per frame) |
| **Export Benchmark PNG** | Save a full benchmark chart to `benchmark_export.png` |

---

## 6. File Structure

```
.
├── environment.py      # Tier 1 — world model and scenario factory
├── agent.py            # Tier 2 — A* and BFS search algorithms
├── visualization.py    # Tier 3 — Tkinter GUI and chart export
├── tests.py            # Automated test suite (27 tests)
└── readme.md           # This file
```
