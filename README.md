# Pathfinding Agent: A* vs BFS Comparison

**Semester Project — AI Application**

---

## 1. Project Overview

This project implements a **Rational Agent** capable of navigating a discrete, static grid environment. It compares two fundamental search strategies:

- **A\* Search (Informed)**  
  Uses the Manhattan distance heuristic to efficiently compute optimal paths.

- **Breadth-First Search (BFS) (Uninformed)**  
  Serves as a baseline to highlight the performance advantages of informed search.

---

## 2. Architecture (3-Tier Design)

The application is structured into three modular layers:

### Tier 1: Environment (`environment.py`)
- **State Space:** Grid represented as `(row, col)` coordinates  
- **Transition Model:**  
  - Valid moves: Up, Down, Left, Right  
  - Handles obstacle detection  

---

### Tier 2: Agent Logic (`agent.py`)
- **A\* Search Implementation**
  - Uses a **priority queue (min-heap)**
  - Evaluation function:  
    ```
    f(n) = g(n) + h(n)
    ```
- **Heuristic Function**
  - Manhattan Distance:
    ```
    h(n) = |x1 - x2| + |y1 - y2|
    ```
  - Admissible → guarantees optimal paths in 4-connected grids

- **Graph Search Features**
  - Closed set to prevent revisiting nodes  
  - Ensures **completeness** and avoids infinite loops  

---

### Tier 3: Visualization (`visualization.py`)
- **Terminal Animation**
  - Frontier nodes → Cyan  
  - Explored nodes → Blue  

- **Benchmarking System**
  - Compares performance across scenarios:
    - Default
    - Maze
    - Dead-End  

  - Metrics collected:
    - Execution Time
    - Nodes Explored (Space Complexity)
    - Path Cost (Optimality)

---

## 3. Key AI Concepts Applied

### PEAS Framework
- **Performance Measure:** Path cost  
- **Environment:** Grid world  
- **Actuators:** Movement (Up, Down, Left, Right)  
- **Sensors:** Valid-cell detection  

---

### Search Evaluation Metrics
- `nodes_explored` → Space complexity  
- `path_cost` → Optimality  

---

### Agent Type
- **Model-Based Agent**
  - Uses known environment rules  
  - No learning required  
  - Operates in a **fully observable** world  

---

## 4. How to Run

### Run the Application
```bash
python visualization.py