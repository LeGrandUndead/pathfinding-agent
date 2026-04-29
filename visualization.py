import tkinter as tk
from tkinter import ttk
import time
from environment import make_scenario
from agent import PathfindingAgent

class AIPathfinderLab:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Lab: A* vs BFS Search Analysis")
        self.root.configure(bg="#121212")
        self.cell_size = 25
        
        # Color Palette
        self.colors = {
            "bg": "#121212", "wall": "#333333", "start": "#4CAF50",
            "goal": "#F44336", "explored": "#1A237E", "path": "#FFEB3B"
        }

        # Sidebar Controls
        self.ctrl = tk.Frame(root, bg="#1e1e1e", width=220)
        self.ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Main Display
        self.canv_frame = tk.Frame(root, bg="#121212")
        self.canv_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # UI Elements
        ttk.Label(self.ctrl, text="SCENARIO SELECTOR", background="#1e1e1e", foreground="white").pack(pady=10)
        self.scen_var = tk.StringVar(value="hard_maze")
        sc = ttk.Combobox(self.ctrl, textvariable=self.scen_var, values=["default", "hard_maze", "dead_end"])
        sc.pack(pady=5)
        sc.bind("<<ComboboxSelected>>", lambda e: self.load_grid())

        tk.Button(self.ctrl, text="RUN A* (Informed)", command=lambda: self.run_sim("astar"), bg="#2E7D32", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=10, padx=10)
        tk.Button(self.ctrl, text="RUN BFS (Uninformed)", command=lambda: self.run_sim("bfs"), bg="#1565C0", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=5, padx=10)
        
        self.stats = tk.StringVar(value="Status: Ready\nNodes: 0\nCost: 0.0")
        tk.Label(self.ctrl, textvariable=self.stats, bg="#1e1e1e", fg="#00FF00", justify=tk.LEFT, font=("Courier", 11)).pack(pady=30, padx=10)
        
        self.load_grid()

    def load_grid(self):
        self.env, self.start, self.goal = make_scenario(self.scen_var.get())
        if hasattr(self, 'canvas'): self.canvas.destroy()
        self.canvas = tk.Canvas(self.canv_frame, width=self.env.cols*self.cell_size, height=self.env.rows*self.cell_size, bg=self.colors["bg"], highlightthickness=0)
        self.canvas.pack(expand=True, padx=20, pady=20)
        self.draw_static_grid()

    def draw_static_grid(self):
        self.canvas.delete("all")
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                color = self.colors["bg"]
                if (r, c) in self.env.obstacles: color = self.colors["wall"]
                if (r, c) == self.start: color = self.colors["start"]
                if (r, c) == self.goal: color = self.colors["goal"]
                self.canvas.create_rectangle(c*self.cell_size, r*self.cell_size, (c+1)*self.cell_size, (r+1)*self.cell_size, fill=color, outline="#222", tags=f"cell_{r}_{c}")

    def run_sim(self, mode):
        self.draw_static_grid()
        agent = PathfindingAgent(self.env)
        res = agent.astar(self.start, self.goal) if mode == "astar" else agent.bfs(self.start, self.goal)
        
        # Step 1: Animate Exploration (The Wave)
        for i, s in enumerate(res.explored_order):
            if s != self.start and s != self.goal:
                self.canvas.itemconfig(f"cell_{s[0]}_{s[1]}", fill=self.colors["explored"])
                # Speed up rendering for large mazes
                if i % 4 == 0: 
                    self.root.update()
                    time.sleep(0.005)
        
        # Step 2: Animate Final Path
        if res.success:
            for s in res.path:
                if s != self.start and s != self.goal:
                    self.canvas.itemconfig(f"cell_{s[0]}_{s[1]}", fill=self.colors["path"])
                    self.root.update()
                    time.sleep(0.03)
            self.stats.set(f"ALGO: {res.algorithm}\nNodes: {res.nodes_explored}\nCost: {res.path_cost}\nSUCCESS!")
        else:
            self.stats.set("Status: BLOCKED\nNo path found.")

if __name__ == "__main__":
    root = tk.Tk()
    AIPathfinderLab(root)
    root.mainloop()