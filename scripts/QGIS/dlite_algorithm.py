import heapq
from math import hypot, isfinite

class DStarLite:
    def __init__(self, graph, start, goal, node_positions, heuristic="euclidean", initial_state=None):
        self.G = graph
        self.start = start
        self.goal = goal
        self.node_positions = node_positions
        self.g = {n: float("inf") for n in self.G.nodes}
        self.rhs = {n: float("inf") for n in self.G.nodes}
        self.U = []
        self.km = 0.0
        self.heuristic_type = heuristic
        self.last_start = start  # km更新用

        if initial_state:
            self._load_state(initial_state)
        else:
            self.rhs[self.goal] = 0.0
            heapq.heappush(self.U, (self._calculate_key(self.goal), self.goal))

    # --- 基本関数 ---
    def _heuristic(self, n1, n2):
        x1, y1 = self.node_positions[n1]
        x2, y2 = self.node_positions[n2]
        if self.heuristic_type == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        return hypot(x1 - x2, y1 - y2)


    def _calculate_key(self, node):
        g_val = self.g[node]
        rhs_val = self.rhs[node]
        min_val = min(g_val, rhs_val)
        return (min_val + self._heuristic(self.start, node) + self.km, min_val)
    
    # --- 頂点更新 ---
    def update_vertex(self, u):
        if u != self.goal:
            nbrs = list(self.G.neighbors(u))
            if nbrs:
                self.rhs[u] = min(self.g[s] + self.G[u][s]["weight"] for s in nbrs)
            else:
                self.rhs[u] = float("inf")

        # キュー再登録（重複OK：小規模用途では問題なし）
        for i, (_, node) in enumerate(self.U):
            if node == u:
                del self.U[i]
                heapq.heapify(self.U)
                break
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self._calculate_key(u), u))

    # --- 最短経路探索 ---
    def compute_shortest_path(self):
        while self.U and (
            self.U[0][0] < self._calculate_key(self.start)
            or self.rhs[self.start] != self.g[self.start]
        ):
            _, u = heapq.heappop(self.U)

            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for pred in self.G.neighbors(u):
                    self.update_vertex(pred)
            else:
                self.g[u] = float("inf")
                for pred in list(self.G.neighbors(u)) + [u]:
                    self.update_vertex(pred)

    # --- 経路抽出 ---
    def extract_path(self):
        if not isfinite(self.g[self.start]):
            return None

        path = [self.start]
        current = self.start
        while current != self.goal:
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                return None
            next_node = min(
                neighbors,
                key=lambda n: self.G[current][n]["weight"] + self.g.get(n, float("inf"))
            )
            if not isfinite(self.g.get(next_node, float("inf"))):
                return None
            path.append(next_node)
            current = next_node
        return path

    # --- km更新 / 通行止め処理 ---
    def set_start(self, new_start):
        """スタート位置が動いたときに呼ぶ（kmを補正）。"""
        if new_start != self.start:
            self.km += self._heuristic(self.last_start, new_start)
            self.last_start = new_start
            self.start = new_start

    def update_edge_cost(self, u, v, new_weight):
        """エッジ重み変更（通行止め/解除）"""
        if not self.G.has_node(u) or not self.G.has_node(v):
            return  # 安全ガード

        if self.G.has_edge(u, v):
            self.G[u][v]["weight"] = float(new_weight)
        if self.G.has_edge(v, u):
            self.G[v][u]["weight"] = float(new_weight)
        self.update_vertex(u)
        self.update_vertex(v)

    # --- 状態保存 / 復元 ---
    def export_state(self):
        return {
            "g": {int(n): v for n, v in self.g.items() if isfinite(v)},
            "rhs": {int(n): v for n, v in self.rhs.items() if isfinite(v)},
            "U": [[int(node), float(k[0]), float(k[1])] for k, node in self.U if node is not None],
            "km": float(self.km),
            "start": int(self.start),
            "goal": int(self.goal),
            "last_start": int(self.last_start),
        }

    def _load_state(self, state):
        for node, val in state.get("g", {}).items():
            self.g[int(node)] = float(val)
        for node, val in state.get("rhs", {}).items():
            self.rhs[int(node)] = float(val)
        self.U = []
        for item in state.get("U", []):
            if isinstance(item, (list, tuple)) and len(item) == 3:
                node = int(item[0])
                key = (float(item[1]), float(item[2]))
                self.U.append((key, node))
        if self.U:
            heapq.heapify(self.U)
        self.km = float(state.get("km", 0.0))
        self.start = int(state.get("start", self.start))
        self.goal = int(state.get("goal", self.goal))
        self.last_start = int(state.get("last_start", self.start))