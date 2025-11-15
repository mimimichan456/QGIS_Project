import heapq
from math import hypot, isfinite

class DStarLite:
    def __init__(self, graph, start, goal, node_positions, heuristic="euclidean", initial_state=None):
        self.G = graph #道路ネットワーク
        self.start = start #出発地点

        self.goal_nodes = self._normalize_goals(goal) #避難所のリスト
        self.goal_set = set(self.goal_nodes) #避難所のノード集合
        self.goal = self.goal_nodes[0] #最短の避難所

        self.node_positions = node_positions

        self.g = {n: float("inf") for n in self.G.nodes} #避難所からの推定推定最短距離
        self.rhs = {n: float("inf") for n in self.G.nodes} #元ノードのg+元ノードからの推定最短距離

        self.U = [] # キー付きオープン道路リスト

        self.km = 0.0 # 出発地点の移動距離
        self.last_start = start  # km更新用

        self.heuristic_type = heuristic # キーに使うヒューリスティックの種類

        self.adj = self.G.adj
        self.reached_goal = None

        if initial_state:
            self._load_state(initial_state) #再探索時の状態復元
        else:
            for goal_node in self.goal_nodes:
                self.rhs[goal_node] = 0.0
                heapq.heappush(self.U, (self._calculate_key(goal_node), goal_node))

    def _normalize_goals(self, goal):
        if goal is None:
            return []
        if isinstance(goal, (list, tuple, set)):
            return [int(g) for g in goal if g is not None]
        return [int(goal)]

    #スタート地点までの直線距離計算
    def _heuristic(self, n1, n2):
        x1, y1 = self.node_positions[n1]
        x2, y2 = self.node_positions[n2]
        if self.heuristic_type == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        return hypot(x1 - x2, y1 - y2)

    #オープン道路リストの優先度キー計算(km, ヒューリスティック, g,rhsの最小値)により優先度キーの決定
    def _calculate_key(self, node):
        val = min(self.g[node], self.rhs[node])
        return (val + self._heuristic(self.start, node) + self.km, val)


    #道路のrhs値更新
    def update_vertex(self, u):
        if u not in self.goal_set:
            neighbors = self.adj[u]#隣接ノード取得
            if neighbors:
                #全ての隣接ノードからの最短距離を計算し、その最小値をrhsとして更新
                self.rhs[u] = min(self.g[s] + data["weight"] for s, data in neighbors.items())
            else:
                self.rhs[u] = float("inf")

        self.U = [(k, n) for k, n in self.U if n != u]#オープン道路リストから削除
        heapq.heapify(self.U)

        # --- rhs と g が異なる場合に再びオープン道路リストに追加 ---
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self._calculate_key(u), u))

    # --- 最短経路探索 ---
    def compute_shortest_path(self):
        while self.U:#オープン道路リストが空になるまで
            top_key, _ = self.U[0]
            if not (top_key < self._calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
                break #スタートノードのキーが最小で、rhsとgが等しい場合は終了

            _, u = heapq.heappop(self.U)

            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]#gがrhsより大きい場合、rhsをgに更新
                for pred in self.adj[u]:#全ての隣接ノードのrhs更新
                    self.update_vertex(pred)
            else:
                self.g[u] = float("inf")# rhsがgより大きい場合をgを無限大に更新
                for pred in list(self.adj[u].keys()) + [u]:#全ての隣接ノード及び自身のrhs更新
                    self.update_vertex(pred)


    # --- 結果より最短経路抽出 ---
    def extract_path(self):
        if not isfinite(self.g[self.start]):
            return None

        path = [self.start]
        current = self.start

        if current in self.goal_set:
            self.reached_goal = current
            return path

        #--- 避難所に到達するまで経路をたどる ---
        while current not in self.goal_set:
            neighbors = self.adj[current]#現在地点からの隣接ノード取得

            if not neighbors:
                self.reached_goal = None
                return None

            #--- 隣接ノードの避難所までの距離 + 隣接ノードまでの距離 が最小のノードを次の道路に選択 ---
            next_node = min(
                neighbors.items(),
                key=lambda item: item[1]["weight"] + self.g.get(item[0], float("inf"))
            )[0]

            if not isfinite(self.g.get(next_node, float("inf"))):
                self.reached_goal = None
                return None

            if len(path) > 1000000:
                print(f"[D*Lite] extract_path path length exceeded at {len(path)}; current={current}")
                self.reached_goal = None
                return None

            path.append(next_node)#経路として追加
            current = next_node

        self.reached_goal = current
        return path

    # --- 移動によるズレ補正用のkm更新 ---
    def set_start(self, new_start):
        if new_start != self.start:
            self.km += self._heuristic(self.last_start, new_start)#kmを移動前からの直線距離を元に更新
            self.last_start = new_start
            self.start = new_start

    # エッジ重み変更（通行止め/解除）
    def update_edge_cost(self, u, v, new_weight):

        if not (self.G.has_node(u) and self.G.has_node(v)):
            return

        def _set_weight(src, dst):
            if src not in self.adj or dst not in self.adj[src]:
                return False
            edge_data = self.adj[src][dst]
            base = edge_data.get("base_weight", edge_data.get("weight", float("inf")))#道路の重みを取得

            if new_weight is None:
                edge_data["weight"] = base #通行止め解除の場合は元の重みに戻す
            else:
                edge_data["weight"] = float(new_weight) #通行止めの場合は無限大に設定
            return True

        updated = _set_weight(u, v)
        updated = _set_weight(v, u) or updated

        if not updated:
            return

        self.update_vertex(u)
        self.update_vertex(v)

    # --- 到達した避難所ノードを取得 ---
    def get_reached_goal(self):
        return self.reached_goal

    # --- 道路の情報保存 ---
    def export_state(self):
        return {
            "g": {int(n): float(v) for n, v in self.g.items() if isfinite(v)},
            "rhs": {int(n): float(v) for n, v in self.rhs.items() if isfinite(v)},
            "U": [
                [int(node), float(key[0]), float(key[1])]
                for key, node in self.U
            ],
            "km": float(self.km),
            "start": int(self.start),
            "goal": int(self.goal),
            "goals": [int(g) for g in self.goal_nodes],
            "last_start": int(self.last_start),
        }

    # --- 道路状態の読み込み ---
    def _load_state(self, state):
        self.g.update({int(n): float(v) for n, v in state.get("g", {}).items()})
        self.rhs.update({int(n): float(v) for n, v in state.get("rhs", {}).items()})
        self.U = []
        for item in state.get("U", []):
            if isinstance(item, (list, tuple)) and len(item) == 3:
                node = int(item[0])
                key = (float(item[1]), float(item[2]))
                heapq.heappush(self.U, (key, node))
        self.km = float(state.get("km", 0.0))
        self.start = int(state.get("start", self.start))
        self.last_start = int(state.get("last_start", self.start))
        goals = state.get("goals")
        if goals:
            self.goal_nodes = [int(g) for g in goals]
        else:
            self.goal_nodes = [int(state.get("goal", self.goal))]
        self.goal_set = set(self.goal_nodes)
        self.goal = self.goal_nodes[0]
        self.reached_goal = None
