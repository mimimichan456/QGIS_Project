import heapq
from math import hypot, isfinite

class DStarLite:
    def __init__(self, graph, start, goal, node_positions, heuristic="euclidean", initial_state=None):
        self.G = graph
        self.start = start
        self.goal = goal
        self.node_positions = node_positions
        self.g = {n: float("inf") for n in self.G.nodes} #確定距離コスト:初期値無限大
        self.rhs = {n: float("inf") for n in self.G.nodes} #見込み距離コスト:初期値無限大
        self.U = [] # 優先順位付き道路キュー
        self.km = 0 # 通行止めが発生した際の調整値

        if initial_state:
            self._load_state(initial_state)
        else:
            # 初期化
            self.rhs[self.goal] = 0
            heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    #ヒューリスティック値(ゴールまでの推定距離)を計算。
    def heuristic(self, n1, n2):
        x1, y1 = self.node_positions[n1]
        x2, y2 = self.node_positions[n2]
        return hypot(x1 - x2, y1 - y2)

    #該当ノードの優先順位を計算
    def calculate_key(self, node):
        #今考えられる最短ルートの見込み距離
        k1 = min(self.g[node], self.rhs[node]) + self.heuristic(self.start, node) + self.km
        #同順位ノードの優先決定用
        k2 = min(self.g[node], self.rhs[node])
        return (k1, k2)

    #隣接ノードのゴールからの見込み最短距離を再計算しキューに追加
    def update_vertex(self, u):
        if u != self.goal:
            #該当隣接ノードの全隣接ノードを元に見込み最短距離を更新
            neighbors = list(self.G.neighbors(u))
            if neighbors:
                self.rhs[u] = min(
                    self.g[s] + self.G[u][s]["weight"]
                    for s in neighbors
                )
            else:
                self.rhs[u] = float("inf")
        # そのノードいままでの優先順位を削除
        for i, (_, node) in enumerate(self.U):
            if node == u:
                del self.U[i]
                heapq.heapify(self.U)
                break
        # 新しい優先順位を追加
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self.calculate_key(u), u))

    # 最短経路を計算
    def compute_shortest_path(self):
        #探索が終わるまで繰り返す
        while self.U and (
            self.U[0][0] < self.calculate_key(self.start)
            or self.rhs[self.start] != self.g[self.start]
        ):
            # 一番優先順位の高いノードをキューから取得
            _, u = heapq.heappop(self.U)
            # ゴールから該当ノードまでの確定最短距離を更新
            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                #該当ノードからの全隣接ノードの優先順位を更新
                for pred in self.G.neighbors(u):
                    self.update_vertex(pred)
            # 確定距離コストを無限大に設定
            else:
                self.g[u] = float("inf")
                for pred in list(self.G.neighbors(u)) + [u]:
                    self.update_vertex(pred)

    # スタートからの経路に変換
    def extract_path(self):
        if self.g[self.start] == float("inf"):
            return None
        path = [self.start]
        current = self.start
        while current != self.goal:
            next_node = min(
                self.G.neighbors(current),
                key=lambda n: self.G[current][n]["weight"] + self.g[n]
            )
            path.append(next_node)
            current = next_node
        return path

    def export_state(self):
        return {
            # DB保存サイズを抑えるため有限値のみ保持
            "g": self._serialize_costs(self.g),
            "rhs": self._serialize_costs(self.rhs),
            "U": self._serialize_queue(),
            "km": self.km,
        }

    def _load_state(self, state):
        self.g.update(self._deserialize_costs(state.get("g", {})))
        self.rhs.update(self._deserialize_costs(state.get("rhs", {})))
        self.U = self._deserialize_queue(state.get("U", []))
        self.km = state.get("km", 0)

    @staticmethod
    def _serialize_costs(costs):
        return {
            int(node): value
            for node, value in costs.items()
            if isfinite(value)
        }

    @staticmethod
    def _deserialize_costs(costs):
        return {
            int(node): (value if value is not None else float("inf"))
            for node, value in costs.items()
        }

    def _serialize_queue(self):
        data = []
        for key, node in self.U:
            if node is None:
                continue
            # [node, k1, k2] 形式で key 名を省いてサイズ削減
            data.append([int(node), float(key[0]), float(key[1])])
        return data

    def _deserialize_queue(self, data):
        queue = []
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            node = int(item[0])
            key = (float(item[1]), float(item[2]))
            queue.append((key, node))
        if queue:
            heapq.heapify(queue)
        return queue
