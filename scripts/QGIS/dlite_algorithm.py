import networkx as nx
import heapq
from math import hypot, isfinite
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Set, Tuple

class DStarLite:
    def __init__(
        self,
        graph: nx.Graph,
        start: int,
        goal: Sequence[int],
        node_positions: MutableMapping[int, Tuple[float, float]],
        heuristic: str = "euclidean",
        initial_state: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> None:
        self.G = graph #道路ネットワーク
        mapping = {n: int(float(n)) for n in self.G.nodes}
        self.G = nx.relabel_nodes(self.G, mapping, copy=True)
        self.adj = self.G.adj
        self.g: Dict[int, float] = {int(n): float("inf") for n in self.G.nodes}
        self.rhs: Dict[int, float] = {int(n): float("inf") for n in self.G.nodes}
        self.start: int = start #出発地点

        self.goal_nodes: List[int] = self._normalize_goals(goal) #避難所のリスト
        self.goal_set: Set[int] = set(self.goal_nodes) #避難所のノード集合
        self.goal: int = self.goal_nodes[0] #最短の避難所

        self.node_positions: MutableMapping[int, Tuple[float, float]] = node_positions

        self.U: List[Tuple[Tuple[float, float], int]] = [] # キー付きオープン道路リスト
        self.queue_keys: Dict[int, Optional[Tuple[float, float]]] = {}

        self.km: float = 0.0 # 出発地点の移動距離
        self.last_start: int = start  # km更新用

        self.heuristic_type: str = heuristic # キーに使うヒューリスティックの種類

        self.reached_goal: Optional[int] = None

        self.debug: bool = bool(debug)

        if initial_state:
            self._load_state(initial_state) #再探索時の状態復元
            self.set_start(start)
        else:
            for goal_node in self.goal_nodes:
                self.rhs[goal_node] = 0.0
                key = self._calculate_key(goal_node)
                self.queue_keys[goal_node] = key
                heapq.heappush(self.U, (key, goal_node))

    def _log(self, *args: Any) -> None:
        if self.debug:
            print(*args)

    def _normalize_goals(self, goal: Sequence[int]) -> List[int]:
        if goal is None:
            return []
        if isinstance(goal, (list, tuple, set)):
            return [int(g) for g in goal if g is not None]
        return [int(goal)]

    def _is_goal(self, node: int) -> bool:
        return node in self.goal_set

    #スタート地点までの直線距離計算
    def _heuristic(self, n1: int, n2: int) -> float:
        x1, y1 = self.node_positions[n1]
        x2, y2 = self.node_positions[n2]
        if self.heuristic_type == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        return hypot(x1 - x2, y1 - y2)

    #オープン道路リストの優先度キー計算(km, ヒューリスティック, g,rhsの最小値)により優先度キーの決定
    def _calculate_key(self, node: int) -> Tuple[float, float]:
        val = min(self.g[node], self.rhs[node])
        return (val + self._heuristic(self.start, node) + self.km, val)


    #道路のrhs値更新
    def update_vertex(self, u: int) -> None:
        self._log(f"[DV] update_vertex({u}): g={self.g.get(u)}, rhs={self.rhs.get(u)}")
        # --- DEBUG: show neighbors and edge weights ---
        if self.debug:
            self._log(f"[DV] neighbors of {u} = {list(self.adj[u].keys())}")
            for s, data in self.adj[u].items():
                self._log(f"    -> neighbor {s}, weight={data.get('weight')}")
        if not self._is_goal(u):
            neighbors = self.adj[u]#隣接ノード取得
            if neighbors:
                #全ての隣接ノードからの最短距離を計算し、その最小値をrhsとして更新
                self.rhs[u] = min(
                    self.g[s] + neighbors[s]["weight"]
                    for s in neighbors
                )
            else:
                self.rhs[u] = float("inf")
        self._log(f"[DV] → updated rhs[{u}] = {self.rhs[u]}")
        self._log(f"[DV] g[{u}]={self.g[u]}, rhs[{u}]={self.rhs[u]} → pushing to U={self.g[u] != self.rhs[u]}")

        if self.g[u] != self.rhs[u]:
            key = self._calculate_key(u)
            self.queue_keys[u] = key
            heapq.heappush(self.U, (key, u))
        else:
            self.queue_keys[u] = None

    # --- 最短経路探索 ---
    def compute_shortest_path(self) -> None:
        while self.U:
            # 無効なエントリをスキップ
            while self.U and self.queue_keys.get(self.U[0][1]) != self.U[0][0]:
                heapq.heappop(self.U)
            if not self.U:
                break
            self._log("[LOOP] top of compute_shortest_path")
            self._log("   U.min_key() =", self.U[0][0] if self.U else None)
            self._log("   key(start) =", self._calculate_key(self.start))
            self._log("   g(start) =", self.g.get(self.start), "rhs(start) =", self.rhs.get(self.start))
            self._log("   g(goal) =", self.g.get(self.goal), "rhs(goal) =", self.rhs.get(self.goal))
            top_key, _ = self.U[0]
            self._log("[STOP-CHECK] top_key <", self._calculate_key(self.start), "=", top_key < self._calculate_key(self.start))
            self._log("[STOP-CHECK] rhs[start] != g[start] =", self.rhs[self.start] != self.g[self.start])
            if not (top_key < self._calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]):
                break #スタートノードのキーが最小で、rhsとgが等しい場合は終了

            _, u = heapq.heappop(self.U)
            if self.queue_keys.get(u) != top_key:
                continue
            self.queue_keys[u] = None

            if self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]#gがrhsより大きい場合、rhsをgに更新
                for pred in self.adj[u]:#全ての隣接ノードのrhs更新
                    self.update_vertex(pred)
            else:
                self.g[u] = float("inf")# rhsがgより大きい場合をgを無限大に更新
                for pred in list(self.adj[u].keys()) + [u]:#全ての隣接ノード及び自身のrhs更新
                    self.update_vertex(pred)


    # --- 結果より最短経路抽出 ---
    def extract_path(self) -> Optional[List[int]]:
        # g(start) が無限 → 経路が存在しない
        if not isfinite(self.g.get(self.start, float("inf"))):
            self.reached_goal = None
            return None

        path = [self.start]
        current = self.start

        # すでに避難所なら終了
        if self._is_goal(current):
            self.reached_goal = current
            return path

        visited = {current: 0}   # ループ防止

        # 安全最大ステップ（異常系の無限ループ防止）
        MAX_STEPS = 10000

        steps = 0
        while not self._is_goal(current):

            steps += 1
            if steps > MAX_STEPS:
                self._log("[extract_path] ERROR: path exceeded MAX_STEPS")
                self.reached_goal = None
                return None

            neighbors = self.adj.get(current)
            if not neighbors:
                self._log("[extract_path] no neighbors from", current)
                self.reached_goal = None
                return None

            # 隣接ノードから最良候補を決定
            best_next = None
            best_cost = float("inf")

            for nxt, data in neighbors.items():
                w = data.get("weight", float("inf"))
                g_next = self.g.get(nxt, float("inf"))

                if not isfinite(w) or not isfinite(g_next):
                    continue

                total = w + g_next
                if total < best_cost:
                    best_cost = total
                    best_next = nxt

            if best_next is None:
                self._log(f"[extract_path] no valid next node from {current}")
                self.reached_goal = None
                return None

            # ループ（戻りや循環）防止
            if best_next in visited:
                prev_step = visited[best_next]
                self._log(
                    f"[extract_path] loop detected: {best_next} was already visited at step {prev_step}"
                )
                self.reached_goal = None
                return None

            visited[best_next] = steps
            path.append(best_next)
            current = best_next

        self.reached_goal = current
        return path

    # --- 移動によるズレ補正用のkm更新 ---
    def set_start(self, new_start: int) -> None:
        if new_start != self.start:
            self.km += self._heuristic(self.last_start, new_start)#kmを移動前からの直線距離を元に更新
            self.last_start = new_start
            self.start = new_start

    # エッジ重み変更（通行止め/解除）
    def update_edge_cost(self, u: int, v: int, new_weight: Optional[float]) -> None:
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

        self._log(f"[EDGE] update_edge_cost u={u}, v={v}, new_weight={new_weight}")
        self.update_vertex(u)
        self.update_vertex(v)

        self.update_vertex(self.start)

    # --- 到達した避難所ノードを取得 ---
    def get_reached_goal(self) -> Optional[int]:
        return self.reached_goal

    # --- 道路の情報保存 ---
    def export_state(self) -> Dict[str, Any]:
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
    def _load_state(self, state: Dict[str, Any]) -> None:
        self.g.update({int(n): float(v) for n, v in state.get("g", {}).items()})
        self.rhs.update({int(n): float(v) for n, v in state.get("rhs", {}).items()})
        self.U = []
        self.queue_keys = {}
        for item in state.get("U", []):
            if isinstance(item, (list, tuple)) and len(item) == 3:
                node = int(item[0])
                key = (float(item[1]), float(item[2]))
                heapq.heappush(self.U, (key, node))
                self.queue_keys[node] = key
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
