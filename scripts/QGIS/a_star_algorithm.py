import heapq
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx


class AStar:
    def __init__(
        self,
        graph: nx.Graph,
        start: int,
        goals: Sequence[int],
        node_positions: Dict[int, Tuple[float, float]],
        heuristic: str = "euclidean",
    ) -> None:
        self.graph = graph
        self.start = start
        self.goals = list(goals)
        self.goal_set: Set[int] = set(self.goals)
        self.node_positions = node_positions
        self.heuristic = heuristic

    def _heuristic(self, n1: int, n2: int) -> float:
        x1, y1 = self.node_positions[n1]
        x2, y2 = self.node_positions[n2]
        if self.heuristic == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        return math.hypot(x1 - x2, y1 - y2)

    def _reconstruct_path(self, came_from: Dict[int, int], current: int) -> List[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _run_single_goal(self, goal: int) -> Optional[Dict[str, float]]:
        open_heap: List[Tuple[float, int]] = []
        heapq.heappush(open_heap, (self._heuristic(self.start, goal), self.start))
        g_score: Dict[int, float] = {self.start: 0.0}
        came_from: Dict[int, int] = {}
        closed: Set[int] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                return {
                    "route": self._reconstruct_path(came_from, current),
                    "distance": g_score[current],
                }
            if current in closed:
                continue
            closed.add(current)

            for neighbor, data in self.graph[current].items():
                weight = float(data.get("weight", math.inf))
                if not math.isfinite(weight):
                    continue
                tentative_g = g_score[current] + weight
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))
        return None

    def run(self) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        for goal in self.goals:
            if goal not in self.graph or self.start not in self.graph:
                continue
            result = self._run_single_goal(goal)
            if not result:
                continue
            current_distance = result["distance"]
            if best is None or best.get("distance", math.inf) > current_distance:
                best = {
                    "route": result["route"],
                    "distance": current_distance,
                    "goal_id": goal,
                }
        return best
