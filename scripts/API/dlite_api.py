import os
import json

from typing import List, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scripts.QGIS.find_shelter import find_nearest_shelter
from scripts.QGIS.run_algorithm import run_dlite_algorithm
from scripts.QGIS.find_load import find_nearest_road_edge
from scripts.DB.dlite_db import (
    save_session_state,
    load_session_state,
    reset_blocked_point as db_reset_blocked_point,
)


GEOJSON_DIR = os.path.join(os.path.dirname(__file__), "../../data/route/geojson")
os.makedirs(GEOJSON_DIR, exist_ok=True)
SAVE_ROUTE_GEOJSON = os.getenv("SAVE_ROUTE_GEOJSON", "0").lower() in {"1", "true", "yes"}

class Coordinate(BaseModel):
    lon: float = Field(..., ge=-180, le=180, description="Longitude in EPSG:4326")
    lat: float = Field(..., ge=-90, le=90, description="Latitude in EPSG:4326")


class FindShelterRequest(BaseModel):
    start_point: Coordinate


class ShelterCandidate(BaseModel):
    goal_point: Coordinate
    shelter_attr: dict
    distance_m: float


class FindShelterResponse(BaseModel):
    start_point: Coordinate
    candidates: List[ShelterCandidate]


class DliteRouteRequest(BaseModel):
    session_id: Optional[str] = None
    start_point: Coordinate
    goal_point: List[Coordinate]


class BlockedEdge(BaseModel):
    u: int
    v: int


class GoalCandidate(BaseModel):
    goal_point: Coordinate


class DliteRouteResponse(BaseModel):
    session_id: str
    start_point: Coordinate
    distance_m: float
    route_coords: List[Coordinate]
    blocked_edges: List[BlockedEdge]
    goal_candidates: List[GoalCandidate] = []


class BlockRoadRequest(BaseModel):
    session_id: str
    blocked_point: Coordinate
    start_point: Optional[Coordinate] = None


class ResetBlockedPointRequest(BaseModel):
    session_id: str


class ResetBlockedPointResponse(BaseModel):
    session_id: str
    status: str


app = FastAPI(
    title="D* Lite Routing API",
    description="Find the closest shelter, compute a route, and manage D* Lite sessions.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _point_to_coordinate(point):
    if hasattr(point, "x") and hasattr(point, "y"):
        return Coordinate(lon=point.x, lat=point.y)
    if isinstance(point, dict) and "lon" in point and "lat" in point:
        return Coordinate(lon=point["lon"], lat=point["lat"])
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return Coordinate(lon=point[0], lat=point[1])
    raise TypeError(f"Unsupported point type: {type(point)}")


def _route_coords_to_models(coords):
    return [Coordinate(lon=lon, lat=lat) for lon, lat in coords]


def _coordinate_to_dict(coord: Coordinate):
    return {"lon": coord.lon, "lat": coord.lat}


def _blocked_edges_to_models(edges):
    return [BlockedEdge(u=edge["u"], v=edge["v"]) for edge in (edges or [])]


def _persist_session(session_id: str, result, start_coord: Coordinate):
    state = result["dlite_state"]
    goal_points = [item["goal_point"] for item in (result.get("goal_candidates") or [])]
    if not goal_points and result.get("goal"):
        goal_points = [result["goal"]]
    goal_id = result.get("goal_id")
    if goal_id is None:
        node_ids = result.get("goal_node_ids") or []
        if node_ids:
            goal_id = node_ids[0]
        else:
            goal_id = result["start_id"]
    try:
        save_session_state(
            session_id,
            g=state["g"],
            rhs=state["rhs"],
            queue=state["U"],
            start_id=result["start_id"],
            goal_id=int(goal_id),
            goal_node_ids=result.get("goal_node_ids") or [],
            blocked_edges=result["blocked_edges"],
            start_point=_coordinate_to_dict(start_coord),
            goal_points=goal_points,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist session: {exc}") from exc

def save_route_geojson(session_id: str, coords):
    """route_coords → GeoJSONファイルとして保存"""
    geojson_data = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(lon), float(lat)] for lon, lat in coords],
        },
        "properties": {
            "session_id": session_id
        }
    }

    file_path = os.path.join(GEOJSON_DIR, f"{session_id}.geojson")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    return file_path

def _build_response(session_id: str, result) -> DliteRouteResponse:
    candidates = [
        GoalCandidate(
            goal_point=_point_to_coordinate(item["goal_point"]),
        )
        for item in (result.get("goal_candidates") or [])
    ]

    return DliteRouteResponse(
        session_id=session_id,
        start_point=_point_to_coordinate(result["start"]),
        distance_m=result["distance_m"],
        route_coords=_route_coords_to_models(result["route_coords"]),
        blocked_edges=_blocked_edges_to_models(result["blocked_edges"]),
        goal_candidates=candidates,
    )


@app.post("/find-shelter", response_model=FindShelterResponse)
def find_shelter(payload: FindShelterRequest):
    try:
        result = find_nearest_shelter(
            start_point=(payload.start_point.lon, payload.start_point.lat)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return FindShelterResponse(
        start_point=_point_to_coordinate(result["start_point"]),
        candidates=[
            ShelterCandidate(
                goal_point=_point_to_coordinate(item["goal_point"]),
                shelter_attr=item.get("shelter_attr", {}),
                distance_m=item.get("distance_m", 0.0),
            )
            for item in result.get("candidate_shelters", [])
        ],
    )


@app.post("/run-dlite", response_model=DliteRouteResponse)
def compute_route(payload: DliteRouteRequest):
    session_id = payload.session_id or str(uuid4())

    try:
        result = run_dlite_algorithm(
            start_point=(payload.start_point.lon, payload.start_point.lat),
            goal_point=[{"lon": p.lon, "lat": p.lat} for p in payload.goal_point],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result:
        raise HTTPException(status_code=404, detail="Route not found")

    _persist_session(session_id, result, payload.start_point)
    if SAVE_ROUTE_GEOJSON:
        save_route_geojson(session_id, result["route_coords"])
    return _build_response(session_id, result)


@app.post("/reroute", response_model=DliteRouteResponse)
def reroute(payload: BlockRoadRequest):
    session = load_session_state(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.get("start_point"):
        raise HTTPException(status_code=500, detail="Session is missing start/goal coordinates")

    blocked_edges = session.get("blocked_edges") or []
    start_node_id = session["start_id"]
    if payload.start_point:
        start_point = _coordinate_to_dict(payload.start_point)
        start_node_id = None
    else:
        start_point = session["start_point"]
    start_coord_model = payload.start_point or _point_to_coordinate(start_point)
    goal_points = session.get("goal_points") or []
    if not goal_points and session.get("goal_point"):
        goal_points = [session["goal_point"]]
    goal_node_ids = session.get("goal_node_ids") or []
    if not goal_node_ids and session.get("goal_id") is not None:
        goal_node_ids = [session["goal_id"]]

    try:
        nearest_edge = find_nearest_road_edge(
            (payload.blocked_point.lon, payload.blocked_point.lat)
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    new_edge = {
        "u": nearest_edge["edge"][0],
        "v": nearest_edge["edge"][1],
        "blocked_point": {
            "lon": payload.blocked_point.lon,
            "lat": payload.blocked_point.lat,
        },
    }
    already_blocked = any(
        edge.get("u") == new_edge["u"] and edge.get("v") == new_edge["v"] for edge in blocked_edges
    )
    if not already_blocked:
        blocked_edges.append(new_edge)

    initial_state = {
        "g": session["g"],
        "rhs": session["rhs"],
        "U": session["queue"],
        "km": session.get("km", 0),
    }

    try:
        result = run_dlite_algorithm(
            start_point=start_point,
            goal_point=goal_points,
            start_node_id=start_node_id,
            goal_node_id=goal_node_ids,
            initial_state=initial_state,
            blocked_edges=blocked_edges,
            new_blocked_edges=[new_edge],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    if not result:
        raise HTTPException(status_code=404, detail="Route not found")

    result["blocked_edges"] = blocked_edges
    _persist_session(payload.session_id, result, start_coord_model)
    if SAVE_ROUTE_GEOJSON:
        save_route_geojson(payload.session_id, result["route_coords"])
    return _build_response(payload.session_id, result)


@app.post("/reset-blocked-point", response_model=ResetBlockedPointResponse)
def reset_blocked_point(payload: ResetBlockedPointRequest):
    updated = db_reset_blocked_point(payload.session_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return ResetBlockedPointResponse(session_id=payload.session_id, status="reset")
