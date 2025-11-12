import json
from pathlib import Path
from fastapi.responses import FileResponse
from typing import List, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from scripts.QGIS.find_shelter import find_nearest_shelter
from scripts.QGIS.run_algorithm import run_dlite_algorithm
from scripts.QGIS.find_load import find_nearest_road_edge
from scripts.DB.dlite_db import save_session_state, load_session_state


class Coordinate(BaseModel):
    lon: float = Field(..., ge=-180, le=180, description="Longitude in EPSG:4326")
    lat: float = Field(..., ge=-90, le=90, description="Latitude in EPSG:4326")


class FindShelterRequest(BaseModel):
    start_point: Coordinate


class FindShelterResponse(BaseModel):
    start_point: Coordinate
    goal_point: Coordinate
    shelter_attr: dict


class DliteRouteRequest(BaseModel):
    session_id: Optional[str] = None
    start_point: Coordinate
    goal_point: Coordinate


class BlockedEdge(BaseModel):
    u: int
    v: int


class DliteRouteResponse(BaseModel):
    session_id: str
    start_point: Coordinate
    goal_point: Coordinate
    distance_m: float
    node_count: int
    route_nodes: List[int]
    route_coords: List[Coordinate]
    route_geojson: dict
    blocked_edges: List[BlockedEdge]


class BlockRoadRequest(BaseModel):
    session_id: str
    blocked_point: Coordinate
    
    
app = FastAPI(
    title="D* Lite Routing API",
    description="Find the closest shelter, compute a route, and manage D* Lite sessions.",
    version="0.2.0",
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


def _route_coords_to_geojson(coords):
    return {
        "type": "LineString",
        "coordinates": coords,
    }


def _coordinate_to_dict(coord: Coordinate):
    return {"lon": coord.lon, "lat": coord.lat}


def _blocked_edges_to_models(edges):
    return [BlockedEdge(u=edge["u"], v=edge["v"]) for edge in (edges or [])]


def _persist_session(session_id: str, result, start_coord: Coordinate, goal_coord: Coordinate):
    state = result["dlite_state"]  # D* Lite 側でシリアライズ時に軽量化済み
    try:
        save_session_state(
            session_id,
            g=state["g"],
            rhs=state["rhs"],
            queue=state["U"],
            start_id=result["start_id"],
            goal_id=result["goal_id"],
            blocked_edges=result["blocked_edges"],
            start_point=_coordinate_to_dict(start_coord),
            goal_point=_coordinate_to_dict(goal_coord),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist session: {exc}") from exc


def _build_response(session_id: str, result) -> DliteRouteResponse:
    return DliteRouteResponse(
        session_id=session_id,
        start_point=_point_to_coordinate(result["start"]),
        goal_point=_point_to_coordinate(result["goal"]),
        distance_m=result["distance_m"],
        node_count=len(result["route_nodes"]),
        route_nodes=result["route_nodes"],
        route_coords=_route_coords_to_models(result["route_coords"]),
        route_geojson=_route_coords_to_geojson(result["route_coords"]),
        blocked_edges=_blocked_edges_to_models(result["blocked_edges"]),
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
        goal_point=_point_to_coordinate(result["goal_point"]),
        shelter_attr=result["shelter_attr"],
    )


@app.post("/run-dlite", response_model=DliteRouteResponse)
def compute_route(payload: DliteRouteRequest):
    session_id = payload.session_id or str(uuid4())
    
    try:
        result = run_dlite_algorithm(
            start_point=(payload.start_point.lon, payload.start_point.lat),
            goal_point=(payload.goal_point.lon, payload.goal_point.lat),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result:
        raise HTTPException(status_code=404, detail="Route not found")

    _persist_session(session_id, result, payload.start_point, payload.goal_point)

    return _build_response(session_id, result)


@app.post("/reroute", response_model=DliteRouteResponse)
def reroute(payload: BlockRoadRequest):
    session = load_session_state(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.get("start_point") or not session.get("goal_point"):
        raise HTTPException(status_code=500, detail="Session is missing start/goal coordinates")

    blocked_edges = session.get("blocked_edges") or []
    start_point = session["start_point"]
    goal_point = session["goal_point"]

    try:
        nearest_edge = find_nearest_road_edge(
            (payload.blocked_point.lon, payload.blocked_point.lat)
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    new_edge = {"u": nearest_edge["edge"][0], "v": nearest_edge["edge"][1]}
    if new_edge not in blocked_edges:
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
            goal_point=goal_point,
            start_node_id=session["start_id"],
            goal_node_id=session["goal_id"],
            initial_state=initial_state,
            blocked_edges=[(edge["u"], edge["v"]) for edge in blocked_edges],
            new_blocked_edges=[(new_edge["u"], new_edge["v"])],
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
    _persist_session(payload.session_id, result, _point_to_coordinate(start_point), _point_to_coordinate(goal_point))
    return _build_response(payload.session_id, result)
