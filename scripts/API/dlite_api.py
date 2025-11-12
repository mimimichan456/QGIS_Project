from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..QGIS.find_shelter import find_nearest_shelter
from ..QGIS.run_algorithm import run_dlite_algorithm

app = FastAPI(
    title="D* Lite Routing API",
    description="Find the closest shelter and compute a driving route.",
    version="0.1.0",
)

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
    start_point: Coordinate
    goal_point: Coordinate


class DliteRouteResponse(BaseModel):
    start_point: Coordinate
    goal_point: Coordinate
    distance_m: float
    node_count: int
    route_coords: List[Coordinate]
    route_geojson: dict


def _point_to_coordinate(point):
    # shapely.Point の場合
    if hasattr(point, "x") and hasattr(point, "y"):
        return Coordinate(lon=point.x, lat=point.y)
    # dict の場合
    if isinstance(point, dict) and "lon" in point and "lat" in point:
        return Coordinate(lon=point["lon"], lat=point["lat"])
    # タプルやリストの場合
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
        shelter_attr=result["shelter_attr"]
    )


@app.post("/run-dlite", response_model=DliteRouteResponse)
def compute_route(payload: DliteRouteRequest):
    try:
        result = run_dlite_algorithm(
            start_point=(payload.start_point.lon, payload.start_point.lat),
            goal_point=(payload.goal_point.lon, payload.goal_point.lat),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TypeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    if not result:
        raise HTTPException(status_code=404, detail="Route not found")

    return DliteRouteResponse(
        start_point=_point_to_coordinate(result["start"]),
        goal_point=_point_to_coordinate(result["goal"]),
        distance_m=result["distance_m"],
        node_count=len(result["route_nodes"]),
        route_coords=_route_coords_to_models(result["route_coords"]),
        route_geojson=_route_coords_to_geojson(result["route_coords"])
    )
