"""
deliveries.py — One delivery per request, always.

*Calibration travels with the data.* Every payload that contains `speed_kmh`
also carries the calibration block, so a client cannot render an estimate
without also receiving the facts that make it an estimate.

*One delivery per request.* Every route here takes a `delivery_id` and returns
only that delivery's fields. There is no endpoint that mixes two deliveries,
which is what stops a client assembling a shot from one ball and a trajectory
from another.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...analytics.pitch_map import pitch_map_svg
from ...core.config import Paths
from ...core.constants import SCHEMA_VERSION
from ...storage.persistence import read_json
from ..helpers import find_delivery, parse_delivery_id, result_or_404

router = APIRouter()


@router.get("/analysis/{analysis_id}/deliveries")
def analysis_deliveries(analysis_id: str) -> dict:
    """
    Every validated delivery, in order.

    Only deliveries that passed purity validation appear here. Anything the
    validator refused is in `/result` under `quarantine`, with the reason.
    """
    doc = result_or_404(analysis_id)
    deliveries = doc.get("deliveries", [])
    return {
        "analysis_id": analysis_id,
        "schema_version": doc.get("schema_version", SCHEMA_VERSION),
        "total_deliveries": len(deliveries),
        "summary": doc.get("summary"),
        "calibration": doc.get("calibration"),
        "deliveries": deliveries,
    }


@router.get("/analysis/{analysis_id}/deliveries/{delivery_id}")
def analysis_delivery(analysis_id: str, delivery_id: str) -> dict:
    """
    One delivery, complete and self-contained.

    Clip URL, overlay URL, shot, bowling, trajectory, bounce and validation for
    this delivery only. Switching between deliveries in a client means swapping
    this response wholesale, so fields from two ids can never be mixed.
    """
    # Parsed before the analysis lookup: a malformed id is malformed whether or
    # not the analysis exists, and reporting it as "no such analysis" sends the
    # caller looking in the wrong place.
    did = parse_delivery_id(delivery_id)
    doc = result_or_404(analysis_id)
    return find_delivery(doc, did, analysis_id)


@router.get("/analyses/{analysis_id}/deliveries/{delivery_id}")
def analysis_delivery_plural(analysis_id: str, delivery_id: int) -> dict:
    """Alias of `/analysis/{id}/deliveries/{delivery_id}`."""
    return analysis_delivery(analysis_id, str(delivery_id))


@router.get("/analysis/{analysis_id}/deliveries/{delivery_id}/trajectory")
def analysis_delivery_trajectory(analysis_id: str, delivery_id: str) -> dict:
    """
    The stored trajectory for one delivery — the pitch-map / chart endpoint.

    Reads `tracking.json`, the same persisted array the overlay renderer drew and
    the analytics were fitted to, so a plotted path and a rendered path cannot
    disagree. Includes a self-describing SVG pitch map when the geometry is
    calibrated, and `null` for the projected coordinates otherwise.
    """
    did = parse_delivery_id(delivery_id)
    doc = read_json(Paths(analysis_id=analysis_id).tracking_json)
    if doc is None:
        raise HTTPException(404, f"No tracking result for analysis {analysis_id}")
    for d in doc.get("deliveries", []):
        if d.get("delivery_id") == did:
            return {
                "analysis_id": analysis_id,
                "delivery_id": did,
                "clip": d.get("clip"),
                "bounce": d.get("bounce"),
                "bowling": d.get("bowling"),
                "trajectory": d.get("trajectory"),
                "tracking_source": d.get("tracking_source"),
                "quality": d.get("quality"),
                "calibration": doc.get("calibration"),
                "pitch_map": pitch_map_svg(d),
            }
    raise HTTPException(
        404, f"Delivery {did} not found in analysis {analysis_id}."
    )


@router.get("/analyses/{analysis_id}/trajectory/{delivery_id}")
def analysis_trajectory(analysis_id: str, delivery_id: int) -> dict:
    """Alias of `/analysis/{id}/deliveries/{id}/trajectory`."""
    return analysis_delivery_trajectory(analysis_id, str(delivery_id))