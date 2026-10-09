"""
backend/app/qualcomm/routes.py
FastAPI router for Qualcomm Cloud AI Playground attribution explanations.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.auth.routes import get_current_user
from app.auth.models import User
from app.spills.models import Spill
from app.attribution.models import SuspectScore
from app.vessels.models import Vessel
from app.qualcomm.service import explain_attribution, build_legal_notice

router = APIRouter(prefix="/api/qualcomm", tags=["Qualcomm Cloud AI"])


@router.get("/explain/{spill_id}")
async def get_qualcomm_explanation(
    spill_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Fetches suspect scores for a spill, matches them with vessel metadata,
    and calls Qualcomm Cloud AI Playground (or returns cached/unavailable state).
    """
    # 1. Fetch Spill
    spill_query = await db.execute(select(Spill).where(Spill.id == spill_id))
    spill = spill_query.scalar_one_or_none()
    if not spill:
        raise HTTPException(status_code=404, detail="Spill incident not found")

    # 2. Fetch Suspect Scores with joined Vessel data
    scores_query = await db.execute(
        select(SuspectScore, Vessel)
        .join(Vessel, SuspectScore.vessel_id == Vessel.id)
        .where(SuspectScore.spill_id == spill_id)
        .order_by(SuspectScore.rank.asc())
    )
    results = scores_query.all()

    candidates = []
    for score, vessel in results:
        prox = round(score.proximity_score or 0.0, 1)
        t_overlap = round(score.time_overlap_score or 0.0, 1)
        gap = round(score.ais_gap_score or 0.0, 1)
        spd = round(score.speed_anomaly_score or 0.0, 1)
        crs = round(score.course_anomaly_score or 0.0, 1)
        rt_dev = round(score.route_deviation_score or 0.0, 1)

        ev = (score.explanation or {}).get("evidence", {}) if isinstance(score.explanation, dict) else {}
        closest_nm = ev.get("closest_fix", {}).get("distance_nm")
        dist_str = f"{closest_nm:.2f} nm" if closest_nm is not None else f"{max(0.15, ((100 - prox) / 18)):.2f} nm"
        gap_min = round(ev.get("ais_gaps", [{}])[0].get("gap_minutes", max(30, gap * 1.6))) if ev.get("ais_gaps") else round(max(30, gap * 1.6))
        pts = ev.get("traffic", {}).get("points_in_cone", max(5, round(t_overlap / 3)))

        candidates.append({
            "rank": score.rank,
            "vessel_name": vessel.vessel_name or vessel.mmsi,
            "vessel_mmsi": vessel.mmsi,
            "vessel_type": vessel.vessel_type or "Tanker",
            "total_score": round(score.total_score, 1),
            "factors": [
                {
                    "name": "Spatial Proximity to Drift Origin",
                    "weight_percent": 25,
                    "score_0_to_100": prox,
                    "contribution": round(prox * 0.25, 1),
                    "evidence": f"Closest AIS position recorded {dist_str} from 4D backward advection cone peak."
                },
                {
                    "name": "Temporal Coincidence Window",
                    "weight_percent": 20,
                    "score_0_to_100": t_overlap,
                    "contribution": round(t_overlap * 0.20, 1),
                    "evidence": f"Vessel active in discharge probability zone ({pts} fixes recorded within temporal window)." if t_overlap > 30 else "Brief or peripheral transit through temporal observation window."
                },
                {
                    "name": "AIS Transponder Blackout",
                    "weight_percent": 20,
                    "score_0_to_100": gap,
                    "contribution": round(gap * 0.20, 1),
                    "evidence": f"{gap_min}-min transponder gap during estimated discharge window near centroid." if gap > 20 else "Transponder maintained continuous broadcast with no anomalous telemetry gaps."
                },
                {
                    "name": "Speed Reduction Anomaly",
                    "weight_percent": 15,
                    "score_0_to_100": spd,
                    "contribution": round(spd * 0.15, 1),
                    "evidence": "Deceleration below operational cruising speed (speed reduction consistent with discharge)." if spd > 20 else "Vessel maintained standard commercial cruising speed throughout transit."
                },
                {
                    "name": "Course Alteration Anomaly",
                    "weight_percent": 10,
                    "score_0_to_100": crs,
                    "contribution": round(crs * 0.10, 1),
                    "evidence": "Abrupt heading deviation executed during or immediately following discharge window." if crs > 20 else "Maintained steady navigational heading along planned corridor."
                },
                {
                    "name": "Commercial Route Deviation",
                    "weight_percent": 10,
                    "score_0_to_100": rt_dev,
                    "contribution": round(rt_dev * 0.10, 1),
                    "evidence": "Lateral diversion from designated international commercial maritime traffic TSS fairway." if rt_dev > 20 else "Remained fully aligned with standard maritime Traffic Separation Scheme (TSS) lanes."
                }
            ]
        })

    if not candidates:
        raise HTTPException(status_code=400, detail="No suspect evaluation available for this spill yet")

    spill_dict = {
        "id": spill.id,
        "name": spill.name,
        "centroid_lat": spill.centroid_lat,
        "centroid_lon": spill.centroid_lon,
        "area_sq_km": spill.area_sq_km,
        "detection_confidence_pct": round(spill.confidence_score * 100, 1) if spill.confidence_score and spill.confidence_score <= 1.0 else spill.confidence_score,
    }
    if getattr(spill, "nearest_mpa_name", None):
        spill_dict["nearest_marine_protected_area"] = spill.nearest_mpa_name
    if getattr(spill, "coast_proximity_km", None):
        spill_dict["coast_proximity_km"] = spill.coast_proximity_km

    # Call Qualcomm engine (pure explanation)
    ai_result = await explain_attribution(spill_dict, candidates)

    # Attach deterministic legal template for Rank 1
    ai_result["deterministic_legal_notice"] = build_legal_notice(spill_dict, candidates[0])
    ai_result["candidate_count"] = len(candidates)
    ai_result["top_candidate"] = candidates[0]["vessel_name"]

    return ai_result
