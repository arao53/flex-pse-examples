"""Sweep adapter for flex_connect_solar. Driven by `tools/sweep.py`.

Two dimensions, flattened into one list of points: every preset site in
``sites.json`` crossed with every utilization target in :data:`model.TARGETS`
(70-100%, the config's 95% among them). Each point is one point on that site's
**flexibility value curve** (:func:`model.value_of_flexibility`): the flexible
schedule on Flex Connect and the flat baseline on firm service, both solved,
and the saving between them priced per MWh shifted.

A target the Flex Connect cap and the solar cannot reach at a site is left out
of the points rather than solved -- 100% is out everywhere, since a curtailed
hour allows at most 3.75 MW from the grid plus 1 MW of solar against a 5 MW
process. Every row carries the site's ``max_utilization`` so the page can still
say where each curve ends.

Each point is two week-long solves on HiGHS, a few seconds apiece. The solar
resource is the committed ``data/solar_cf.parquet``, so this runs offline.
"""

import model

#: The reduced sweep CI runs: the most and least congested sites, the default
#: target and one below the process's 80% floor, over two days.
SMOKE_SITES = ("moss_landing", "metcalf")
SMOKE_TARGETS = (0.75, 0.95)
SMOKE_DAYS = 2


def _reachable(cfg: dict, site: dict, days) -> float:
    """The :func:`model.max_utilization` screen for one site."""
    index = model.horizon_index(cfg, days)
    return model.max_utilization(
        cfg,
        model.capacity_factor(site["id"], index),
        model.flex_connect_limits(cfg, site["congestion_share"], index),
    )


def points(*, smoke: bool = False) -> list[dict]:
    """Every (site, target) pair the screen says is reachable, most congested first."""
    cfg = model.load_config()
    days = SMOKE_DAYS if smoke else None
    sites = sorted(model.load_sites()["sites"], key=lambda s: -s["congestion_share"])
    targets = SMOKE_TARGETS if smoke else model.TARGETS
    if smoke:
        sites = [s for s in sites if s["id"] in SMOKE_SITES]
    out = []
    for site in sites:
        best = _reachable(cfg, site, days)
        for target in targets:
            if target > best + 1e-9:
                continue
            out.append(
                {
                    "label": f"{site['name']} @ {target:.0%}",
                    "site": site["id"],
                    "utilization_target": target,
                }
            )
    return out


def setup(*, smoke: bool = False) -> dict:
    """Load the config once and remember the horizon this run uses."""
    return {"cfg": model.load_config(), "days": SMOKE_DAYS if smoke else None}


def solve_point(ctx: dict, point: dict):
    """Solve one point on one site's value curve.

    Returns:
        ``(frame, summary)`` -- the flexible schedule (with the baseline's flat
        process load alongside it as ``baseline_kw``), and the scalars that
        become this point's summary row.
    """
    cfg = ctx["cfg"]
    site = model.site_by_id(point["site"])
    result = model.value_of_flexibility(
        cfg, point["site"], point["utilization_target"], horizon_days=ctx["days"]
    )
    frame = result["frame"].copy()
    band = model.lmp_band(cfg, point["site"], frame.index)
    frame["baseline_kw"] = result["baseline_frame"]["process_kw"]
    m = result["model"]
    curtailed_load = result["curtailed_load"]
    return frame, {
        "site": point["site"],
        "site_name": site["name"],
        "lat": site["lat"],
        "lon": site["lon"],
        "congestion_share": site["congestion_share"],
        "max_utilization": result["max_utilization"],
        "objective": result["flexible_cost"],
        "operating_cost": result["flexible_cost"],
        "baseline_cost": result["baseline_cost"],
        "savings": result["savings"],
        "mwh_shifted": result["mwh_shifted"],
        "value_per_mwh": result["value_per_mwh"],
        "utilization": result["utilization"],
        "peak_kw": result["peak_kw"],
        "peak_window_kw": result["peak_window_kw"],
        "grid_kwh": result["grid_kwh"],
        "solar_kwh": result["solar_kwh"],
        "solar_share": result["solar_share"],
        "curtailed_hours": result["curtailed_hours"],
        "curtailed_load": float("nan") if curtailed_load is None else curtailed_load,
        "starts": result["starts"],
        "hours_off": result["hours_off"],
        **band,
        "solver": m.solver_name,
        "termination": "optimal",  # solve_model asserts it, so anything else raised
        "mip_gap": float(m.mip_gap),
    }
