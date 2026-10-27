"""The Flex Connect + solar example still solves, and still says what it claims.

Runs on a free CI runner: two days at 15 minutes, HiGHS, a few seconds a solve.
The solar resource is the committed ``data/solar_cf.parquet``, so nothing here
touches the network -- hefty, and the GFS download behind it, are exercised only
by ``sites.py``, which no test runs. A broken forecast fetch therefore shows up
when someone rebuilds the cache, not here.

The assertions are about the *claims*: the plant hits its weekly output, stays
in its 80-100% band, respects the Flex Connect cap every step, never runs or
rests for less than an hour, gets something out of its solar, and -- at the
default 95% target -- earns money by moving load rather than running flat.
"""

from itertools import pairwise

import pytest

pytest.importorskip("flexops", reason="needs the flex-pse environment")

from conftest import EXAMPLES_ROOT, load_example_module

EXAMPLE = EXAMPLES_ROOT / "flex_connect_solar"
DAYS = 2
SITE = "moss_landing"  # the most congested: the most curtailed hours to test


@pytest.fixture(scope="module")
def model():
    """The example's ``model`` module, under a name unique to this example."""
    return load_example_module(EXAMPLE)


@pytest.fixture(scope="module")
def cfg(model):
    return model.load_config()


@pytest.fixture(scope="module")
def point(model, cfg):
    """The default-target point on Moss Landing's value curve, solved once."""
    return model.value_of_flexibility(cfg, SITE, horizon_days=DAYS)


@pytest.fixture(scope="module")
def frame(point):
    return point["frame"]


def test_meets_the_utilization_target(point, cfg):
    target = cfg["process"]["utilization_target"]
    assert point["feasible"]
    assert point["utilization"] >= target - 1e-6


def test_load_stays_in_its_band(frame, cfg):
    rated = cfg["process"]["rated_power_kw"]
    floor = cfg["process"]["min_load_fraction"] * rated
    on = frame["process_status"] == 1
    assert (frame.loc[~on, "process_kw"].abs() < 1e-3).all()
    assert (frame.loc[on, "process_kw"] >= floor - 1e-3).all()
    assert (frame["process_kw"] <= rated + 1e-3).all()


def test_flex_connect_cap_holds_every_step(frame):
    assert (frame["grid_import_kw"] <= frame["import_limit_kw"] + 1e-3).all()
    assert (frame["grid_export_kw"].abs() < 1e-3).all()
    assert frame["curtailed"].sum() > 0, "Moss Landing should see curtailed hours"


def test_no_run_or_rest_shorter_than_an_hour(frame, model, cfg):
    min_steps = model.steps(cfg, cfg["process"]["min_up_hours"])
    status = frame["process_status"].tolist()
    runs, length = [], 1
    for prev, cur in pairwise(status):
        if cur == prev:
            length += 1
        else:
            runs.append(length)
            length = 1
    # The first and last runs are cut by the horizon, not by the plant.
    for interior in runs[1:]:
        assert interior >= min_steps


def test_flexibility_pays_at_the_default_target(point):
    """The example's headline: moving load is worth money at 95%."""
    assert point["mwh_shifted"] > 0
    assert point["savings"] > 0
    assert point["value_per_mwh"] > 0


def test_solar_is_what_makes_the_target_reachable(model, cfg):
    """At the most congested site, 95% is out of reach without the solar.

    The curtailed cap (3.75 MW) is under the process's 4 MW floor, so with no
    solar every curtailed hour is an hour off -- a quarter of the midday hours
    at Moss Landing, which is more than a 95% target can absorb.
    """
    with pytest.raises(model.InfeasibleTarget):
        model.build_model(cfg, SITE, horizon_days=DAYS, solar_kw=0)


def test_solar_lowers_the_bill(model, cfg):
    """Where the target is reachable either way, the solar still pays."""
    site = "metcalf"  # the least congested: feasible with or without solar
    with_solar = model.build_model(cfg, site, horizon_days=DAYS)
    no_solar = model.build_model(cfg, site, horizon_days=DAYS, solar_kw=0)
    costs = []
    for m in (with_solar, no_solar):
        model.solve_model(m)
        costs.append(m.costing.report_cost(m).operating.total)
    assert costs[0] < costs[1]


def test_full_output_is_out_of_reach(model, cfg):
    """100% cannot be met under any curtailment; the curve says so, not the solver."""
    full = model.value_of_flexibility(cfg, SITE, 1.0, horizon_days=DAYS)
    assert not full["feasible"]
    assert full["max_utilization"] < 1.0


def test_lmp_cache_covers_every_node(model, cfg):
    """The committed OASIS prices cover the horizon at all ten pricing nodes."""
    hours = model.horizon_index(cfg).floor("h").unique()
    for site in model.load_sites()["sites"]:
        prices = model.site_lmps(site["id"])
        assert prices.reindex(hours).notna().all(), site["id"]


def test_lmp_band_is_over_the_curtailed_hours(model, cfg):
    band = model.lmp_band(cfg, SITE)
    index = model.horizon_index(cfg)
    assert band["lmp_hours"] == len(
        model.curtailed_hours(cfg, model.site_by_id(SITE)["congestion_share"], index)
    )
    assert band["lmp_p10"] <= band["lmp_p50"] <= band["lmp_p90"]
