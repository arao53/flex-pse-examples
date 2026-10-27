"""Flex Connect + solar: build the flex-pse model from ``config.json``.

The plant. An electrically driven **aqueous process** -- a single
``ConstantEnergyIntensityModel``, so its draw is ``energy_intensity * flow`` --
takes water from a ``Feed`` and delivers it to a ``Product``. It is rated at
5 MW. When it runs it runs anywhere in 80-100% of rated, and once it starts or
stops it holds that state for at least an hour. Over the week it has to average
95% of rated output, which is the whole of its obligation: *when* it makes that
water is the schedule's to decide.

Beside it sit 1 MW of **solar** (``GenericRenewables``, its capacity factor a
GFS forecast run through hefty's PV model) and a **grid connection**
(``GridConnection``) on a PG&E **Flex Connect** service. Flex Connect is PG&E's
flexible service connection: rather than wait for a feeder upgrade, the customer
connects now and accepts an import limit PG&E sends a day ahead, hour by hour --
full service in most hours, a lower cap in the hours the local grid is
constrained. Export is not allowed.

The question the example asks is what that cap does to a plant that cannot turn
down below 80%. The cap here drops to 3.75 MW, under the process's 4 MW floor, so
in a curtailed hour the plant either runs on solar as well as grid, or it stops
-- and stopping costs it at least an hour of output it has to make up elsewhere
at full rate to hold its 95%.

**Where the curtailment falls.** The ten preset sites (``sites.json``) sit at the
ends of the PG&E transmission constraints CAISO's market monitor found most
congested in 2025, and the share of hours each site is curtailed is the share of
hours its constraint bound. Those constraints bound in **solar hours**
(hour-ending 9-17), so that is where the curtailed hours go -- nearest midday
first. Transmission congestion is a stand-in for the distribution constraints
Flex Connect actually manages; it puts the curtailment in a believable place,
nothing more. It also means the on-site solar is producing exactly when the cap
binds, which is what makes it worth more here than its kWh.

``flexops`` has no config-to-model bridge yet, so :func:`build_model` wires the
flowsheet by hand and is the only place config keys are read.

**Every balance is a flex-pse block.** The 95% obligation is the ``Product``'s
horizon-basis ``min_demand``, so it lands on the scalar ``delivery_total`` and
the solver shapes the profile. The import cap is ``GridConnection``'s per-period
``max_import`` (a length-T series), and ``max_export=0`` keeps the plant's net
draw non-negative, which is also what lets EECO bill it. The one hand-written
constraint lets the solar *curtail*: ``GenericRenewables`` pins generation to the
forecast exactly, which with export forbidden is infeasible in any step the
process must stop while the sun is up (see :func:`build_model`).

**The solar is load-bearing, not a bill credit.** With no solar, every curtailed
hour is an hour off (the 3.75 MW cap is under the 4 MW floor), and at the most
congested sites that alone puts 95% out of reach. With it, the process rides
through most curtailed hours at 80-90% on grid plus sun.

**The value of flexibility** (:func:`value_of_flexibility`) compares the plant
with an inflexible twin running flat at the same weekly output on firm service,
at utilization targets from 70% to 100% (:data:`TARGETS`): dollars saved per MWh
the flexible schedule moved in time.
"""

import json
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
from flexops import (
    ConstantEnergyIntensityModel,
    Feed,
    FlexCosting,
    GenericRenewables,
    GridConnection,
    PlantBlock,
    Product,
    SimpleAqueousFlow,
    TimeBlock,
)
from flexops.logic import add_startup_shutdown, add_status
from pyomo.environ import units as pyunits
from pyomo.network import Arc

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "config.json"
SITES_PATH = HERE / "sites.json"
CACHE_PATH = HERE / "data" / "solar_cf.parquet"

#: The site :func:`run_site` and the ``__main__`` block use when none is named.
DEFAULT_SITE = "moss_landing"

_M3_HR = pyunits.m**3 / pyunits.hr


#: The utilization targets the value curve walks, as fractions of rated. 0.95,
#: the config's default, is on the grid so the headline case is a point on it.
TARGETS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


class InfeasibleTarget(ValueError):
    """The Flex Connect cap and solar cannot carry the plant to its target.

    Raised before a model is built, from the :func:`max_utilization` screen.
    ``best`` is the screen's upper bound on reachable utilization.
    """

    def __init__(self, message: str, *, best: float):
        super().__init__(message)
        self.best = best


# -- config ------------------------------------------------------------------


def load_config(path=None) -> dict:
    """Load and validate the example's config.

    Raises:
        ValueError: If the horizon is not a whole number of steps, a minimum
            up/down time is not a whole number of steps, or the Flex Connect
            service cannot carry the process at rated load.
    """
    cfg = json.loads(Path(path or CONFIG_PATH).read_text())
    dt = cfg["time"]["time_step_hours"]
    span = (
        pd.Timestamp(cfg["time"]["end_date"]) - pd.Timestamp(cfg["time"]["start_date"])
    ) / pd.Timedelta(hours=1)
    if abs(span / dt - round(span / dt)) > 1e-9:
        raise ValueError(f"the horizon ({span} h) is not a whole number of {dt} h steps")
    process = cfg["process"]
    for key in ("min_up_hours", "min_down_hours"):
        steps = process[key] / dt
        if abs(steps - round(steps)) > 1e-9 or steps < 1:
            raise ValueError(
                f"process.{key}={process[key]} is not a whole number of "
                f"{dt} h steps"
            )
    if cfg["flex_connect"]["service_kw"] < process["rated_power_kw"]:
        raise ValueError(
            "flex_connect.service_kw is below the process's rated power, so the "
            "plant could never run flat out even in an uncurtailed hour."
        )
    return cfg


def load_sites() -> dict:
    """Return ``sites.json`` parsed."""
    return json.loads(SITES_PATH.read_text())


def site_by_id(site_id: str) -> dict:
    """Return one preset site's record from ``sites.json``."""
    for site in load_sites()["sites"]:
        if site["id"] == site_id:
            return site
    known = [s["id"] for s in load_sites()["sites"]]
    raise KeyError(f"unknown site {site_id!r}; expected one of {known}")


def rated_flow_m3_per_hr(cfg: dict) -> float:
    """Rated process flow: rated power over energy intensity (kW / kWh/m^3)."""
    process = cfg["process"]
    return process["rated_power_kw"] / process["energy_intensity_kwh_per_m3"]


def steps(cfg: dict, hours: float) -> int:
    """A duration in hours as a whole number of model steps."""
    return int(round(hours / cfg["time"]["time_step_hours"]))


def horizon_index(cfg: dict, horizon_days: float | None = None) -> pd.DatetimeIndex:
    """The model's time points as naive local timestamps.

    ``horizon_days`` shortens the horizon from the config's start date, which is
    how the smoke test and the CI sweep get a small instance of the same week.
    """
    time = cfg["time"]
    step = pd.Timedelta(hours=time["time_step_hours"])
    start = pd.Timestamp(time["start_date"])
    end = pd.Timestamp(time["end_date"])
    if horizon_days is not None:
        end = min(end, start + pd.Timedelta(days=horizon_days))
    return pd.date_range(start, end - step, freq=step)


# -- exogenous series: solar and the Flex Connect cap ------------------------


def capacity_factor(site_id: str, index: pd.DatetimeIndex) -> pd.Series:
    """A preset site's cached solar capacity factor over ``index``.

    Reads ``data/solar_cf.parquet``, which ``sites.py`` writes from a hefty GFS
    forecast. Offline by design: neither the sweep nor the tests download
    anything.
    """
    import duckdb

    frame = (
        duckdb.sql(f"SELECT * FROM read_parquet('{CACHE_PATH}') ORDER BY timestamp")
        .df()
        .set_index("timestamp")
    )
    if site_id not in frame.columns:
        raise KeyError(f"no cached capacity factor for site {site_id!r}")
    series = frame[site_id].reindex(index)
    if series.isna().any():
        raise ValueError(
            f"the cached capacity factor does not cover the horizon "
            f"{index[0]} -> {index[-1]}; rebuild it with sites.py"
        )
    return series


def curtailed_hours(cfg: dict, congestion_share: float, index: pd.DatetimeIndex):
    """The hour-starts Flex Connect curtails, for a site with this congestion share.

    The number of curtailed hours is ``congestion_share`` of the horizon's hours,
    rounded. They are drawn from the constraint's binding window
    (``sites.json`` ``binding_hours_ending``, hour-ending 9-17 -- so hour-starts
    8-16), nearest midday first and spread across the days, so a 24% site loses
    the middle of most days and a 1% site loses one midday hour in the week.

    Returns:
        A sorted list of naive local hour-start timestamps.
    """
    he_first, he_last = load_sites()["binding_hours_ending"]
    window = range(he_first - 1, he_last)  # hour-starts
    hours = pd.date_range(index[0].floor("h"), index[-1].floor("h"), freq="h")
    candidates = [h for h in hours if h.hour in window]
    n_curtailed = min(len(candidates), round(congestion_share * len(hours)))
    midday = (he_first - 1 + he_last) / 2  # centre of the window, in hour-starts
    ranked = sorted(
        candidates, key=lambda h: (abs(h.hour + 0.5 - midday), h.hour, h.normalize())
    )
    return sorted(ranked[:n_curtailed])


def flex_connect_limits(
    cfg: dict, congestion_share: float, index: pd.DatetimeIndex
) -> pd.Series:
    """The synthetic day-ahead Flex Connect import cap, in kW, per model step.

    ``service_kw`` in every hour but the curtailed ones (:func:`curtailed_hours`),
    which get ``curtailed_kw``. PG&E sends hourly limits, so each hour's cap is
    held across that hour's steps.
    """
    fc = cfg["flex_connect"]
    curtailed = set(curtailed_hours(cfg, congestion_share, index))
    values = [
        fc["curtailed_kw"] if t.floor("h") in curtailed else fc["service_kw"]
        for t in index
    ]
    return pd.Series(values, index=index, name="import_limit_kw")


LMP_PATH = HERE / "data" / "lmp.parquet"


def site_lmps(site_id: str) -> pd.Series:
    """A preset site's cached day-ahead hourly LMP over the horizon, $/MWh.

    Reads ``data/lmp.parquet``, which ``sites.py --lmp`` writes from CAISO
    OASIS for the site's ``pnode``. Indexed by naive local hour-start.
    """
    import duckdb

    frame = (
        duckdb.sql(f"SELECT * FROM read_parquet('{LMP_PATH}') ORDER BY timestamp")
        .df()
        .set_index("timestamp")
    )
    return frame[site_id]


def lmp_band(
    cfg: dict, site_id: str, index: pd.DatetimeIndex | None = None
) -> dict:
    """The site's LMP across its Flex Connect-curtailed hours: P10, median, P90.

    This is the *price* side of the flexibility market the value curve is the
    other side of: what the grid was paying for energy at that node in exactly
    the hours Flex Connect asked the plant to take less of it. Where the value
    curve sits above the band, a MWh of flexibility is worth more to the plant
    than the energy it moves.

    Returns:
        ``{"lmp_p10", "lmp_p50", "lmp_p90", "lmp_hours"}``; the percentiles are
        ``nan`` if the site has no curtailed hours in ``index``.
    """
    if index is None:
        index = horizon_index(cfg)
    hours = curtailed_hours(cfg, site_by_id(site_id)["congestion_share"], index)
    prices = site_lmps(site_id).reindex(hours).dropna()
    quantiles = prices.quantile([0.1, 0.5, 0.9]) if len(prices) else None
    return {
        "lmp_p10": float(quantiles[0.1]) if quantiles is not None else float("nan"),
        "lmp_p50": float(quantiles[0.5]) if quantiles is not None else float("nan"),
        "lmp_p90": float(quantiles[0.9]) if quantiles is not None else float("nan"),
        "lmp_hours": len(prices),
    }


def max_utilization(cfg: dict, cf: pd.Series, limits: pd.Series) -> float:
    """An upper bound on the utilization the plant can reach under this cap.

    In each step the process can draw at most the import cap plus what the solar
    makes, up to rated; if that is under its 80% floor it cannot run at all. The
    minimum up/down times are ignored, so the true best is somewhat lower -- this
    is a fast screen for a hopeless instance, not a feasibility proof.
    """
    process = cfg["process"]
    rated = process["rated_power_kw"]
    floor = process["min_load_fraction"] * rated
    available = (limits + cf * cfg["solar"]["nameplate_ac_kw"]).clip(upper=rated)
    available = available.where(available >= floor, 0.0)
    return float(available.mean() / rated)


# -- the model ---------------------------------------------------------------


def build_model(
    cfg: dict,
    site: str | None = DEFAULT_SITE,
    *,
    horizon_days: float | None = None,
    solar_cf: pd.Series | None = None,
    congestion_share: float | None = None,
    solar_kw: float | None = None,
    utilization_target: float | None = None,
    flexible: bool = True,
):
    """Build the Flex Connect + solar model for one site.

    ``flexible=False`` builds the **inflexible baseline** the value curve is
    measured against (:func:`value_of_flexibility`): the process pinned flat at
    ``utilization_target`` of rated in every step, with no on/off band and no
    minimum up/down times, on a *firm* grid service -- no Flex Connect cap,
    since a plant that cannot move its load is exactly the one that would buy
    firm service. It has no degrees of freedom; it is solved only so EECO prices
    it on the same tariff.

    Args:
        cfg: The loaded config (see :func:`load_config`).
        site: A preset site id from ``sites.json``. Supplies the cached solar
            capacity factor and the congestion share, unless those are passed
            explicitly.
        horizon_days: Build only the first ``horizon_days`` of the week.
        solar_cf: A capacity-factor series over the horizon, overriding the
            cached one -- how notebook.py feeds a live hefty forecast for a
            typed-in location.
        congestion_share: Override the site's congestion share.
        solar_kw: Override the solar nameplate (AC kW); 0 builds the plant with
            no solar at all.
        utilization_target: Override ``process.utilization_target`` -- the
            horizon-average output as a fraction of rated.
        flexible: ``False`` for the inflexible baseline described above.

    Returns:
        A built ``pyo.ConcreteModel`` with ``plant``, ``costing`` and
        ``objective`` (the operating cost), plus ``m.inputs`` -- a DataFrame of
        the exogenous series the model was built from.

    Raises:
        ValueError: If even the optimistic screen in :func:`max_utilization`
            says the utilization target cannot be met.
    """
    process = cfg["process"]
    solar = cfg["solar"]
    fc = cfg["flex_connect"]

    index = horizon_index(cfg, horizon_days)
    if solar_cf is None:
        solar_cf = capacity_factor(site, index)
    else:
        solar_cf = pd.Series(list(solar_cf)[: len(index)], index=index)
    if congestion_share is None:
        congestion_share = site_by_id(site)["congestion_share"]
    nameplate = solar["nameplate_ac_kw"] if solar_kw is None else float(solar_kw)
    limits = flex_connect_limits(cfg, congestion_share, index)
    target = (
        process["utilization_target"]
        if utilization_target is None
        else float(utilization_target)
    )

    best = max_utilization(
        cfg, solar_cf * (nameplate / solar["nameplate_ac_kw"]), limits
    )
    if flexible and best < target - 1e-9:
        raise InfeasibleTarget(
            f"at site {site!r} the Flex Connect cap and solar allow at most "
            f"{best:.1%} utilization, under the {target:.0%} target. Raise "
            "flex_connect.curtailed_kw or the solar nameplate.",
            best=best,
        )

    dt = cfg["time"]["time_step_hours"]
    end = index[-1] + pd.Timedelta(hours=dt)
    m = pyo.ConcreteModel(name=cfg["name"])
    m.time_block = TimeBlock(
        start_date=index[0].isoformat(),
        end_date=end.isoformat(),
        time_step=dt * pyunits.hr,
    )
    tb = m.time_block
    m.properties = SimpleAqueousFlow(
        density=cfg["fluid"]["density_kg_per_m3"] * pyunits.kg / pyunits.m**3,
    )
    m.costing = FlexCosting(time_block=tb, tariff=cfg["tariff"])
    m.plant = PlantBlock(time_block=tb)
    plant = m.plant

    rated_flow = rated_flow_m3_per_hr(cfg)
    horizon_hours = len(index) * dt

    plant.water = Feed(property_package=m.properties, resource_name="water")
    plant.process = ConstantEnergyIntensityModel(
        property_package=m.properties,
        energy_intensity=process["energy_intensity_kwh_per_m3"]
        * pyunits.kWh
        / pyunits.m**3,
        costing_package=m.costing,
    )
    # The utilization target, as a horizon-basis limit on the product boundary:
    # 95% of rated output over the week, as a volume. It bounds the scalar
    # delivery_total, not any one step, so the schedule is free to shape it.
    plant.product = Product(
        property_package=m.properties,
        resource_name="product",
        min_demand=target * rated_flow * horizon_hours * pyunits.m**3,
        demand_basis="horizon",
    )
    plant.water_to_process = Arc(
        source=plant.water.outlet_a, destination=plant.process.inlet
    )
    plant.process_to_product = Arc(
        source=plant.process.outlet, destination=plant.product.inlet_a
    )
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    if flexible:
        # Off, or anywhere in [min_load_fraction, 1] of rated -- the
        # semicontinuous band that makes this a MILP. Then a minimum run and
        # rest of one hour each.
        add_status(
            plant.process,
            plant.process.flow_in,
            process["min_load_fraction"] * rated_flow * _M3_HR,
            rated_flow * _M3_HR,
        )
        add_startup_shutdown(
            plant.process,
            plant.process.status,
            min_uptime=steps(cfg, process["min_up_hours"]),
            min_downtime=steps(cfg, process["min_down_hours"]),
        )
    else:
        # The baseline: flat at the target in every step. A reference profile
        # for pricing, not a schedule the plant could run below its 80% floor.
        for t in tb.time_index:
            plant.process.flow_in[t].fix(target * rated_flow)

    if nameplate > 0:
        plant.solar = GenericRenewables(
            capacity=nameplate * pyunits.kW,
            capacity_factor=solar_cf.clip(0.0, 1.0).tolist(),
            technology="solar",
        )
        # Let the inverters curtail. GenericRenewables pins generation to
        # exactly capacity_factor * capacity, which with export forbidden makes
        # any step infeasible where the process has to stop but the sun is up --
        # and a curtailed Flex Connect hour under cloud is exactly that step (cap
        # plus solar below the 4 MW floor). A real plant spills the solar, so
        # the equality becomes <= here. Hand-written because GenericRenewables
        # has no curtailment option yet; it is the one constraint in this model
        # that is not a flex-pse block's.
        plant.solar.power_electrical_relation.deactivate()
        _cf = dict(zip(tb.time_index, solar_cf.clip(0.0, 1.0).tolist(), strict=True))

        @plant.Constraint(
            tb.time_index,
            doc="Solar generation is at most the forecast; the rest is curtailed.",
        )
        def solar_available(b, t):
            return plant.solar.power_generated[t] <= _cf[t] * plant.solar.capacity

    # Built on the plant, after every unit that draws or makes power: it ties
    # grid_import - grid_export to the plant's net electrical draw.
    plant.grid = GridConnection(
        max_import=[v * pyunits.kW for v in limits] if flexible else None,
        max_export=fc["max_export_kw"] * pyunits.kW,
    )

    m.costing.cost_process()
    m.objective = pyo.Objective(
        expr=m.costing.aggregate_operating_cost, sense=pyo.minimize
    )

    m.site = site
    m.congestion_share = congestion_share
    m.solar_nameplate_kw = nameplate
    m.utilization_target = target
    m.max_utilization = best
    m.is_flexible = flexible
    m.inputs = pd.DataFrame(
        {
            "solar_cf": solar_cf.to_numpy(),
            "import_limit_kw": limits.to_numpy(),
        },
        index=index,
    )
    return m


#: Per-solver options :func:`solve_model` passes, keyed on the solver
#: ``get_solver`` returned. A week at 15 minutes is ~2k binaries; both close it
#: in seconds to a minute, and the gap is small enough to read costs off.
SOLVER_OPTIONS = {
    "appsi_highs": {"mip_rel_gap": 1e-4, "time_limit": 600},
    "highs": {"mip_rel_gap": 1e-4, "time_limit": 600},
    "gurobi": {"MIPGap": 1e-4, "TimeLimit": 600},
    "gurobi_persistent": {"MIPGap": 1e-4, "TimeLimit": 600},
}


def solve_model(m, *, prefer: str = "highs"):
    """Solve ``m`` and assert an optimal termination.

    Records the relative MIP gap on ``m.mip_gap`` and the solver's name on
    ``m.solver_name``, and snaps power noise to zero -- EECO refuses to bill a
    draw of minus a few nanowatts.
    """
    from flexcore.solvers import get_solver
    from pyomo.opt import assert_optimal_termination

    solver = get_solver(model=m, prefer=prefer)
    results = solver.solve(m, options=dict(SOLVER_OPTIONS.get(solver.name, {})))
    assert_optimal_termination(results)
    lower, upper = results.problem.lower_bound, results.problem.upper_bound
    m.mip_gap = (
        0.0
        if lower is None or upper is None or not upper
        else abs(upper - lower) / abs(upper)
    )
    m.solver_name = solver.name
    for entry in m.costing.aggregate_power.values():
        if entry.value is not None and abs(entry.value) < 1e-6:
            entry.set_value(0.0, skip_validation=True)
    return results


def results_frame(m, cfg: dict) -> pd.DataFrame:
    """Return the solved schedule as a DataFrame indexed by local timestamp."""
    from flexops.costing import load_tariff, price_series

    tb = m.time_block
    ti = list(tb.time_index)
    plant = m.plant
    rated_kw = cfg["process"]["rated_power_kw"]
    has_solar = plant.component("solar") is not None

    process_kw = [pyo.value(plant.process.power_electrical[t]) for t in ti]
    data = {
        "process_kw": process_kw,
        "process_load": [kw / rated_kw for kw in process_kw],
        # The baseline has no status binary; it is on by construction.
        "process_status": (
            [round(pyo.value(plant.process.status[t])) for t in ti]
            if plant.process.component("status") is not None
            else [1] * len(ti)
        ),
        "solar_kw": (
            [pyo.value(plant.solar.power_generated[t]) for t in ti]
            if has_solar
            else [0.0] * len(ti)
        ),
        "grid_import_kw": [pyo.value(plant.grid.grid_import[t]) for t in ti],
        "grid_export_kw": [pyo.value(plant.grid.grid_export[t]) for t in ti],
        # The firm-service baseline has no cap: report it from the inputs, so
        # the column still says what Flex Connect *would* have allowed.
        "import_limit_kw": m.inputs["import_limit_kw"].to_list(),
    }
    frame = pd.DataFrame(data, index=tb.datetime_index)
    frame.index.name = "timestamp"
    frame = frame.mask(frame.abs() < 1e-6, 0.0)
    frame["curtailed"] = (
        frame["import_limit_kw"] < cfg["flex_connect"]["service_kw"]
    ).astype(int)
    frame["energy_price"] = price_series(
        load_tariff(cfg["tariff"]), tb.datetime_index
    ).to_numpy()
    return frame


def summarize(m, cfg: dict, frame: pd.DataFrame) -> dict:
    """The scalars a sweep row or a notebook table reports for one solve."""
    report = m.costing.report_cost(m)
    dt = cfg["time"]["time_step_hours"]
    peak_hours = frame.index.hour.isin(range(16, 21))
    solar_kwh = frame["solar_kw"].sum() * dt
    process_kwh = frame["process_kw"].sum() * dt
    curtailed = frame["curtailed"] == 1
    available_kwh = (m.inputs["solar_cf"] * m.solar_nameplate_kw).sum() * dt
    return {
        "solar_spilled_kwh": float(max(available_kwh - solar_kwh, 0.0)),
        "operating_cost": float(report.operating.total),
        "electricity_cost": float(report.operating.electricity),
        "utilization": float(frame["process_load"].mean()),
        "peak_kw": float(frame["grid_import_kw"].max()),
        "peak_window_kw": float(frame.loc[peak_hours, "grid_import_kw"].max()),
        "grid_kwh": float(frame["grid_import_kw"].sum() * dt),
        "solar_kwh": float(solar_kwh),
        "solar_share": float(solar_kwh / process_kwh) if process_kwh else 0.0,
        "curtailed_hours": float(curtailed.sum() * dt),
        "curtailed_load": (
            float(frame.loc[curtailed, "process_load"].mean()) if curtailed.any() else None
        ),
        "starts": int((frame["process_status"].diff() == 1).sum()),
        "hours_off": float((frame["process_status"] == 0).sum() * dt),
    }


def run_site(cfg: dict, site: str = DEFAULT_SITE, **kwargs) -> dict:
    """Build, solve and summarize one site. ``kwargs`` go to :func:`build_model`."""
    m = build_model(cfg, site, **kwargs)
    solve_model(m)
    frame = results_frame(m, cfg)
    return {"site": site, "model": m, "frame": frame, **summarize(m, cfg, frame)}


# -- the value of flexibility ------------------------------------------------


def mwh_shifted(flexible: pd.DataFrame, baseline: pd.DataFrame, dt_hours: float) -> float:
    """Energy the flexible schedule moved in time relative to the flat baseline, MWh.

    Half the summed absolute gap between the two process-load profiles: every
    MWh taken out of one step and put into another shows up once as a shortfall
    and once as a surplus, so the half counts it once. Both profiles deliver the
    same weekly output, so this is pure re-timing, not curtailment.
    """
    gap = (flexible["process_kw"] - baseline["process_kw"]).abs().sum()
    return float(gap * dt_hours / 2 / 1000)


def value_of_flexibility(
    cfg: dict, site: str = DEFAULT_SITE, target: float | None = None, **kwargs
) -> dict:
    """One point on a site's flexibility value curve.

    Solves the plant twice at the same utilization target -- the flexible
    schedule on Flex Connect, and the inflexible flat baseline on firm service
    (``build_model(flexible=False)``) -- and prices the difference.

    Returns:
        A dict with the target, both operating costs, ``savings`` (baseline
        minus flexible, $ over the horizon), ``mwh_shifted``
        (:func:`mwh_shifted`) and ``value_per_mwh`` (savings / MWh shifted), plus
        the flexible run's :func:`summarize` scalars and both frames. A target
        the cap and solar cannot reach comes back with ``feasible=False``, the
        screen's ``max_utilization``, and no costs -- it is a point on the curve
        too, the one that says where flexibility runs out.
    """
    dt = cfg["time"]["time_step_hours"]
    target = cfg["process"]["utilization_target"] if target is None else target
    point = {"site": site, "utilization_target": float(target)}
    try:
        m = build_model(cfg, site, utilization_target=target, **kwargs)
    except InfeasibleTarget as exc:
        return {**point, "feasible": False, "max_utilization": exc.best}
    solve_model(m)
    frame = results_frame(m, cfg)

    base = build_model(
        cfg, site, utilization_target=target, flexible=False, **kwargs
    )
    solve_model(base)
    base_frame = results_frame(base, cfg)

    flex_cost = float(m.costing.report_cost(m).operating.total)
    base_cost = float(base.costing.report_cost(base).operating.total)
    shifted = mwh_shifted(frame, base_frame, dt)
    savings = base_cost - flex_cost
    return {
        **point,
        "feasible": True,
        "max_utilization": m.max_utilization,
        "baseline_cost": base_cost,
        "flexible_cost": flex_cost,
        "savings": savings,
        "mwh_shifted": shifted,
        "value_per_mwh": savings / shifted if shifted > 1e-6 else float("nan"),
        **summarize(m, cfg, frame),
        "operating_cost": flex_cost,
        "model": m,
        "frame": frame,
        "baseline_frame": base_frame,
    }


def value_curve(cfg: dict, site: str = DEFAULT_SITE, targets=TARGETS, **kwargs):
    """A site's whole flexibility value curve, one row per utilization target.

    Returns:
        A DataFrame indexed by ``utilization_target`` with the scalar columns of
        :func:`value_of_flexibility` (models and frames dropped).
    """
    rows = []
    for target in targets:
        point = value_of_flexibility(cfg, site, target, **kwargs)
        rows.append(
            {k: v for k, v in point.items() if k not in ("model", "frame", "baseline_frame")}
        )
    return pd.DataFrame(rows).set_index("utilization_target")


if __name__ == "__main__":
    import sys

    config = load_config()
    site_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SITE
    curve = value_curve(config, site_id)
    columns = [
        "feasible", "max_utilization", "baseline_cost", "flexible_cost",
        "savings", "mwh_shifted", "value_per_mwh", "solar_share", "hours_off",
    ]
    print(f"site {site_id}: flexibility value curve")
    print(curve[columns].round(3).to_string())
