"""Flex Connect + solar: builds and solves the model locally."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    try:
        here = Path(__file__).parent
    except NameError:  # pragma: no cover - interactive fallback
        here = Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    import diagram
    import model
    import sites

    from flex_pse_examples.diagrams import render as render_diagram

    return diagram, mo, model, pd, plt, render_diagram, sites


@app.cell
def _(mo):
    mo.md(r"""
    # Flex Connect + solar

    **What is a MWh of load shifting worth to a plant on a flexible grid connection?**

    A 5 MW electrically driven aqueous process -- one `ConstantEnergyIntensityModel`,
    power proportional to flow -- has to average **95% of rated output over a week**.
    When it runs it runs at 80-100%, and it never runs or rests for less than an hour.

    It is connected to PG&E through **Flex Connect**: instead of waiting for a grid
    upgrade it accepts an hourly import cap PG&E sends a day ahead. Most hours the cap
    is the full 5.5 MW service; in the hours the local grid is congested it drops to
    3.75 MW -- *below the plant's 4 MW floor*. The site also has **1 MW of solar**
    (`GenericRenewables`), its output a GFS weather forecast run through
    [hefty](https://github.com/williamhobbs/hefty)'s PV model, and a `GridConnection`
    that enforces the cap and forbids export. The bill is a PG&E B-20-shaped tariff:
    TOU energy, a max-demand charge and a 4-9 pm peak-demand charge.

    The ten preset locations are PG&E substations at the ends of the transmission
    constraints CAISO's market monitor found most congested in 2025; each is curtailed
    in the share of hours its constraint bound, in the midday hours it bound in. Or
    type in any latitude/longitude in PG&E territory and hefty fetches the forecast
    live.

    The **value curve** solves the plant at utilization targets from 70% to 100% and
    compares each with an inflexible baseline running flat at the same output on firm
    service: the saving per MWh the flexible schedule moved in time.
    """)
    return


@app.cell
def _(diagram, mo, render_diagram):
    # The flowsheet, from the flex_pse_examples.diagrams icon library. Edit
    # diagram.py, then run tools/render_diagrams.py so the WebAssembly page's copy
    # (public/flex_connect_solar/flowsheet.svg) stays in step.
    mo.Html(render_diagram(diagram.NODES, diagram.EDGES))
    return


@app.cell
def _(mo, model):
    cfg = model.load_config()
    _presets = {
        f"{s['name']} ({s['congestion_share']:.0%} congested)": s["id"]
        for s in model.load_sites()["sites"]
    }
    _presets["Custom latitude/longitude (live hefty forecast)"] = "custom"
    location = mo.ui.dropdown(
        options=_presets,
        value=next(iter(_presets)),
        label="Location",
    )
    lat = mo.ui.number(start=34.9, stop=42.0, step=0.01, value=38.58, label="Latitude")
    lon = mo.ui.number(
        start=-124.5, stop=-118.9, step=0.01, value=-121.49, label="Longitude"
    )
    fetch = mo.ui.run_button(label="Fetch forecast and solve")
    mo.vstack(
        [
            location,
            mo.hstack([lat, lon, fetch], justify="start"),
            mo.md(
                "Latitude/longitude are used only for *Custom*. A live GFS fetch "
                "downloads a week of forecast and takes a few minutes."
            ).style({"color": "#898781", "font-size": "0.85em"}),
        ]
    )
    return cfg, fetch, lat, location, lon


@app.cell
def _(cfg, fetch, lat, location, lon, mo, model, sites):
    # Resolve the location into the two site-specific inputs the model takes: a
    # solar capacity-factor series and a congestion share.
    if location.value != "custom":
        site_id = location.value
        _site = model.site_by_id(site_id)
        solar_cf = None  # build_model reads the committed cache
        congestion_share = _site["congestion_share"]
        where = f"**{_site['name']}** ({_site['lat']:.3f}, {_site['lon']:.3f})"
    else:
        mo.stop(
            not sites.in_pge_bounds(lat.value, lon.value),
            mo.callout(
                mo.md("That point is outside PG&E's service territory."), kind="warn"
            ),
        )
        mo.stop(
            not fetch.value,
            mo.callout(mo.md("Press **Fetch forecast and solve**."), kind="info"),
        )
        _nearest = sites.nearest_site(lat.value, lon.value)
        site_id = _nearest["id"]
        congestion_share = _nearest["congestion_share"]
        with mo.status.spinner(title="Fetching the GFS forecast with hefty..."):
            solar_cf = sites.solar_capacity_factor(cfg, lat.value, lon.value)[0]
        where = (
            f"**({lat.value:.3f}, {lon.value:.3f})** -- live forecast; Flex Connect "
            f"curtailment borrowed from the nearest preset, {_nearest['name']} "
            f"({congestion_share:.0%})"
        )
    mo.md(f"Location: {where}")
    return congestion_share, site_id, solar_cf


@app.cell
def _(cfg, congestion_share, mo, model, site_id, solar_cf):
    _kwargs = {"congestion_share": congestion_share}
    if solar_cf is not None:
        _kwargs["solar_cf"] = solar_cf
    with mo.status.spinner(title="Solving the value curve (14 week-long solves)..."):
        points = {
            target: model.value_of_flexibility(cfg, site_id, target, **_kwargs)
            for target in model.TARGETS
        }
    default = points[cfg["process"]["utilization_target"]]
    return default, points


@app.cell
def _(cfg, default, mo):
    mo.stop(not default["feasible"], mo.md("The default target is out of reach here."))
    _base = default["baseline_frame"]
    mo.md(
        f"""
        ## The week at {cfg["process"]["utilization_target"]:.0%}

        | | Flexible on Flex Connect | Flat on firm service |
        |---|---|---|
        | Operating cost | **\\${default["flexible_cost"]:,.0f}** | \\${default["baseline_cost"]:,.0f} |
        | Peak grid draw | {default["peak_kw"]:,.0f} kW | {_base["grid_import_kw"].max():,.0f} kW |
        | Peak grid draw, 4-9 pm | {default["peak_window_kw"]:,.0f} kW | {_base.loc[_base.index.hour.isin(range(16, 21)), "grid_import_kw"].max():,.0f} kW |
        | Draw above the Flex Connect cap | 0 kWh | {((_base["grid_import_kw"] - _base["import_limit_kw"]).clip(lower=0).sum() * cfg["time"]["time_step_hours"]):,.0f} kWh |
        | Curtailed hours | {default["curtailed_hours"]:.0f} h, process averaging {default["curtailed_load"] or 0:.0%} of rated in them | -- |
        | Hours off | {default["hours_off"]:.2f} h in {default["starts"]} restarts | 0 |
        | Solar share of process energy | {default["solar_share"]:.1%} | |

        Flexibility moved **{default["mwh_shifted"]:.1f} MWh** and saved
        **\\${default["savings"]:,.0f}**: **\\${default["value_per_mwh"]:,.0f} per MWh shifted**.
        """
    )
    return


@app.cell
def _(default, plt):
    _f = default["frame"]
    _base = default["baseline_frame"]
    fig_week, _ax = plt.subplots(figsize=(11, 3.8))
    _ax.fill_between(_f.index, 0, _f["grid_import_kw"], step="post", color="#9aa5b1", alpha=0.6, lw=0, label="Grid import")
    _ax.fill_between(
        _f.index, _f["grid_import_kw"], _f["grid_import_kw"] + _f["solar_kw"],
        step="post", color="#f0cf8a", lw=0, label="Solar",
    )
    _ax.step(_f.index, _f["import_limit_kw"], where="post", color="#eb6834", lw=1.2, label="Flex Connect cap")
    _ax.step(_f.index, _f["process_kw"], where="post", color="#2a78d6", lw=1.4, label="Process load")
    _ax.step(_base.index, _base["process_kw"], where="post", color="#0b0b0b", lw=0.8, ls="--", label="Flat baseline")
    _ax.set_ylabel("kW")
    _ax.set_ylim(0, 6000)
    _ax.legend(ncol=5, fontsize=8, loc="lower left", frameon=False)
    for _side in ("top", "right"):
        _ax.spines[_side].set_visible(False)
    fig_week.tight_layout()
    fig_week
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The value curve

    Each point is one utilization target -- the plant's *demand* curve for
    flexibility. The orange band is the *supply* side: the node's CAISO day-ahead
    LMP over the hours Flex Connect curtailed it (P10-P90, median line). **MWh shifted** is how much of the process's
    energy the flexible schedule moved away from the flat profile (half the summed
    absolute gap, so a MWh moved counts once); **\$/MWh** is the cost saved against
    the flat baseline per MWh moved. Below 80% the flat baseline is a reference, not
    a schedule -- the process cannot run below its floor, so it has to cycle, and the
    cycling can cost more than it saves. At 100% there is nothing left to shift, and
    under any Flex Connect curtailment it is out of reach.
    """)
    return


@app.cell
def _(model, pd, points):
    curve = pd.DataFrame(
        [
            {k: v for k, v in p.items() if k not in ("model", "frame", "baseline_frame")}
            for p in points.values()
        ]
    ).set_index("utilization_target")
    return (curve,)


@app.cell
def _(cfg, curve, model, plt, site_id):
    # A demand curve runs in order of quantity, not of target.
    _ok = curve[curve["feasible"]].sort_values("mwh_shifted")
    # The grid's side: the node's day-ahead LMP over the site's curtailed hours.
    # A typed-in location borrows the nearest preset's node.
    _band = model.lmp_band(cfg, site_id)
    fig_curve, _ax = plt.subplots(figsize=(7, 4))
    _ax.axhline(0, color="#c3c2b7", lw=0.8)
    if _band["lmp_hours"]:
        _ax.axhspan(_band["lmp_p10"], _band["lmp_p90"], color="#eb6834", alpha=0.18, lw=0)
        _ax.axhline(_band["lmp_p50"], color="#eb6834", lw=1.2,
                    label=f"LMP in curtailed hours, P10-P90 (n={_band['lmp_hours']})")
        _ax.legend(fontsize=8, frameon=False, loc="upper right")
    _ax.plot(_ok["mwh_shifted"], _ok["value_per_mwh"], color="#2a78d6", marker="o")
    for _t, _row in _ok.iterrows():
        _ax.annotate(
            f"{_t:.0%}", (_row["mwh_shifted"], _row["value_per_mwh"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    _ax.set_xlabel("MWh shifted over the week")
    _ax.set_ylabel("$ saved per MWh shifted")
    for _side in ("top", "right"):
        _ax.spines[_side].set_visible(False)
    fig_curve.tight_layout()
    fig_curve
    return


@app.cell
def _(curve):
    curve[
        [
            "feasible", "max_utilization", "baseline_cost", "flexible_cost",
            "savings", "mwh_shifted", "value_per_mwh", "hours_off", "solar_share",
        ]
    ].round(3)
    return


if __name__ == "__main__":
    app.run()
