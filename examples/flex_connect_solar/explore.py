# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib", "duckdb", "altair"]
# ///
"""Flex Connect + solar: the WebAssembly page.

Runs in the browser, so it can only import what Pyodide has -- no pyomo, no
flexops, no hefty, no cartopy, no sibling `model`. Everything here is DuckDB,
matplotlib and Altair over the Parquet files committed under `public/`: the sweep
from `tools/sweep.py`, and the map background `sites.py --basemap` cut from
Natural Earth with cartopy, offline. See CONTRIBUTING.md.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import altair as alt
    import duckdb
    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    # `notebook_location()` is a path locally and a URL in the browser. The name
    # is doubled because every example's `public/` merges into one directory.
    EXAMPLE = "flex_connect_solar"
    DATA = f"{mo.notebook_location()}/public/{EXAMPLE}"
    return DATA, EXAMPLE, alt, duckdb, json, mdates, mo, plt


@app.cell
def _():
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    SURFACE, SUN, GRIDDRAW = "#fcfcfb", "#f0cf8a", "#9aa5b1"

    def style(ax, ylabel=""):
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
        return ax

    return AXIS, BLUE, GRIDDRAW, INK, MUTED, ORANGE, SUN, SURFACE, style


@app.cell
def _(mo):
    mo.md(r"""
    # Flex Connect + solar

    **What is a MWh of load shifting worth to a plant on a flexible grid connection?**

    A 5 MW process on PG&E's **Flex Connect** -- an hourly import cap, 3.75 MW in the
    hours the local grid is congested, below the plant's 4 MW floor -- with 1 MW of
    solar, at ten of the most congested PG&E substations. **Hover over a site** for
    its numbers; **click one** to load its sweep below.
    """)
    return


@app.cell
def _(EXAMPLE, mo):
    _repo = f"https://github.com/flex-pse/flex-pse-examples/tree/main/examples/{EXAMPLE}"
    mo.callout(
        mo.md(
            f"""
    These are **precomputed** results. The model needs Pyomo and a MILP solver,
    and the solar forecast needs a live weather download -- neither runs in a
    browser, so every case was solved ahead of time. The model and a notebook that
    solves it, for any location you type in, live in [the repo]({_repo}).
    """
        ),
        kind="info",
    )
    return


@app.cell
def _(DATA, duckdb, mo):
    try:
        summary = duckdb.sql(
            f"SELECT * FROM read_parquet('{DATA}/summary.parquet') ORDER BY sweep_id"
        ).df()
        provenance = (
            duckdb.sql(f"SELECT key, value FROM read_parquet('{DATA}/provenance.parquet')")
            .df()
            .set_index("key")["value"]
        )
        load_error = None
    except Exception as exc:
        summary, provenance, load_error = None, None, exc

    mo.stop(
        load_error is not None,
        mo.callout(
            mo.md(f"**Couldn't load the results** from `{DATA}`.\n\n```\n{load_error}\n```"),
            kind="danger",
        ),
    )
    #: One row per site, most congested first, with its 95% point's scalars --
    #: what the map is drawn from and what a selection is matched against.
    sites = (
        summary[(summary.utilization_target - 0.95).abs() < 1e-9]
        .sort_values("congestion_share", ascending=False)
        .reset_index(drop=True)
    )
    return provenance, sites, summary


@app.cell
def _(DATA, duckdb, json, sites):
    # The map background: Natural Earth land, lakes and rivers, cut and simplified
    # offline with cartopy by `sites.py --basemap`. One row per shape, geometry as
    # a GeoJSON string -- read with DuckDB, since `open()` fails in the browser.
    _shapes = duckdb.sql(f"SELECT * FROM read_parquet('{DATA}/basemap.parquet')").df()
    basemap = {
        kind: [
            {"type": "Feature", "properties": {"name": r.name},
             "geometry": json.loads(r.geojson)}
            for r in _shapes[_shapes.kind == kind].itertuples()
        ]
        for kind in ("state", "lake", "river")
    }
    _meta = duckdb.sql(f"SELECT * FROM read_parquet('{DATA}/site_meta.parquet')").df()
    nodes = sites.merge(_meta, on="site", how="left")
    return basemap, nodes


@app.cell
def _(alt, basemap, mo, nodes):
    #: Palette for the map: sea, land, water lines, and the selection ring.
    _SEA, _LAND, _COAST = "#cfe4f1", "#f6f1e4", "#a9b8c2"
    _WATER, _RING = "#8fbfdc", "#0b0b0b"

    #: Reference towns, for orientation.
    _towns = alt.Data(values=[
        {"name": "San Francisco", "lat": 37.77, "lon": -122.42},
        {"name": "Sacramento", "lat": 38.58, "lon": -121.49},
        {"name": "Fresno", "lat": 36.74, "lon": -119.79},
        {"name": "Monterey", "lat": 36.60, "lon": -121.89},
        {"name": "Reno", "lat": 39.53, "lon": -119.81},
    ])

    # Fit the projection to the sites plus a margin, not to the clipped shapes.
    _w, _s, _e, _n = -123.5, 35.0, -118.7, 39.2
    # Two corner points, not a box: a polygon's winding would matter to d3-geo.
    _frame = {"type": "Feature", "properties": {}, "geometry": {
        "type": "MultiPoint", "coordinates": [[_w, _s], [_e, _n]],
    }}
    _proj = {"type": "mercator", "fit": _frame}

    # The sea: the whole sphere, under everything. (configure_view's fill does
    # not reach under a projected layer.)
    _sea = alt.Chart(alt.sphere()).mark_geoshape(fill=_SEA, clip=True).project(**_proj)
    _land = (
        alt.Chart(alt.Data(values=basemap["state"]))
        .mark_geoshape(fill=_LAND, stroke=_COAST, strokeWidth=0.9, clip=True)
        .project(**_proj)
    )
    _lakes = (
        alt.Chart(alt.Data(values=basemap["lake"]))
        .mark_geoshape(fill=_SEA, stroke=_WATER, strokeWidth=0.5, clip=True)
        .project(**_proj)
    )
    _rivers = (
        alt.Chart(alt.Data(values=basemap["river"]))
        .mark_geoshape(filled=False, stroke=_WATER, strokeWidth=1.1, opacity=0.8, clip=True)
        .project(**_proj)
    )
    _town_marks = alt.Chart(_towns).encode(longitude="lon:Q", latitude="lat:Q")
    _town_layer = _town_marks.mark_square(size=14, color="#b9b2a0", clip=True) + _town_marks.mark_text(
        align="left", dx=6, dy=-5, fontSize=10, fontStyle="italic", color="#a39c8a",
        clip=True,
    ).encode(text="name:N")

    # One click selects one site; clicking empty sea or land clears it.
    pick = alt.selection_point(name="site_pick", fields=["site"], on="click",
                               clear="click", empty=False)
    _base = alt.Chart(nodes).encode(longitude="lon:Q", latitude="lat:Q")
    _tooltip = [
        alt.Tooltip("site_name:N", title="Site"),
        alt.Tooltip("pnode:N", title="CAISO pricing node"),
        alt.Tooltip("congestion_share:Q", title="Hours congested (2025)", format=".0%"),
        alt.Tooltip("curtailed_hours:Q", title="Flex Connect curtailed h / week", format=".0f"),
        alt.Tooltip("value_per_mwh:Q", title="$ saved per MWh shifted (95%)", format="$,.0f"),
        alt.Tooltip("savings:Q", title="Saved vs flat, week (95%)", format="$,.0f"),
        alt.Tooltip("mwh_shifted:Q", title="MWh shifted (95%)", format=".1f"),
        alt.Tooltip("lmp_p50:Q", title="Median LMP, curtailed hours", format="$,.1f"),
        alt.Tooltip("solar_share:Q", title="Solar share of process energy", format=".1%"),
        alt.Tooltip("max_utilization:Q", title="Most utilization reachable", format=".0%"),
    ]
    _dots = _base.mark_circle(size=260, opacity=1).encode(
        color=alt.Color(
            "value_per_mwh:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=alt.Legend(
                title="$ saved per MWh shifted, at 95%", orient="bottom-left",
                direction="horizontal", gradientLength=180, titleFontSize=10,
                labelFontSize=9, fillColor="rgba(255,255,255,0.85)", padding=8,
                cornerRadius=6, strokeColor="#d5e3ec",
                titleLimit=260, format="$,.0f",
            ),
        ),
        stroke=alt.condition(pick, alt.value(_RING), alt.value("white")),
        strokeWidth=alt.condition(pick, alt.value(3.5), alt.value(1.6)),
        tooltip=_tooltip,
    ).add_params(pick)
    # Labels sit right of their dot, except in the South Bay, where four sites
    # are within a few km of each other: two go left, and each pair splits up
    # and down.
    #: site -> (align, dx, dy); anything not listed is right-aligned at (11, 0).
    _PLACE = {
        "mountain_view": ("right", -11, -5),
        "el_patio": ("right", -11, 9),
        "san_jose_a": ("left", 11, -8),
        "metcalf": ("left", 11, 4),
    }

    def _label_layer(align, dx, dy, sites_in=None, exclude=None):
        layer = _base.mark_text(
            align=align, dx=dx, dy=dy, fontSize=11, fontWeight=500, color="#2b2b2b",
            clip=True,
        ).encode(text="site_name:N", tooltip=_tooltip)
        if sites_in is not None:
            layer = layer.transform_filter(alt.FieldOneOfPredicate("site", sites_in))
        if exclude is not None:
            layer = layer.transform_filter(
                {"not": alt.FieldOneOfPredicate("site", exclude)}
            )
        return layer

    _labels = [_label_layer("left", 11, 0, exclude=list(_PLACE))] + [
        _label_layer(a, dx, dy, sites_in=[site]) for site, (a, dx, dy) in _PLACE.items()
    ]

    _chart = (
        alt.layer(_sea, _land, _lakes, _rivers, _town_layer, _dots, *_labels)
        .project(**_proj)
        .properties(width="container", height=520)
        .configure_view(fill=_SEA, stroke=None, cornerRadius=10)
        .configure(background="transparent", font="Inter, system-ui, sans-serif")
    )
    site_map = mo.ui.altair_chart(_chart, chart_selection=False, legend_selection=False)
    mo.vstack(
        [
            mo.md(
                "<span style='font-size:0.8rem;letter-spacing:0.08em;"
                "text-transform:uppercase;color:#4a6b82'>Ten congested PG&E nodes "
                "&middot; hover for detail &middot; click to explore</span>"
            ),
            site_map,
        ],
        gap=0.4,
    ).style(
        {
            "background": "linear-gradient(160deg, #eef5fa 0%, #e3eef6 55%, #f4f0e6 100%)",
            "border": "1px solid #d5e3ec",
            "border-radius": "14px",
            "padding": "14px 16px 10px",
            "box-shadow": "0 1px 2px rgba(20,40,60,0.06), 0 6px 20px rgba(20,40,60,0.06)",
        }
    )
    return (site_map,)


@app.cell
def _(mo, summary):
    _targets = sorted(summary["utilization_target"].unique())
    target_pick = mo.ui.dropdown(
        options={f"{t:.0%}": float(t) for t in _targets},
        value="95%" if any(abs(t - 0.95) < 1e-9 for t in _targets) else f"{_targets[-1]:.0%}",
        label="Utilization target",
    )
    return (target_pick,)


@app.cell
def _(mo, nodes, site_map, target_pick):
    # The clicked site, or -- with nothing clicked -- the most congested one.
    # `nodes` is most-congested first. A layered chart hands its selection back
    # through apply_selection rather than .value.
    _picked = site_map.apply_selection(nodes)
    _from_map = len(_picked) > 0
    active = _picked["site"].iloc[0] if _from_map else nodes["site"].iloc[0]
    selected = [active]
    _name = nodes.set_index("site").loc[active, "site_name"]
    _how = (
        "clicked on the map -- click empty map to reset"
        if _from_map
        else "the most congested site -- click any site on the map to switch"
    )
    mo.hstack(
        [mo.md(f"Showing **{_name}** ({_how})."), target_pick],
        justify="space-between",
        align="center",
    )
    return active, selected


@app.cell
def _(active, mo, summary, target_pick):
    _site_rows = summary[summary.site == active]
    _row = _site_rows[(_site_rows.utilization_target - target_pick.value).abs() < 1e-9]
    _reach = float(_site_rows["max_utilization"].iloc[0])
    mo.stop(
        _row.empty,
        mo.callout(
            mo.md(
                f"**{target_pick.value:.0%} is out of reach here.** With this "
                f"site's Flex Connect cap and solar the plant can average at most "
                f"about {_reach:.0%}."
            ),
            kind="warn",
        ),
    )
    point = _row.iloc[0]
    mo.md(
        f"""
        ### {point.site_name} at {point.utilization_target:.0%}

        | | |
        |---|---|
        | Flexible cost (week) | **\\${point.operating_cost:,.0f}** |
        | Flat-on-firm-service cost | \\${point.baseline_cost:,.0f} |
        | Saved | **\\${point.savings:,.0f}** on **{point.mwh_shifted:.1f} MWh** shifted -- **\\${point.value_per_mwh:,.0f}/MWh** |
        | Nodal LMP in curtailed hours | \\${point.lmp_p10:,.0f}-{point.lmp_p90:,.0f}/MWh, median \\${point.lmp_p50:,.0f} ({int(point.lmp_hours)} h) |
        | Curtailed hours | {point.curtailed_hours:.0f} h |
        | Hours off | {point.hours_off:.2f} h in {int(point.starts)} restarts |
        | Solar share of process energy | {point.solar_share:.1%} |
        | Most this site can reach | ~{_reach:.0%} |
        """
    )
    return (point,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Supply and demand for flexibility

    The blue curve is the plant's side: each point is a weekly **utilization target**,
    where the plant was solved twice -- flexibly on Flex Connect, and running flat at
    the same output on an ordinary firm connection. **MWh shifted** is how much
    energy the flexible schedule moved in time; **\$/MWh** is what that saved per MWh
    moved.

    The orange band is the grid's side: the node's **day-ahead LMP** (CAISO OASIS)
    across the hours Flex Connect curtailed the site that week -- P10 to P90, median
    marked. Where the curve sits above the band, a MWh of flexibility is worth more
    to the plant than the energy it moves is worth to the grid; most of that gap is
    demand charges, which the LMP does not see.

    Below 80% the flat run is only a yardstick -- the real process cannot run under
    its floor, so it has to cycle, and that can cost more than it saves. 100% is
    missing everywhere: no schedule runs flat out through a curtailed hour.
    """)
    return


@app.cell
def _(AXIS, BLUE, INK, MUTED, ORANGE, SURFACE, active, plt, selected, sites, style, summary):
    def _band(ax, rows):
        """The site's LMP band over its curtailed hours, P10-P90 with the median."""
        r = rows.iloc[0]
        if r.lmp_hours:
            ax.axhspan(r.lmp_p10, r.lmp_p90, color=ORANGE, alpha=0.18, lw=0, zorder=1)
            ax.axhline(r.lmp_p50, color=ORANGE, lw=1.2, zorder=2)

    def _curves():
        fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
        fig.patch.set_facecolor(SURFACE)
        style(ax, "$/MWh")
        ax.axhline(0, color=AXIS, lw=0.8)
        for site, rows in summary.groupby("site"):
            if site == active:
                continue
            # A demand curve runs in order of quantity, not of target.
            rows = rows.sort_values("mwh_shifted")
            picked = site in selected
            ax.plot(rows["mwh_shifted"], rows["value_per_mwh"],
                    color="#7fa9e0" if picked else "#c8d0da",
                    lw=1.4 if picked else 1, marker="o", ms=2, zorder=2)
        rows = summary[summary.site == active].sort_values("mwh_shifted")
        _band(ax, rows)
        ax.plot(rows["mwh_shifted"], rows["value_per_mwh"], color=BLUE, lw=2.2,
                marker="o", ms=4, zorder=3)
        for r in rows.itertuples():
            ax.annotate(f"{r.utilization_target:.0%}", (r.mwh_shifted, r.value_per_mwh),
                        textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
        r = rows.iloc[0]
        ax.annotate(
            # Escaped: two bare dollar signs would open matplotlib mathtext.
            f"LMP in curtailed hours (n={int(r.lmp_hours)}): "
            f"\\${r.lmp_p10:.0f}-{r.lmp_p90:.0f}, median \\${r.lmp_p50:.0f}",
            (1, r.lmp_p50), xycoords=("axes fraction", "data"), xytext=(-4, 5),
            textcoords="offset points", ha="right", fontsize=8, color=ORANGE,
        )
        ax.set_xlabel("MWh shifted over the week", color=MUTED, fontsize=9)
        others = "other selected sites in light blue, the rest in grey" if len(selected) > 1 else "other sites in grey"
        ax.set_title(
            f"{r.site_name}: value of flexibility (blue) vs nodal LMP (orange); {others}",
            loc="left", fontsize=10, color=INK,
        )
        return fig

    def _grid():
        fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True,
                                 layout="constrained")
        fig.patch.set_facecolor(SURFACE)
        for ax, site in zip(axes.flat, sites["site"]):
            rows = summary[summary.site == site].sort_values("mwh_shifted")
            style(ax)
            ax.axhline(0, color=AXIS, lw=0.8)
            _band(ax, rows)
            picked = site in selected
            ax.plot(rows["mwh_shifted"], rows["value_per_mwh"],
                    color=BLUE if picked else "#9aa5b1",
                    lw=2 if picked else 1.2, marker="o", ms=3, zorder=3)
            if picked:
                for side in ("left", "bottom"):
                    ax.spines[side].set_color(BLUE)
            r = rows.iloc[0]
            ax.set_title(f"{r.site_name}\n{r.congestion_share:.0%} congested",
                         fontsize=8, color=INK, loc="left",
                         fontweight="bold" if picked else "normal")
        for ax in axes[1]:
            ax.set_xlabel("MWh shifted", color=MUTED, fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("$/MWh", color=MUTED, fontsize=8)
        fig.suptitle("Every node: value of flexibility vs LMP in its curtailed hours "
                     "(selected sites in blue)",
                     x=0.01, ha="left", fontsize=10, color=INK)
        return fig

    curves_fig, grid_fig = _curves(), _grid()
    return curves_fig, grid_fig


@app.cell
def _(curves_fig):
    curves_fig
    return


@app.cell
def _(mo, point):
    #: The widest energy spread in config.json's B-20-shaped tariff, $/MWh: peak
    #: $0.245/kWh less off-peak $0.155/kWh. Re-timing energy alone cannot earn more.
    _TOU_SPREAD = 90.0
    _demand_floor = max(point.value_per_mwh - _TOU_SPREAD, 0.0)
    mo.md(
        f"""
    #### Reading the gap between the curve and the band

    The vertical distance is how much more a MWh of flexibility is worth **to the
    plant, under its retail tariff**, than the wholesale energy it moves is worth at
    that node. The curve is retail-bill money -- TOU energy plus demand charges --
    divided by MWh shifted. The band is the day-ahead LMP in the curtailed hours,
    roughly what the grid pays for energy there and then.

    Almost all of the gap is **demand charges**. The tariff's widest energy spread
    is \\$0.245 - \\$0.155/kWh = **\\${_TOU_SPREAD:.0f}/MWh**, so re-timing energy can
    explain at most that much of the curve. At **{point.site_name}** at
    {point.utilization_target:.0%}, the curve sits at
    **\\${point.value_per_mwh:,.0f}/MWh**, so at least
    **\\${_demand_floor:,.0f}/MWh** of it is demand-charge savings, which the LMP
    never sees.

    Before reading the gap as surplus:

    1. **The demand-charge part is overstated.** Demand charges are monthly but
       the horizon is one week, and the model applies the full monthly charge to
       it. A month's demand saving is divided by a week's shifted MWh; over a real
       month the plant would shift roughly 4x the MWh for the same saving. The gap
       would shrink a lot, though likely not close.
    2. **The curve is an average, not a marginal value.** Each point is total
       saving over total MWh at that target. A true supply-and-demand reading
       needs the value of the *next* MWh -- the slope between points -- which is
       lower at high shift volumes.
    3. **The LMP level is the wrong benchmark for shifting.** Moving a MWh out of a
       curtailed hour puts it into another hour that also has a price. The
       wholesale value of the shift is the *spread* between the LMPs of the hours
       load left and the hours it moved to, not the level -- with midday LMPs of
       \\$15-45/MWh, probably a few to tens of dollars.
    4. **The LMP leaves out what Flex Connect actually buys.** It prices bulk
       energy and transmission congestion, not the distribution capacity the plant
       frees up by accepting the cap.

    So the gap measures how hard the B-20 rate design rewards flexibility beyond
    its wholesale energy value -- inflated, in size, by the monthly-charge
    simplification. It is not a clean consumer surplus.
    """
    )
    return


@app.cell
def _(grid_fig):
    grid_fig
    return


@app.cell
def _(mo, point):
    mo.md(f"""
    ## {point.site_name}'s week at {point.utilization_target:.0%}
    """)
    return


@app.cell
def _(DATA, duckdb, point):
    series = duckdb.sql(
        f"""
        SELECT * FROM read_parquet('{DATA}/series.parquet')
        WHERE sweep_id = '{point.sweep_id}'
        ORDER BY timestamp
        """
    ).df()
    return (series,)


@app.cell
def _(
    BLUE,
    GRIDDRAW,
    INK,
    MUTED,
    ORANGE,
    SUN,
    SURFACE,
    mdates,
    plt,
    series,
    style,
):
    def _week():
        s = series
        fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
        fig.patch.set_facecolor(SURFACE)
        style(ax, "kW")
        t = s["timestamp"]
        ax.fill_between(t, 0, s["grid_import_kw"], step="post", color=GRIDDRAW, alpha=0.6, lw=0, label="Grid import")
        ax.fill_between(
            t, s["grid_import_kw"], s["grid_import_kw"] + s["solar_kw"],
            step="post", color=SUN, lw=0, label="Solar",
        )
        ax.step(t, s["import_limit_kw"], where="post", color=ORANGE, lw=1.2, label="Flex Connect cap")
        ax.step(t, s["process_kw"], where="post", color=BLUE, lw=1.4, label="Process load")
        ax.step(t, s["baseline_kw"], where="post", color=INK, lw=0.8, ls="--", label="Flat baseline")
        ax.set_ylim(0, 6200)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
        ax.legend(ncol=5, fontsize=8, loc="lower left", frameon=False, labelcolor=MUTED)
        return fig

    _week()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The plant

    A 5 MW electrically driven process has to average a set share of its rated output
    over a week. When it runs it runs at 80-100%, and it never runs or rests for less
    than an hour. It is on PG&E's **Flex Connect**: a grid connection that comes with
    an hourly import cap, sent a day ahead -- 5.5 MW most hours, **3.75 MW** in the
    hours the local grid is congested, which is *below the plant's 4 MW floor*. The
    site has **1 MW of solar** and cannot export.

    The ten sites are PG&E substations at the ends of the transmission constraints
    CAISO's market monitor found most congested in 2025. Each is curtailed in the
    share of hours its constraint bound, in the midday hours it bound in -- which is
    exactly when the solar is producing.
    """)
    return


@app.cell
def _(DATA, mo):
    # Rendered offline from diagram.py by tools/render_diagrams.py and committed
    # beside the sweep data -- the renderer is not something Pyodide installs.
    mo.image(src=f"{DATA}/flowsheet.svg", alt="Process flow diagram: water through a 5 MW process, powered by solar and a Flex Connect grid connection")
    return


@app.cell
def _(mo, selected, sites):
    _t = sites.assign(selected=sites["site"].isin(selected))
    _table = (
        _t[["site_name", "selected", "congestion_share", "curtailed_hours", "savings",
            "mwh_shifted", "value_per_mwh", "lmp_p50", "solar_share", "max_utilization"]]
        .rename(columns={
            "site_name": "Site", "selected": "Selected", "congestion_share": "Congested",
            "curtailed_hours": "Curtailed h", "savings": "Saved $",
            "mwh_shifted": "MWh shifted", "value_per_mwh": "$/MWh",
            "lmp_p50": "Median LMP, curtailed h",
            "solar_share": "Solar share", "max_utilization": "Max reachable",
        })
        .round(3)
    )
    mo.vstack([mo.md("### Every site at the default 95%"), mo.ui.table(_table, selection=None)])
    return


@app.cell
def _(mo, provenance):
    _commit = str(provenance.get("flexpse_commit", ""))[:7]
    mo.md(f"""
    <small>Generated {provenance.get("generated_utc", "?")} with flex-pse
    {provenance.get("flexpse_version", "?")} ({_commit}), solver
    {provenance.get("solver", "?")}. Congestion shares: CAISO DMM, 2025 Annual Report on
    Market Issues and Performance, Table 5.3. LMPs: CAISO OASIS day-ahead, same week.
    Solar: GFS forecast via hefty. Tariff: PG&E B-20-shaped, representative rates.</small>
    """)
    return


if __name__ == "__main__":
    app.run()
