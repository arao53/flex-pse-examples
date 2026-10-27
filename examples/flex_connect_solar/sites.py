"""Where the plant is: the preset PG&E sites, and the solar resource at any of them.

Everything in this module that touches the network lives behind
:func:`solar_capacity_factor`, which asks `hefty <https://github.com/williamhobbs/hefty>`_
for a GFS solar-resource forecast and runs it through hefty's pvlib-based plant
model. Nothing else in the example calls it: :func:`build_cache` runs it once for
the ten preset sites and writes ``data/solar_cf.parquet``, which ``model.py``,
``sweep.py`` and the tests read offline. ``notebook.py`` calls it live for a
latitude/longitude typed in by hand.

The ten sites themselves are in ``sites.json`` -- substations at the ends of the
PG&E-area day-ahead constraints CAISO's Department of Market Monitoring found most
congested in 2025. See the comment in that file for exactly what was taken from
the report and what was not.

Run it as a script to rebuild the cache::

    python examples/flex_connect_solar/sites.py          # solar forecast
    python examples/flex_connect_solar/sites.py --lmp    # CAISO day-ahead LMPs
    python examples/flex_connect_solar/sites.py --basemap  # page map (cartopy)
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SITES_PATH = HERE / "sites.json"
CACHE_PATH = HERE / "data" / "solar_cf.parquet"

#: PG&E's service territory, roughly: the bounding box notebook.py checks a typed
#: latitude/longitude against before it spends minutes downloading a forecast for
#: it. A box, not the territory's outline, so it admits some of Nevada and the
#: ocean -- it is a guard against a typo, not a service-area lookup.
PGE_BOUNDS = {"lat": (34.9, 42.0), "lon": (-124.5, -118.9)}

#: Local clock the horizon is written in. GFS is in UTC.
TIMEZONE = "America/Los_Angeles"


def load_sites() -> dict:
    """Return ``sites.json`` parsed."""
    return json.loads(SITES_PATH.read_text())


def site_by_id(site_id: str) -> dict:
    """Return one preset site's record.

    Raises:
        KeyError: If ``site_id`` is not one of the presets.
    """
    for site in load_sites()["sites"]:
        if site["id"] == site_id:
            return site
    known = [s["id"] for s in load_sites()["sites"]]
    raise KeyError(f"unknown site {site_id!r}; expected one of {known}")


def in_pge_bounds(lat: float, lon: float) -> bool:
    """Whether a point sits inside :data:`PGE_BOUNDS`."""
    (lat_lo, lat_hi), (lon_lo, lon_hi) = PGE_BOUNDS["lat"], PGE_BOUNDS["lon"]
    return lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi


def nearest_site(lat: float, lon: float) -> dict:
    """The preset site closest to a point, by a flat-earth distance.

    A typed-in location has no congestion history of its own, so notebook.py
    borrows its Flex Connect curtailment from the nearest preset.
    """
    import math

    def distance(site):
        dx = (site["lon"] - lon) * math.cos(math.radians(lat))
        return math.hypot(site["lat"] - lat, dx)

    return min(load_sites()["sites"], key=distance)


def horizon_index(cfg: dict) -> pd.DatetimeIndex:
    """The model's time points: naive local timestamps, one per step."""
    time = cfg["time"]
    step = pd.Timedelta(hours=time["time_step_hours"])
    start = pd.Timestamp(time["start_date"])
    end = pd.Timestamp(time["end_date"])
    return pd.date_range(start, end - step, freq=step)


def solar_capacity_factor(cfg: dict, lat, lon) -> pd.DataFrame:
    """Forecast the plant's capacity factor at one or more points with hefty.

    Downloads a GFS forecast initialized at ``weather.init_utc`` covering the
    model horizon -- from dynamical.org's Zarr archive when ``weather.priority``
    is ``"dynamical"``, which reads only the points it needs and is an order of
    magnitude faster than pulling GRIB files from NOAA's bucket, decomposes it to plane-of-array irradiance and runs it
    through :func:`hefty.pv_model.model_pv_power` for a single-axis tracker
    sized per ``cfg["solar"]``. The result is AC output over AC nameplate,
    clipped to ``[0, 1]`` (the bound ``GenericRenewables`` enforces).

    GFS gives hourly *interval averages* stamped at the interval's midpoint, so
    each hour's value is held across that hour's model steps rather than
    interpolated -- interpolating would smear the morning ramp into the night
    before it.

    Args:
        cfg: The example config (``time``, ``solar`` and ``weather`` are read).
        lat: A latitude, or a list of them.
        lon: A longitude, or a list of them, the same length as ``lat``.

    Returns:
        A DataFrame indexed by the model's naive local time points, one column
        per point (``0``, ``1``, ... in the order given).
    """
    from hefty.pv_model import model_pv_power
    from hefty.solar import get_solar_forecast

    lats = list(lat) if isinstance(lat, (list, tuple)) else [lat]
    lons = list(lon) if isinstance(lon, (list, tuple)) else [lon]
    index = horizon_index(cfg)

    init = pd.Timestamp(cfg["weather"]["init_utc"], tz="UTC")
    first_hour = index[0].tz_localize(TIMEZONE).tz_convert("UTC")
    lead = int((first_hour - init) / pd.Timedelta(hours=1))
    hours = round((index[-1] - index[0]) / pd.Timedelta(hours=1)) + 1

    resource = get_solar_forecast(
        lats if len(lats) > 1 else lats[0],
        lons if len(lons) > 1 else lons[0],
        init.tz_localize(None),
        # One hour spare: the last interval average is stamped at its midpoint
        # and can land just past the window.
        run_length=hours + 1,
        lead_time_to_start=lead,
        model=cfg["weather"]["model"],
        priority=cfg["weather"].get("priority"),
        attempts=4,
    )
    if "point" not in resource.columns:
        resource = resource.assign(point=0)

    solar = cfg["solar"]
    nameplate_ac = solar["nameplate_ac_kw"]
    out = {}
    for point, (p_lat, p_lon) in enumerate(zip(lats, lons, strict=True)):
        data = resource[resource["point"] == point].drop(columns="point")
        # GFS carries no surface albedo; the PV model's ground-reflected term
        # needs one, so it is a config constant.
        if "albedo" not in data.columns:
            data = data.assign(albedo=solar["albedo"])
        power_ac, _ = model_pv_power(
            data,
            p_lat,
            p_lon,
            mount_type="single-axis",
            gcr=solar["gcr"],
            nameplate_dc=nameplate_ac * solar["dc_ac_ratio"],
            nameplate_ac=nameplate_ac,
            dc_loss_fraction=solar["dc_loss_fraction"],
            gamma_pdc=solar["gamma_pdc"],
            shade_loss_model="linear",
            max_tracker_angle=solar["max_tracker_angle"],
            eta_inv_nom=solar["eta_inv_nom"],
        )
        hourly = (power_ac / nameplate_ac).clip(0.0, 1.0).fillna(0.0)
        # Interval-midpoint UTC stamps -> the local hour each interval starts.
        starts = (hourly.index - pd.Timedelta(minutes=30)).tz_convert(TIMEZONE)
        hourly.index = starts.tz_localize(None).floor("h")
        # Past F120 GFS is 3-hourly; fill the hours between linearly. Short
        # gaps only -- anything longer is a failed download, and build_cache
        # refuses to write a frame with holes in it.
        hours = pd.date_range(index[0].floor("h"), index[-1].floor("h"), freq="h")
        hourly = hourly.reindex(hourly.index.union(hours)).interpolate(
            method="time", limit=2, limit_area="inside"
        )
        out[point] = hourly.reindex(index.floor("h")).to_numpy()
    return pd.DataFrame(out, index=index)


def build_cache(cfg: dict | None = None) -> pd.DataFrame:
    """Forecast every preset site and write :data:`CACHE_PATH`.

    One hefty call for all ten sites: hefty downloads each GRIB field once and
    samples every point from it, so ten sites cost barely more than one.

    Returns:
        The cached frame, indexed by local time, one column per site id.
    """
    if cfg is None:
        cfg = json.loads((HERE / "config.json").read_text())
    sites = load_sites()["sites"]
    frame = solar_capacity_factor(
        cfg, [s["lat"] for s in sites], [s["lon"] for s in sites]
    )
    frame.columns = [s["id"] for s in sites]
    frame.index.name = "timestamp"
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"forecast left gaps in the horizon for {missing}")
    CACHE_PATH.parent.mkdir(exist_ok=True)
    # DuckDB writes the Parquet, as tools/sweep.py does, so no pyarrow is needed.
    import duckdb

    duckdb.from_df(frame.reset_index()).write_parquet(str(CACHE_PATH))
    return frame


def cached_capacity_factor() -> pd.DataFrame:
    """Read :data:`CACHE_PATH` -- every preset site's capacity factor, offline."""
    import duckdb

    return (
        duckdb.sql(f"SELECT * FROM read_parquet('{CACHE_PATH}') ORDER BY timestamp")
        .df()
        .set_index("timestamp")
    )


LMP_PATH = HERE / "data" / "lmp.parquet"
OASIS_URL = "https://oasis.caiso.com/oasisapi/SingleZip"


def fetch_lmps(cfg: dict) -> pd.DataFrame:
    """Download the horizon's day-ahead hourly LMPs at every preset site's pnode.

    Straight from CAISO OASIS (``PRC_LMP``, ``DAM``), which is public and needs
    no key: one request for all ten nodes over the whole horizon. The zip holds
    the components (LMP, energy, congestion, loss, GHG) in one CSV, or one CSV
    per component; only the total LMP is kept.

    Returns:
        A DataFrame indexed by naive local hour-start, one column per site id,
        in $/MWh.
    """
    import io
    import urllib.parse
    import urllib.request
    import zipfile

    sites = load_sites()["sites"]
    index = horizon_index(cfg)
    start = index[0].tz_localize(TIMEZONE).tz_convert("UTC")
    end = (index[-1] + pd.Timedelta(hours=cfg["time"]["time_step_hours"])).tz_localize(
        TIMEZONE
    ).tz_convert("UTC")
    query = urllib.parse.urlencode(
        {
            "queryname": "PRC_LMP",
            "startdatetime": start.strftime("%Y%m%dT%H:%M-0000"),
            "enddatetime": end.strftime("%Y%m%dT%H:%M-0000"),
            "version": 12,
            "market_run_id": "DAM",
            "node": ",".join(s["pnode"] for s in sites),
            "resultformat": 6,
        },
        safe=":,",
    )
    with urllib.request.urlopen(f"{OASIS_URL}?{query}", timeout=300) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    names = [n for n in archive.namelist() if n.endswith(".csv")]
    if not names:
        raise ValueError(
            f"OASIS returned no CSV (got {archive.namelist()}); it sends an "
            "error XML in the zip when a request is rejected."
        )
    raw = pd.concat(pd.read_csv(archive.open(n)) for n in names)
    raw = raw[raw["LMP_TYPE"] == "LMP"]
    stamps = pd.to_datetime(raw["INTERVALSTARTTIME_GMT"], utc=True)
    raw = raw.assign(timestamp=stamps.dt.tz_convert(TIMEZONE).dt.tz_localize(None))
    by_node = raw.pivot_table(index="timestamp", columns="NODE", values="MW")
    frame = pd.DataFrame(
        {s["id"]: by_node.get(s["pnode"]) for s in sites},
        index=pd.date_range(index[0], index[-1].floor("h"), freq="h", name="timestamp"),
    )
    missing = frame.columns[frame.isna().any()].tolist()
    if missing:
        raise ValueError(f"OASIS returned no complete LMP series for {missing}")
    return frame


def build_lmp_cache(cfg: dict | None = None) -> pd.DataFrame:
    """Fetch the LMPs once and write :data:`LMP_PATH`, offline from then on."""
    import duckdb

    if cfg is None:
        cfg = json.loads((HERE / "config.json").read_text())
    frame = fetch_lmps(cfg)
    LMP_PATH.parent.mkdir(exist_ok=True)
    duckdb.from_df(frame.reset_index()).write_parquet(str(LMP_PATH))
    return frame


BASEMAP_PATH = HERE / "public" / "flex_connect_solar" / "basemap.parquet"

#: The map's window, lon/lat degrees: PG&E's sites plus a margin of coast and
#: Sierra. Every shape is clipped to it, which is most of what keeps the file small.
BASEMAP_BOUNDS = (-126.0, 33.8, -115.5, 40.6)


def build_basemap() -> "pd.DataFrame":
    """Write the page's map background: land, state lines, lakes, rivers, as GeoJSON.

    The browser page cannot run cartopy, so this runs it once, offline: Natural
    Earth's 10 m states (land, and the coast as its edge), lakes and rivers, clipped
    to :data:`BASEMAP_BOUNDS` and simplified to ~1 km. Each shape is one row --
    ``kind``, ``name``, and its GeoJSON geometry as a string -- in a Parquet file
    the page reads with DuckDB like the rest of its data (``open()`` does not
    work in the browser). Natural Earth is public domain.

    Returns:
        The rows written.
    """
    import duckdb
    import shapely
    import shapely.geometry.polygon
    from cartopy.io import shapereader
    from shapely.geometry import box, mapping

    window = box(*BASEMAP_BOUNDS)
    rows = []

    def clockwise(geom):
        """Exterior rings clockwise, the winding d3-geo (and so Vega) expects.

        GeoJSON and shapely wind exteriors counter-clockwise; d3-geo is
        spherical and reads that as "everything on the globe *except* this
        shape", which fills the map and breaks any projection fit.
        """
        if geom.geom_type == "Polygon":
            return shapely.geometry.polygon.orient(geom, sign=-1.0)
        if geom.geom_type == "MultiPolygon":
            return shapely.geometry.MultiPolygon([clockwise(g) for g in geom.geoms])
        return geom

    def add(kind, name, geom):
        geom = shapely.simplify(geom.intersection(window), 0.01, preserve_topology=True)
        geom = clockwise(geom)
        if not geom.is_empty:
            rows.append({"kind": kind, "name": name, "geojson": json.dumps(mapping(geom))})

    states = shapereader.natural_earth("10m", "cultural", "admin_1_states_provinces_lakes")
    for record in shapereader.Reader(states).records():
        attrs = record.attributes
        if attrs.get("adm0_a3") == "USA" and record.geometry.intersects(window):
            add("state", attrs["name"], record.geometry)

    lakes = shapereader.natural_earth("10m", "physical", "lakes")
    for record in shapereader.Reader(lakes).records():
        if record.geometry.intersects(window) and record.geometry.area > 0.002:
            add("lake", record.attributes.get("name") or "", record.geometry)

    rivers = shapereader.natural_earth("10m", "physical", "rivers_lake_centerlines")
    for record in shapereader.Reader(rivers).records():
        if record.geometry.intersects(window) and record.attributes.get("scalerank", 99) <= 9:
            add("river", record.attributes.get("name") or "", record.geometry)

    frame = pd.DataFrame(rows)
    BASEMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    duckdb.from_df(frame).write_parquet(str(BASEMAP_PATH))

    # The site metadata the map's tooltips show and the sweep summary does not
    # carry: which CAISO pricing node and which congested constraint each is.
    meta = pd.DataFrame(load_sites()["sites"])[["id", "pnode", "constraint"]]
    duckdb.from_df(meta.rename(columns={"id": "site"})).write_parquet(
        str(BASEMAP_PATH.with_name("site_meta.parquet"))
    )
    return frame


if __name__ == "__main__":
    import sys

    if "--basemap" in sys.argv:
        shapes = build_basemap()
        print(shapes[["kind", "name"]].to_string())
        print(f"{BASEMAP_PATH.stat().st_size / 1024:.0f} KB")
        raise SystemExit
    if "--lmp" in sys.argv:
        lmp = build_lmp_cache()
        print(lmp.describe().T.round(2).to_string())
        raise SystemExit
    cf = build_cache()
    summary = pd.DataFrame(
        {
            "mean_cf": cf.mean(),
            "peak_cf": cf.max(),
            "peak_hour": cf.groupby(cf.index.hour).mean().idxmax(),
        }
    )
    print(summary.round(3).to_string())
