"""Build data/reference_places.db: national capitals, state/province capitals and city centers.

Source: GeoNames (https://www.geonames.org, CC BY 4.0). Download these four files into one folder first:
    https://download.geonames.org/export/dump/cities500.zip   (unzip it: cities500.txt)
    https://download.geonames.org/export/dump/admin1CodesASCII.txt
    https://download.geonames.org/export/dump/admin2Codes.txt
    https://download.geonames.org/export/dump/countryInfo.txt
Then, from the project root:
    python tools/build_reference_places.py /path/to/that/folder [--nominatim]

With --nominatim, cities that GeoNames cannot place and that hold at least 2 breweries are looked up once in
OpenStreetMap Nominatim (structured query, 1 request per second, results cached in the same folder as
nominatim_cache.json). Data from OpenStreetMap contributors, ODbL.

GeoNames feature codes used:
    PPLC  capital of a country          -> country capital
    PPLA  seat of a first-order division -> state/province capital
    PPLA2 seat of a second-order division -> capital where the dataset's "state" is a county (England, Scotland,
          Ireland, Bouches-du-Rhone, Bolzano)
    any populated place (feature class P) -> city center (the town's GeoNames point)

Every match carries a `match_method`, so the table says how each row was found. Labels that differ between the
brewery data and GeoNames are mapped in the alias tables below; nothing outside them is guessed. Places that
cannot be matched are written with empty coordinates and method "unmatched".
"""
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import quality_core as qc  # noqa: E402

STATE = "state_province_harmonized"

# Country names in the brewery data -> ISO code used by GeoNames.
COUNTRY_ISO = {"England": "GB", "Scotland": "GB", "Netherlands": "NL", "Isle of Man": "IM", "South Korea": "KR"}
# England and Scotland are GeoNames first-order divisions of GB: their capital is that division's seat.
CONSTITUENT = {"England": "ENG", "Scotland": "SCT"}

# Dataset state label -> GeoNames first-order division name (same country).
ADMIN1_ALIAS = {
    ("Australia", "ACT"): "Australian Capital Territory", ("Australia", "NSW"): "New South Wales",
    ("Australia", "NT"): "Northern Territory", ("Australia", "QLD"): "Queensland",
    ("Australia", "SA"): "South Australia", ("Australia", "TAS"): "Tasmania",
    ("Australia", "VIC"): "Victoria", ("Australia", "WA"): "Western Australia",
    ("Austria", "Kärnten"): "Carinthia", ("Austria", "Niederösterreich"): "Lower Austria",
    ("Austria", "Oberösterreich"): "Upper Austria", ("Austria", "Steiermark"): "Styria",
    ("Belgium", "Brussels Capital Region"): "Brussels Capital",
    ("Finland", "Northern Ostrobothnia"): "North Ostrobothnia", ("Finland", "Varsinais-Suomi"): "Southwest Finland",
    ("Germany", "Bayern"): "Bavaria", ("Germany", "Berlin"): "State of Berlin", ("Germany", "Hessen"): "Hesse",
    ("Germany", "Niedersachsen"): "Lower Saxony", ("Germany", "Nordrhein-Westfalen"): "North Rhine-Westphalia",
    ("Germany", "Sachsen"): "Saxony", ("Germany", "Sachsen-Anhalt"): "Saxony-Anhalt",
    ("Germany", "Thüringen"): "Thuringia",
    ("New Zealand", "Bay of Plenty Region"): "Bay of Plenty", ("New Zealand", "Canterbury Region"): "Canterbury",
    ("New Zealand", "Gisborne Region"): "Gisborne", ("New Zealand", "Manawatū-Whanganui Region"): "Manawatu-Wanganui",
    ("New Zealand", "Marlborough Region"): "Marlborough", ("New Zealand", "Northland Region"): "Northland",
    ("New Zealand", "Otago Region"): "Otago", ("New Zealand", "Southland Region"): "Southland",
    ("New Zealand", "Tasman Region"): "Tasman District", ("New Zealand", "West Coast Region"): "West Coast",
    ("Netherlands", "Noord-Brabant"): "North Brabant", ("Netherlands", "Noord-Holland"): "North Holland",
    ("Netherlands", "Zuid-Holland"): "South Holland",
    ("Poland", "dolnośląskie"): "Lower Silesia",
    ("Portugal", "Açores"): "Azores", ("Portugal", "Lisboa"): "Lisbon",
    ("South Korea", "Chungcheongbukdo"): "North Chungcheong", ("South Korea", "Deagu"): "Daegu",
    ("Sweden", "Stockholm County"): "Stockholm",
    ("Ukraine", "Dnipropetrovsk Oblast"): "Dnipropetrovsk",
}

# Dataset state label -> (GeoNames first-order division, second-order division) where the dataset's "state" is a
# county or province one level below GeoNames' first level.
ADMIN2_ALIAS = {
    ("England", "East Sussex"): ("England", "East Sussex"), ("England", "West Sussex"): ("England", "West Sussex"),
    ("Scotland", "Argyll"): ("Scotland", "Argyll and Bute"), ("Scotland", "Bute"): ("Scotland", "Argyll and Bute"),
    ("Scotland", "East Dunbartonshire"): ("Scotland", "East Dunbartonshire"),
    ("Scotland", "West Dunbartonshire"): ("Scotland", "West Dunbartonshire"),
    ("France", "Bouche du Rhône"): ("Provence-Alpes-Côte d'Azur", "Bouches-du-Rhône"),
    ("Italy", "Bolzano"): ("Trentino-Alto Adige", "Bolzano"),
    ("Ireland", "Dublin"): ("Leinster", "Dublin City"), ("Ireland", "Longford"): ("Leinster", "An Longfort"),
    ("Ireland", "Offaly"): ("Leinster", "Uíbh Fhailí"), ("Ireland", "Mayo"): ("Connacht", "Mayo County"),
    ("Ireland", "Limerick"): ("Munster", "Limerick City and County Council"),
}
# Seat priority when a division has several flagged places: national capital, then first-, then second-order seat.
SEAT_PRIORITY = {"PPLC": 0, "PPLA": 1, "PPLA2": 2}
# Irish counties are GeoNames second-order divisions; their first-order parent (province) is looked up by name.
ADMIN2_BY_NAME_COUNTRIES = {"Ireland"}

# The whole "state" is the country itself (city-state) or lies in another GeoNames country.
STATE_IS_COUNTRY = {("Singapore", "Singapore"): "SG", ("Finland", "Åland"): "AX"}


def norm(s):
    """Lower-case ASCII without accents, spaces or punctuation; 'saint'/'st' unified."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bsaint\b|\bste?\b\.?", "st", s)
    return re.sub(r"[^a-z0-9]", "", s)


def load_geonames(folder):
    folder = Path(folder)
    rows = [line.rstrip("\n").split("\t") for line in open(folder / "countryInfo.txt", encoding="utf-8")
            if not line.startswith("#")]
    countries = pd.DataFrame([(r[0], r[4]) for r in rows], columns=["iso", "name"])
    a1 = pd.read_csv(folder / "admin1CodesASCII.txt", sep="\t", header=None, dtype=str, keep_default_na=False,
                     names=["code", "name", "ascii", "gid"], quoting=3)
    a2 = pd.read_csv(folder / "admin2Codes.txt", sep="\t", header=None, dtype=str, keep_default_na=False,
                     names=["code", "name", "ascii", "gid"], quoting=3)
    cols = ["gid", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "iso", "cc2", "a1", "a2", "a3", "a4",
            "population", "elevation", "dem", "tz", "modified"]
    places = pd.read_csv(folder / "cities500.txt", sep="\t", header=None, names=cols, dtype=str,
                         keep_default_na=False, quoting=3)
    places["lat"], places["lon"] = places["lat"].astype(float), places["lon"].astype(float)
    places["population"] = pd.to_numeric(places["population"], errors="coerce").fillna(0)
    return countries, a1, a2, places


def nominatim_fallback(cities, cache_path, iso, min_records=2):
    """Look up unmatched cities with >= `min_records` breweries in OpenStreetMap Nominatim (structured search).
    Only settlement-like results inside the right country are accepted."""
    import json
    import time
    import urllib.parse
    import urllib.request
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ok_types = {"city", "town", "village", "municipality", "hamlet", "suburb", "borough", "quarter",
                "neighbourhood", "city_district", "locality", "isolated_dwelling"}
    todo = cities[cities["city_latitude"].isna() & (cities["raw_records"] >= min_records)]  # noqa: E501
    for i, r in todo.iterrows():
        key = f"{r['country']}|{r[STATE]}|{r['city']}"
        if key not in cache:
            results = []
            for params in ({"city": r["city"], "state": r[STATE], "country": r["country"]},
                           {"city": r["city"], "country": r["country"]}):
                q = urllib.parse.urlencode({**params, "format": "jsonv2", "limit": 3, "addressdetails": 1})
                req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{q}",
                                             headers={"User-Agent": "geolysis-portfolio-reference-build/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    results = json.loads(resp.read().decode())
                time.sleep(1.1)
                if results:
                    break
            cache[key] = results
            cache_path.write_text(json.dumps(cache))
        code = (iso.get(r["country"]) or "").lower()
        good = [x for x in cache[key] if x.get("addresstype") in ok_types
                and x.get("address", {}).get("country_code", "").lower() == code]
        if good:
            x = good[0]
            cities.loc[i, ["geonames_name", "city_latitude", "city_longitude", "match_method"]] = [
                x.get("name"), float(x["lat"]), float(x["lon"]), f"OpenStreetMap Nominatim ({x.get('addresstype')})"]
    return cities


def build(folder, use_nominatim=False):
    countries, a1, a2, places = load_geonames(folder)
    iso = dict(zip(countries["name"], countries["iso"]))
    iso.update(COUNTRY_ISO)
    a1["iso"], a1["a1"] = a1["code"].str.split(".").str[0], a1["code"].str.split(".").str[1]
    a2["iso"] = a2["code"].str.split(".").str[0]
    a2["a1"], a2["a2"] = a2["code"].str.split(".").str[1], a2["code"].str.split(".").str[2]
    a1_by_name = {(r.iso, norm(n)): r.a1 for r in a1.itertuples() for n in (r.name, r.ascii)}
    a1_name = {(r.iso, r.a1): r.name for r in a1.itertuples()}

    with sqlite3.connect(ROOT / "data" / "raw_breweries.db") as con:
        raw = qc.harmonize_states(pd.read_sql_query("SELECT * FROM raw_breweries", con))

    # ---------------- country capitals
    crow = []
    for c in sorted(raw["country"].unique()):
        code = iso.get(c)
        if c in CONSTITUENT:
            a1c = CONSTITUENT[c]
            cand = places[(places["iso"] == code) & (places["a1"] == a1c) & places["fcode"].isin(["PPLA", "PPLC"])]
            method = "seat of GeoNames first-order division " + a1c
        else:
            cand = places[(places["iso"] == code) & (places["fcode"] == "PPLC")]
            method = "GeoNames PPLC"
        cand = cand.sort_values("population", ascending=False)
        if len(cand):
            p = cand.iloc[0]
            crow.append({"country": c, "iso": code, "capital": p["name"], "capital_latitude": p["lat"],
                         "capital_longitude": p["lon"], "geonameid": p["gid"], "match_method": method})
        else:
            crow.append({"country": c, "iso": code, "capital": None, "capital_latitude": None,
                         "capital_longitude": None, "geonameid": None, "match_method": "unmatched"})
    country_capitals = pd.DataFrame(crow)

    # ---------------- state capitals (and the GeoNames admin codes of every dataset state)
    srow = []
    for (c, s) in sorted(raw[["country", STATE]].drop_duplicates().itertuples(index=False)):
        code = iso.get(c)
        a1c = a2c = None
        method = "unmatched"
        if (c, s) in STATE_IS_COUNTRY:
            other = STATE_IS_COUNTRY[(c, s)]
            cand = places[(places["iso"] == other) & (places["fcode"] == "PPLC")]
            method = f"state is the territory {other}: its GeoNames PPLC"
        elif (c, s) in ADMIN2_ALIAS or c in ADMIN2_BY_NAME_COUNTRIES:
            if (c, s) in ADMIN2_ALIAS:
                p1, p2 = ADMIN2_ALIAS[(c, s)]
                a1c = a1_by_name.get((code, norm(p1)))
                hit = a2[(a2["iso"] == code) & (a2["a1"] == a1c) & (a2["name"].map(norm) == norm(p2))]
            else:
                hit = a2[(a2["iso"] == code) & ((a2["name"].map(norm) == norm(s))
                                                | (a2["name"].map(norm) == norm("County " + s)))]
            if len(hit) == 1:
                a1c, a2c = hit.iloc[0]["a1"], hit.iloc[0]["a2"]
                cand = places[(places["iso"] == code) & (places["a1"] == a1c) & (places["a2"] == a2c)
                              & places["fcode"].isin(list(SEAT_PRIORITY))]
                cand = cand.assign(_p=cand["fcode"].map(SEAT_PRIORITY)).sort_values(["_p", "population"],
                                                                                   ascending=[True, False])
                method = "GeoNames seat of second-order division" + (" via alias" if (c, s) in ADMIN2_ALIAS else "")
            else:
                cand = places.iloc[0:0]
        else:
            target = ADMIN1_ALIAS.get((c, s), s)
            a1c = a1_by_name.get((code, norm(target)))
            if a1c:
                cand = places[(places["iso"] == code) & (places["a1"] == a1c) & (places["fcode"] == "PPLA")]
                method = "GeoNames PPLA" + (" via alias" if (c, s) in ADMIN1_ALIAS else "")
                for fallback in ("PPLG", "PPLC"):  # seat of government (e.g. The Hague), or the national capital
                    if cand.empty:
                        cand = places[(places["iso"] == code) & (places["a1"] == a1c) & (places["fcode"] == fallback)]
                        method = f"GeoNames {fallback} inside the division" + (" via alias" if (c, s) in ADMIN1_ALIAS else "")
            else:
                cand = places.iloc[0:0]
        if "_p" not in cand:
            cand = cand.sort_values("population", ascending=False)
        row = {"country": c, STATE: s, "iso": code, "geonames_admin1": a1c, "geonames_admin2": a2c,
               "geonames_division": a1_name.get((code, a1c)) if a1c else None,
               "capital": None, "capital_latitude": None, "capital_longitude": None, "geonameid": None,
               "candidates": len(cand), "match_method": "unmatched"}
        if len(cand):
            p = cand.iloc[0]
            row.update({"capital": p["name"], "capital_latitude": p["lat"], "capital_longitude": p["lon"],
                        "geonameid": p["gid"], "match_method": method})
        srow.append(row)
    state_capitals = pd.DataFrame(srow)

    # ---------------- city centers
    places["n_name"], places["n_ascii"] = places["name"].map(norm), places["ascii"].map(norm)
    alt = places[["gid", "alt"]].assign(alt=places["alt"].str.split(",")).explode("alt")
    alt = alt[alt["alt"].str.len() > 0]
    alt["n_alt"] = alt["alt"].map(norm)
    by_gid = places.set_index("gid")
    keys = pd.concat([places[["gid", "n_name"]].rename(columns={"n_name": "key"}).assign(rank=0),
                      places[["gid", "n_ascii"]].rename(columns={"n_ascii": "key"}).assign(rank=0),
                      alt[["gid", "n_alt"]].rename(columns={"n_alt": "key"}).assign(rank=1)]).drop_duplicates(["gid", "key"])
    keys = keys.join(by_gid[["iso", "a1", "a2", "population", "lat", "lon", "name", "fclass"]], on="gid")
    keys = keys[keys["fclass"] == "P"]
    sc = state_capitals.set_index(["country", STATE])
    cities = raw.groupby(["country", STATE, "city"]).size().rename("raw_records").reset_index()
    located = raw.assign(latitude=pd.to_numeric(raw["latitude"], errors="coerce"),
                         longitude=pd.to_numeric(raw["longitude"], errors="coerce")).dropna(subset=["latitude", "longitude"])
    own = located.groupby(["country", STATE, "city"])[["latitude", "longitude"]].mean()
    from geographic_utils import haversine_km
    crow = []
    for r in cities.itertuples(index=False):
        code = iso.get(r.country)
        st = sc.loc[(r.country, getattr(r, STATE))]
        k = norm(r.city)
        pool = keys[(keys["key"] == k) & (keys["iso"] == (STATE_IS_COUNTRY.get((r.country, getattr(r, STATE))) or code))]
        method = None
        if st["geonames_admin1"] and len(pool):
            same = pool[pool["a1"] == st["geonames_admin1"]]
            if st["geonames_admin2"] and len(same):
                same2 = same[same["a2"] == st["geonames_admin2"]]
                same = same2 if len(same2) else same
            if len(same):
                pool, method = same, "name + state"
        if method is None and len(pool):
            method = "only found in another state"  # a match outside the recorded state would be wrong
        if method == "name + state":
            n_cand = pool["gid"].nunique()
            key3 = (r.country, getattr(r, STATE), r.city)
            if n_cand > 1 and key3 in own.index:
                # several same-name places in the right state: take the one nearest to the city's own breweries
                o = own.loc[key3]
                pool = pool.assign(_d=haversine_km(pool["lat"], pool["lon"], o["latitude"], o["longitude"]))
                best = pool.sort_values(["_d", "rank"]).iloc[0]
                method = f"name + state, nearest of {n_cand} same-name places"
            else:
                best = pool.sort_values(["rank", "population"], ascending=[True, False]).iloc[0]
                if n_cand > 1:
                    method = f"name + state, largest of {n_cand} same-name places"
            crow.append({"country": r.country, STATE: getattr(r, STATE), "city": r.city, "raw_records": r.raw_records,
                         "geonames_name": best["name"], "city_latitude": best["lat"], "city_longitude": best["lon"],
                         "geonameid": best["gid"], "population": int(best["population"]),
                         "match_method": method + (" (alternate name)" if best["rank"] == 1 else "")})
        else:
            crow.append({"country": r.country, STATE: getattr(r, STATE), "city": r.city, "raw_records": r.raw_records,
                         "geonames_name": None, "city_latitude": None, "city_longitude": None, "geonameid": None,
                         "population": None, "match_method": method or "unmatched"})
    city_centers = pd.DataFrame(crow)
    if use_nominatim:
        city_centers = nominatim_fallback(city_centers, Path(folder) / "nominatim_cache.json", iso)
    # sanity check: distance between the looked-up center and the mean position of the city's own breweries
    geo = raw.assign(latitude=pd.to_numeric(raw["latitude"], errors="coerce"),
                     longitude=pd.to_numeric(raw["longitude"], errors="coerce")).dropna(subset=["latitude", "longitude"])
    mean = geo.groupby(["country", STATE, "city"])[["latitude", "longitude"]].mean().rename(
        columns={"latitude": "brewery_mean_latitude", "longitude": "brewery_mean_longitude"})
    city_centers = city_centers.join(mean, on=["country", STATE, "city"])
    from geographic_utils import haversine_km
    city_centers["center_to_brewery_mean_km"] = haversine_km(
        city_centers["city_latitude"], city_centers["city_longitude"],
        city_centers["brewery_mean_latitude"], city_centers["brewery_mean_longitude"])

    sources = pd.DataFrame([{
        "source": "GeoNames", "url": "https://www.geonames.org", "license": "CC BY 4.0",
        "files": "cities500.txt, admin1CodesASCII.txt, admin2Codes.txt, countryInfo.txt",
        "built": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "note": "Capitals from feature codes PPLC/PPLA/PPLA2; city centers are the GeoNames point of the populated "
                "place. cities500 holds places with at least 500 inhabitants (and all seats of divisions)."}]
                           + ([{"source": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org",
                                "license": "ODbL (data © OpenStreetMap contributors)", "files": "search API",
                                "built": pd.Timestamp.now().strftime("%Y-%m-%d"),
                                "note": "Fallback for cities with at least 2 breweries that GeoNames could not place."}]
                              if use_nominatim else []))
    out = ROOT / "data" / "reference_places.db"
    with sqlite3.connect(out) as con:
        for name, t in {"country_capitals": country_capitals, "state_capitals": state_capitals,
                        "city_centers": city_centers, "sources": sources}.items():
            t.to_sql(name, con, if_exists="replace", index=False)
    return country_capitals, state_capitals, city_centers


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    cc, sc, ct = build(args[0], use_nominatim="--nominatim" in sys.argv)
    print(f"country capitals: {cc['capital'].notna().sum()} / {len(cc)}")
    print(f"state capitals:   {sc['capital'].notna().sum()} / {len(sc)}")
    print(f"city centers:     {ct['city_latitude'].notna().sum()} / {len(ct)} cities, "
          f"{ct.loc[ct['city_latitude'].notna(), 'raw_records'].sum()} / {ct['raw_records'].sum()} raw records")
    print(ct["match_method"].value_counts().to_string())
