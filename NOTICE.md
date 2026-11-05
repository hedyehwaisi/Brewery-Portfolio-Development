# Third-party notices

This project, *Brewery Portfolio Development* (Copyright (c) 2026 Hedyeh Waisi), bundles and
builds on data and services from the parties below. Their notices are reproduced here so that they
travel with every copy of the repository, as their licences require. See `LICENSE` (code, MIT),
`LICENSE-DOCS.md` (text and figures, CC BY 4.0) and `data/LICENSE.md` (data) for the terms that
apply to the author's own work.

## 1. Open Brewery DB

- Website: https://www.openbrewerydb.org/
- Dataset repository: https://github.com/openbrewerydb/openbrewerydb
- Used for: the 11,959 brewery records in `data/raw_breweries.db` and every table derived from them.
- Licence: MIT License. Notice reproduced verbatim:

```
MIT License

Copyright (c) 2025 Open Brewery DB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. GeoNames

- Website: https://www.geonames.org
- Files used: `cities500.txt`, `admin1CodesASCII.txt`, `admin2Codes.txt`, `countryInfo.txt`
  from https://download.geonames.org/export/dump/ (downloaded 2026-10-30).
- Used for: country, state and province capitals and city center points in `data/reference_places.db`,
  and the distance measures built on them.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Credit: **Data from GeoNames (https://www.geonames.org), licensed under CC BY 4.0.**
  GeoNames data is provided "as is" without warranty.

## 3. OpenStreetMap

- Website: https://www.openstreetmap.org — copyright and licence page:
  https://www.openstreetmap.org/copyright
- Licence: Open Data Commons Open Database License 1.0 (ODbL),
  https://opendatacommons.org/licenses/odbl/1-0/
- Credit: **© OpenStreetMap contributors.**
- Used for:
  - **Nominatim geocoding** (https://nominatim.openstreetmap.org): center points of 68 cities that
    GeoNames could not place, stored in `data/reference_places.db`, and a handful of reverse-geocoded
    addresses quoted as evidence for coordinate corrections. Lookups followed the Nominatim usage
    policy (https://operations.osmfoundation.org/policies/nominatim/): identifying User-Agent, at most
    one request per second, results cached and not re-requested.
  - **Map tiles**: the "OpenStreetMap" basemap in the dashboard draws the standard OSM raster tiles
    (https://operations.osmfoundation.org/policies/tiles/).
- Because `data/reference_places.db` contains OpenStreetMap data, that database is offered under
  the ODbL (see `data/LICENSE.md`).

## 4. CARTO

- Website: https://carto.com — attribution page: https://carto.com/attributions
- Used for: the "CARTO Positron (light grey)" basemap in the dashboard, served from CARTO's free
  basemap tiles through Plotly.
- Credit: **© CARTO**, together with **© OpenStreetMap contributors** for the underlying map data.
  CARTO's free basemaps require this credit to stay visible on every map that uses them
  (https://carto.com/legal/basemap-terms/).

## 5. Software libraries

The code depends on open-source Python packages (pandas, NumPy, SciPy, scikit-learn, hdbscan,
Streamlit, Plotly and others listed in `requirements.txt`). They are not bundled in this repository;
each is distributed under its own licence from PyPI.

## How the credits are shown

- In the repository: this file, `README.md` and `data/LICENSE.md`.
- In the dashboard: a credit line in the sidebar of every page, and the tile credit Plotly prints on
  every map.
- In `data/reference_places.db`: the `sources` table.
