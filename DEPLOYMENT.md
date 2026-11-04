# Reproducibility and deployment guide

How to run the dashboard locally, regenerate its data, and host it, with the memory and CPU figures we measured to
choose a plan.

## Run it locally

Commands run from the repository root. Render reads the Python version from `.python-version` (3.12.8); we developed
and measured with Python 3.13.7, and both work with the pinned packages.

### macOS / Linux / WSL

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-deploy.txt
python -m streamlit run app.py
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-deploy.txt
python -m streamlit run app.py
```

If script activation is blocked, call the venv's Python directly: `.\.venv\Scripts\python.exe -m streamlit run app.py`.

Open the URL Streamlit prints (normally `http://localhost:8501`). The SQLite files in `data/` are all the dashboard
needs; it calls no API and fits no model while you browse.

`requirements-deploy.txt` holds only what the dashboard imports: NumPy, pandas, Streamlit and Plotly.
`requirements.txt` adds what the notebooks and tests need (SciPy, scikit-learn, `hdbscan`, Jupyter, `pytest`, …).

## Tests

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

95 tests check the helper logic (harmonized labels, coordinate corrections, distances, type statistics, prediction
model) and the integrity of the stored results: every clustering run is present, its labels reproduce its stored
metrics, and the record order matches the analysis tables. They also check that `data/dashboard_stats.db` was built
from the current input files and equals a fresh computation, that the NumPy replacements (convex hull, adjusted Rand
index, kernel density) match SciPy and scikit-learn, and that running every page loads neither library. They do not
rerun the notebooks.

## Regenerating the data

Run the notebooks in this order. Each one reads what the previous ones wrote.

| Step | Notebook / script | Writes | Last run (8-core laptop) |
|---|---|---|---|
| 1 | `merged_brewery_cleaned_for_eda.ipynb` | `raw_breweries.db`, `analysis_all_records.db`, `analysis_unique_records.db` | not timed |
| 2 | `tools/build_reference_places.py` | `reference_places.db` from GeoNames files you download first (the script's docstring lists them); `--nominatim` adds the fallback lookups | not timed |
| 3 | `brewery_geographic_analysis.ipynb` | `results.db` (first-pass geographic tables) | not timed |
| 4 | `brewery_clustering_all_runs.ipynb` | `coordinate_corrections.csv` and the `allruns_*` tables of `clustering_results.db` | 5 min 52 s |
| 5 | `brewery_prediction.ipynb` | `prediction_results.db` | 1 min 25 s |
| 6 | `brewery_dashboard_statistics.ipynb` | `dashboard_stats.db`: the tests the dashboard shows that need SciPy or scikit-learn, for every page setting | 1 min 4 s |

Run them with `jupyter nbconvert --to notebook --execute --inplace <notebook>`, then run the tests.

`brewery_clustering_revised.ipynb` is the historical first clustering pass. **Do not run its last cell against
`data/clustering_results.db`**: it drops every table in that file, the `allruns_*` tables included, before writing its
own.

No GPU is used. The clustering needs Haversine distances, which the GPU library we checked (cuML) does not offer for
these algorithms, and the CPU run takes minutes.

## Render deployment

The public dashboard runs on Render as a native Python web service, configured in `render.yaml`:

- build: `pip install -r requirements-deploy.txt`
- start: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- health check: `/_stcore/health`
- environment: `ARROW_DEFAULT_MEMORY_POOL=jemalloc` and `MALLOC_ARENA_MAX=2` (see below)
- Python: from `.python-version`

With the Blueprint flow, check the service name and `plan` in `render.yaml` before creating the service. A service
configured by hand can use the same commands and environment variables in its settings.

### What the app needs: measurements

We started the app as Render does (`streamlit run`, headless, in a fresh environment with `requirements-deploy.txt`
only) and drove it with websocket clients. Each simulated visitor opens all nine pages with their default settings,
and several visitors run at the same time. We then read the server's resident memory. `tools/memory_probe.py` repeats
the test.

| Load | Phase 8 start | After shared caches | Now |
|---|---|---|---|
| Server idle | 75 MB | 75 MB | 75 MB |
| 1 visitor, all pages | 440–453 MB | 380–398 MB | 267–270 MB |
| 3 visitors at once | 539 MB | 465–471 MB | 333–363 MB |
| 6 visitors at once | 770–833 MB | 597–614 MB | 413–422 MB |
| 10 visitors at once | — | — | 449–486 MB |

Each range covers two cases: a server that started cold with all visitors arriving together, and a server that had
already served one visitor. The libraries the dashboard imports (pandas with pyarrow, Streamlit, Plotly) take about
140 MB; the rest is cached data plus each visitor's working memory.

CPU on the same machine: a first visitor after a restart costs about 14 CPU-seconds for all nine pages (2.8 for the
opening page, 2.1 for Prediction), a later visitor about 7, because the cached tables are then reused.

### What we changed to fit 512 MB

1. **Shared caches instead of copied caches.** Every cached loader now uses `common.shared_cache`, a wrapper around
   `st.cache_resource`. It keeps one copy of each table for all visitors. `st.cache_data` kept a pickled copy plus a
   fresh copy for every call. Callers get a shallow copy: pandas copy-on-write shares the data until someone writes,
   so a page that changes its copy cannot change the cache. This change gave most of the saving.
2. **Bounded per-run caches.** The clustering page keeps at most 64 runs in each per-run cache, so a visitor browsing
   hundreds of runs cannot grow memory without limit.
3. **No SciPy or scikit-learn at run time.** The statistics that need them (chi-square tests, Spearman correlations,
   Ward grouping of states, the co-location permutation test) are computed for every setting the pages offer by
   `brewery_dashboard_statistics.ipynb` and stored in `data/dashboard_stats.db` (1.7 MB); the pages read them. The
   convex hull, the adjusted Rand index and the kernel density are now NumPy code that tests check against the
   libraries. scikit-learn is gone from `requirements-deploy.txt`, which also makes the build smaller. This saved
   about 80 MB of imports.
4. **pyarrow's jemalloc pool (`ARROW_DEFAULT_MEMORY_POOL=jemalloc`).** Streamlit serializes every table it sends to the
   browser with pyarrow. pyarrow's default pool (mimalloc) kept that memory; with jemalloc, six visitors at once on a
   warm server need 413–416 MB instead of 532–534 MB (two runs each). The pool ships with the pyarrow wheels for Linux.
5. **`MALLOC_ARENA_MAX=2`.** Streamlit runs each visitor in its own thread, and glibc gives threads their own memory
   arenas. With the settings above, capping them saves another 15 MB or so with six visitors.
6. **The kernel density runs in small chunks.** 25 grid points at a time keep its temporary arrays at a few MB
   instead of about 50 MB, at the same speed.

### If we stay on Render

Plans as listed by Render on 2026-10-30 (check [render.com/pricing](https://render.com/pricing) before deciding):

| Plan | CPU / RAM | Price | What it means for this app |
|---|---|---|---|
| Free | 0.1 CPU / 512 MB | $0 | Memory is no longer the limit: ten visitors touring every page at once stay under 490 MB. CPU is: at 0.1 CPU the opening page takes roughly 30 s after a restart (2.8 CPU-s ÷ 0.1; an estimate, CPUs differ), and visitors share that tenth of a core. Spins down after 15 minutes without traffic and needs about a minute to wake |
| Starter | 0.5 CPU / 512 MB | $7 / month | No spin-down and pages about five times faster than Free; the same 512 MB, which held ten simultaneous visitors in our test |
| Standard | 1 CPU / 2 GB | $25 / month | Twice Starter's speed and far more memory than the app needs |

Our recommendation:

1. **Portfolio link, light traffic: Free works.** Expect the wake-up minute and a slow first page after an idle
   period. Say so next to the link ("may take a minute to wake up").
2. **Shared publicly (a LinkedIn post, a job application): Starter for that period.** Its memory now holds the burst a
   post can bring, and it removes the wake-up wait and most of the slowness. Standard is only needed if the traffic is
   much larger than that.
3. **Do not keep a Free service awake with scheduled pings.** A service awake all month uses about 744 of the 750 free
   instance hours in a workspace, which leaves nothing for any other free service, and it still has only 0.1 CPU.

### Measured and left as it is

- **Lighter maps.** The dark ring under every marker is drawn as a second trace, so maps carry each point twice.
  Turning the rings off saved only 7–8 MB with six visitors (two runs each), and the rings are what keep markers
  readable on both basemaps, so they stay. All figures of a full tour add up to about 9 MB, already sent as binary
  arrays.
- **jemalloc for Python itself** (`LD_PRELOAD` in a Docker image) may shave a little more off the glibc side. We have
  not tested it; `tools/memory_probe.py` can, once the Docker image exists.

### Docker later

Render builds from a Dockerfile too. The same measurements apply: base the image on `python:3.12-slim`, install
`requirements-deploy.txt` only, set `ARROW_DEFAULT_MEMORY_POOL=jemalloc` and `MALLOC_ARENA_MAX=2`, expose `$PORT`
with the start command above, and copy `data/`, `pages/` and the Python modules. The memory figures above are for the
app, not the platform, so a container needs the same plan size.

## Other free option

Streamlit Community Cloud (checked 2026-10-30) gives between about 690 MB and 2.7 GB of memory and up to 2 CPU cores,
hibernates an app after 12 hours without traffic, can deploy a private repository once GitHub access is granted, and
does not run Docker images. It gives more CPU than Render's free tier; Render remains the better fit for a
Docker-based setup.

## Publishing changes

Work happens on the `optimalism` branch of
[hedyehwaisi/Brewery-Portfolio-Development](https://github.com/hedyehwaisi/Brewery-Portfolio-Development). Before
pushing:

1. Run the tests and open the dashboard locally.
2. Check `git status`: no virtual environments, caches, secrets or scratch files.
3. Keep the SQLite files: the dashboard and the tests need them. GitHub warns above 50 MB per file and refuses
   100 MB; `clustering_results.db` is about 30 MB.
4. Respect the data licences: Open Brewery DB, GeoNames (CC BY 4.0) and OpenStreetMap Nominatim (ODbL) are credited in
   the README, and `reference_places.db` records its sources in the `sources` table.
