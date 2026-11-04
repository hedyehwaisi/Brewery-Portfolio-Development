# Data dictionary and metric glossary

What each field, file and derived measure means in this project, and what it cannot tell you. Field
availability differs between the raw snapshot and the two geocoded analysis datasets.

## Source fields

| Field | Meaning in this project | Main use | Caveat |
|---|---|---|---|
| `id` | Open Brewery DB record identifier | Record identity, duplicate checks, aligning stored cluster labels with records, record-level fixes | A unique id does not prove the other fields are right |
| `name` | Brewery or business name | Tables, map labels, identity checks, names vs websites | Spelling, punctuation and legal form vary |
| `brewery_type` | Category supplied by Open Brewery DB (micro, brewpub, regional, large, planning, closed, …) | Type counts, type mix, prediction of type | Labels follow country conventions (Germany is 87% brewpub, Australia 96% micro), so they are not comparable across countries |
| `address_1`, `address_2`, `address_3`, `postal_code` | Address fields | Identity checks, shared-coordinate analysis | Often missing; `address_2`/`address_3` are almost always empty |
| `street` | Deprecated copy of `address_1` | Data-quality check only | Identical to `address_1` on every raw record |
| `city` | City or locality label | City counts, hierarchy, city-center check | The same name exists in several states; we always group by city and state |
| `state` | Deprecated copy of `state_province` | Data-quality check only | Identical to `state_province` on all 11,959 raw records after trimming and ignoring case |
| `state_province` | State, province or region label as supplied | Kept untouched as the source value | Not harmonized: Belgian regions appear in three languages and one US record says `MIssouri`; use `state_province_harmonized` for grouping |
| `country` | Country label | Country counts and hierarchy | 23 countries in the raw snapshot, 22 with coordinates (Ukraine's one record has none); very uneven (the United States holds 63.6% of geocoded records) |
| `latitude`, `longitude` | Coordinates in decimal degrees | Maps, Haversine distances, clustering, density | 2,468 raw records have none, and the gaps are not random; a few stored values are wrong (see `coordinate_status`) |
| `phone` | Phone number | Identity checks, shared-value groups | Missing or formatted inconsistently |
| `website_url` | Website address | Identity checks, names vs websites | Missing for some records; some point to social-media pages |

## Columns we add

| Column | Where it comes from | Meaning |
|---|---|---|
| `state_province_harmonized` | `quality_core.harmonize_states` | One label per real region: Belgian region names in Dutch, French and English merged (Flanders, Wallonia), `MIssouri` → Missouri, and one record-level fix (Frontier Beer Company, Salt River in Cape Town, labelled Gauteng → Western Cape). Every state-level grouping uses it |
| `coordinate_status` | `quality_core.apply_corrections` | `as recorded`; `corrected: latitude/longitude swapped back`; `corrected: coordinate sign flipped back`; `corrected: copied from identical record`; or `flagged: far from its city` (position kept) |

## Files

| File | Contents |
|---|---|
| `data/raw_breweries.db` | The snapshot: 11,959 records, including the 2,468 without coordinates |
| `data/analysis_all_records.db` | 9,488 geocoded records after removing 3 exact duplicates |
| `data/analysis_unique_records.db` | 9,487 records: the same, with one Paulaner record that differs only by its website removed |
| `data/coordinate_corrections.csv` | The corrections log written by `brewery_clustering_all_runs.ipynb`: one row per record the coordinate check caught, with original and corrected coordinates, the action and the evidence. Applied when data is loaded; the databases keep their original values |
| `data/reference_places.db` | Reference points from GeoNames (CC BY 4.0) with an OpenStreetMap Nominatim fallback (ODbL): `country_capitals`, `state_capitals`, `city_centers`, each with its match method, and `sources`. Built by `tools/build_reference_places.py` |
| `data/results.db` | First-pass geographic tables (centers, dispersion, outlier flags, coordinate groups) from `brewery_geographic_analysis.ipynb`, on uncorrected coordinates |
| `data/clustering_results.db` | `allruns_*` tables from `brewery_clustering_all_runs.ipynb` (below), plus the summary tables of the first clustering pass |
| `data/prediction_results.db` | Tuned settings and validation results from `brewery_prediction.ipynb` (below) |
| `data/dashboard_stats.db` | Statistics the dashboard shows that need SciPy or scikit-learn, computed for every page setting by `brewery_dashboard_statistics.ipynb` (below) |

### `allruns_*` tables (clustering)

| Table | Contents |
|---|---|
| `allruns_runs` | One row per run (1,266): algorithm, parameters, cluster count, clustered and noise share, cluster sizes, spread, density, how many clusters cross a state or country border, hub merges, stability |
| `allruns_labels` | Cluster label of every record for every run, as a zlib-compressed int16 array (`-1` = noise) |
| `allruns_records` | Record order of the label arrays per dataset (`position`, `id`, `coordinate_status`) |
| `allruns_hdbscan_extras` | Per HDBSCAN run: membership probabilities and GLOSH scores (uint8), cluster persistence, the condensed tree (float32) |
| `allruns_optics` | Per OPTICS `min_samples`: ordering, reachability and core distance in km |
| `allruns_dbscan_star` | DBSCAN against a flat cut of the HDBSCAN tree at the same `eps` (ARI on all and on core points, border points) |
| `allruns_hdbscan_check` | scikit-learn HDBSCAN against the `hdbscan` library on the same settings (cluster counts, ARI) |
| `allruns_settings`, `allruns_verification` | Grid definitions and a row-count check of every table written |

### Prediction tables

| Table | Contents |
|---|---|
| `cv_places` | 5-fold cross-validation of the place model for every smoothing `alpha`, at state and city level, with the uniform baseline |
| `cv_types` | The same for the type model and every shrinkage `beta`, against state-only and global type mixes |
| `summary` | Headline log-loss and top-k accuracy of each task against its baseline |
| `pipeline_by_state` | Operating, planning and closed records per state, shares and lift |
| `planning_test` | The model trained on operating breweries, scored on planning ones, incl. the new-city prediction |
| `kde_cv` | Held-out log-likelihood of the density map for every bandwidth |
| `params` | The chosen settings: `alpha` 0.05, `beta` 20, bandwidth 30 km |

### Dashboard statistics tables

| Table | Contents |
|---|---|
| `missingness` | Chi-square test and Cramér's V of each column's gaps against brewery type, country, state and city, per dataset |
| `spearman` | Per state: Spearman correlation of a city's distance from the brewery center / capital with its share of the state |
| `type_association` | Chi-square test of brewery type against place, per dataset, level and minimum records (5–100) |
| `type_profile_states`, `type_profile_mix`, `type_profile_silhouette` | Ward groups of states by type mix: each state's group, each group's mean type shares, the silhouette score per number of groups |
| `type_colocation` | Observed and expected share of same-type nearest neighbours, per dataset, country and same-point option |
| `inputs` | SHA-256 of every input file at build time (the tests compare it with the current files) |

## Geographic measures

| Measure | Definition | Limitation |
|---|---|---|
| Coordinate coverage | Share of records with both latitude and longitude | Coverage differs strongly by type and place |
| Bounding box | Minimum/maximum latitude and longitude | Not an occupied area; spans across the antimeridian look huge |
| Origin 1: brewery center | Mean latitude and longitude of a place's breweries | Pulled by a single wrong record; can fall where no brewery is |
| Origin 2: reference point | National or state capital (GeoNames), or at city level the real city center | 207 of 209 state capitals and about 95% of raw records' cities are matched |
| Haversine distance | Great-circle distance with Earth radius 6,371.0088 km | Sphere, not ellipsoid; fine at this scale |
| Median, IQR, P90 distance | Robust spread of distances to an origin | Depends on geocoded records only |
| Share within 10/25/50/100 km | Share of a place's breweries within that distance of the origin | Large states look dispersed by definition |
| Mean-to-median ratio | Mean distance / median distance; read automatically as ≈ 1 (±0.1), above or below | Undefined when the median is 0 |
| Geographic outlier flag | Distance to the state center above Q3 + 1.5 × IQR of that state | A statistical flag, not proof of an error; unstable for small states |
| Coordinate check | Distance to the record's own city center (or state capital): `latitude/longitude swapped` (>500 km, <25 km once exchanged), `sign error` (>500 km, close once one sign is reversed), `far from its city` (>50 km) | Records whose city has no reference point are only checked for sign errors |

## Distribution measures

| Measure | Definition | Limitation |
|---|---|---|
| Location quotient | A type's share in a place / its share overall (1 = the overall mix) | Unstable for small places, so places below a minimum count are dropped |
| Chi-square and Cramér's V | Is the type mix the same in every place, and how strong is the difference (0–1) | Sparse categories are pooled; significance is easy with this many records |
| Type profiles | States as vectors of type shares, square-rooted (Hellinger), grouped with Ward clustering; group count by silhouette | Groups describe labelling as much as business models |
| Same-type neighbours | Share of breweries whose nearest neighbour has the same type, against permutations within each state | Stacked records at one point can inflate it (optionally excluded) |
| Gini, HHI, effective groups | Inequality of group sizes; HHI = sum of squared shares | Depends on how places are labelled |
| Zipf exponent | Slope of log(size) against log(rank) over groups with at least 3 records | Only a summary of the tail |
| Type A / Type B city | Top city whose state is / is not among the largest states | Thresholds are settings, not facts |

## Clustering measures

| Measure | Meaning | What it cannot establish alone |
|---|---|---|
| `eps_km` (DBSCAN) | Neighbourhood radius in km, converted to radians for the Haversine metric | Whether two hubs inside it are one region |
| `min_samples` | Points (the point itself included) needed in a neighbourhood for a core point | A universal definition of a hub |
| `min_cluster_size` (HDBSCAN) | Smallest group HDBSCAN accepts as a cluster | That smaller groups do not matter |
| `xi` (OPTICS) | Minimum steepness of the reachability plot that marks a cluster boundary | A density threshold in km |
| Noise share | Share of records labelled `-1` | That noise records are wrong |
| Spread | Median, P90 and maximum distance of members to the cluster center | Shape (a chain can have a modest median) |
| Hull density | Records per km² of the cluster's convex hull | Population or market size |
| Hub merge | A cluster holding ≥ 3 records from each of two cities with ≥ 10 breweries that are > 100 km apart (all adjustable on the page) | Whether the merge is wrong for every purpose |
| Persistence | How long an HDBSCAN cluster survives in the condensed tree | Geographic validity |
| GLOSH | HDBSCAN outlier score per record (0 = typical, 1 = isolated) | Coordinate errors |
| Core distance | Distance to the `min_samples`-th neighbour; small = dense area | Density in breweries per person |
| DBSCAN* | DBSCAN without border points, read off the HDBSCAN tree at the same `eps` | Clusters built around a single core point, which it leaves as noise (otherwise it matches DBSCAN on core points in all 528 settings) |
| Neighbour ARI | Mean adjusted Rand index with the runs one grid step away | Which clusters changed |
| Cluster recurrence | Best Jaccard overlap of each cluster with any cluster of a neighbouring run; a cluster is stable at ≥ 0.5 | Stability under data changes |

## Prediction measures

| Measure | Meaning | Limitation |
|---|---|---|
| Log-loss | Mean −log probability given to the true answer on held-out records (lower is better) | Comparable only on the same records |
| Top-1 / top-3 accuracy | Share of held-out records whose true answer is the model's first / among its first three choices | Ignores how confident the model was |
| Smoothing `alpha` | Pseudo-count added to every known place | Chosen by cross-validation |
| Shrinkage `beta` | Weight pulling a city's type mix towards its state's | Chosen by cross-validation |
| Good–Turing new-place share | (N1 + 1) / (N + 1): the chance that the next brewery opens in a place with none yet, N1 = places with one record | Assumes the future looks like the snapshot |
| Pipeline lift | A state's share of planning breweries / its share of operating breweries | Planning records are a proxy for openings, not dated events |
| KDE bandwidth | Width of the kernel of the density map, chosen by held-out likelihood (30 km). The density is summed exactly over every brewery | Describes where breweries are, not where they will be |

## Interpretation rule

Keep **record counts**, **geographic concentration**, **business performance** and **real-world population** apart.
The data supports descriptive analysis of recorded brewery locations. It does not measure sales, production, market
share or the reasons behind a location, and its predictions are probabilities learned from the snapshot, not
forecasts of dated openings.
