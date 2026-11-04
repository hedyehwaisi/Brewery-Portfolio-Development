"""The running dashboard never imports SciPy or scikit-learn (about 80 MB of memory per server process).

Every page is run headless in a fresh interpreter; afterwards neither library may be in sys.modules. A static check
also keeps pages from calling the functions that need them; their results come from data/dashboard_stats.db.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py"))
NEEDS_SCIPY = {"type_association", "type_profiles", "type_colocation", "missingness_association", "spearman_by_state",
               "fit_dbscan", "fit_optics", "optics_labels", "hdbscan_tree", "hdbscan_from_tree", "dbscan_star_cut",
               "k_distance", "fit", "agreement", "dbscan_grid", "hdbscan_grid"}
MODULES = {"dc", "qc", "oc", "ca", "ac", "pc"}

PROBE = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from streamlit.testing.v1 import AppTest
failed = []
for script in {SCRIPTS!r}:
    at = AppTest.from_file({str(ROOT)!r} + "/" + script, default_timeout=300)
    at.run()
    if at.exception:
        failed.append(script)
loaded = sorted({{m.split(".")[0] for m in sys.modules}} & {{"scipy", "sklearn", "hdbscan"}})
print("FAILED" if failed else "OK", failed, loaded)
"""


def test_pages_do_not_call_scipy_functions():
    for script in SCRIPTS:
        tree = ast.parse((ROOT / script).read_text())
        calls = {f"{n.value.id}.{n.attr}" for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in MODULES
                 and n.attr in NEEDS_SCIPY}
        imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in getattr(n, "names", []) if a.name.split(".")[0] in ("scipy", "sklearn", "hdbscan")}
        imports |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    and (n.module or "").split(".")[0] in ("scipy", "sklearn", "hdbscan")}
        assert not calls and not imports, (script, calls, imports)


def test_running_every_page_loads_neither_library():
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=600)
    last = out.stdout.strip().splitlines()[-1]
    assert last == "OK [] []", out.stdout[-2000:] + out.stderr[-2000:]
