
from pathlib import Path
import gdown
import sqlite3

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "clustering_results.db"
FILE_ID = "1ekrWrLBKH_e1wM3jVQLnngysRB_a-bit"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Reuse a valid existing database if available.
if DB_PATH.exists():
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
            con.execute("PRAGMA schema_version").fetchone()
        print("Clustering database already exists.")
        raise SystemExit(0)
    except sqlite3.Error:
        DB_PATH.unlink(missing_ok=True)

print("Downloading clustering database from Google Drive...")

temp_path = DB_PATH.with_suffix(".download")
temp_path.unlink(missing_ok=True)

result = gdown.download(
    id=FILE_ID,
    output=str(temp_path),
    quiet=False,
)

if not result or not temp_path.exists():
    raise RuntimeError(
        "Download failed. Check Google Drive sharing permissions."
    )

try:
    with sqlite3.connect(f"file:{temp_path}?mode=ro", uri=True) as con:
        con.execute("PRAGMA schema_version").fetchone()
except sqlite3.Error as exc:
    temp_path.unlink(missing_ok=True)
    raise RuntimeError("Downloaded file is not a valid SQLite database.") from exc

temp_path.replace(DB_PATH)
print(f"Database ready: {DB_PATH}")
