"""Memory and CPU probe for the dashboard on a real Streamlit server.

Starts `streamlit run app.py` headless, then opens N websocket sessions at once; each session runs every page with
its default settings, like a visitor clicking through the whole dashboard. Prints the server's resident memory (RSS)
before and after, so a hosting plan can be checked against it. With N = 0 it runs one cold and one warm visitor in
turn and prints the CPU seconds each one cost.

Usage (repository root, venv active; needs `psutil` and `websockets`):
    python tools/memory_probe.py 3
    python tools/memory_probe.py 6 MALLOC_ARENA_MAX=2
    python tools/memory_probe.py 0
"""
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import websockets
from streamlit.proto import BackMsg_pb2, ForwardMsg_pb2

ROOT = Path(__file__).resolve().parent.parent
PORT = 8599
PAGES = ["", "Data_Quality", "Coordinate_Groups", "Lat_Lon_Distribution", "Geographic_Analysis",
         "Spatial_Clustering", "Brewery_Types", "Countries_States_Cities", "Prediction"]
URL = f"ws://localhost:{PORT}/_stcore/stream"


async def run_page(ws, page):
    """Ask the server to run one page and wait until it finishes. Returns the number of exceptions shown."""
    msg = BackMsg_pb2.BackMsg()
    msg.rerun_script.query_string = ""
    msg.rerun_script.page_name = page
    msg.rerun_script.widget_states.SetInParent()
    await ws.send(msg.SerializeToString())
    errors = 0
    while True:
        fm = ForwardMsg_pb2.ForwardMsg()
        fm.ParseFromString(await asyncio.wait_for(ws.recv(), timeout=300))
        kind = fm.WhichOneof("type")
        if kind == "delta" and fm.delta.WhichOneof("type") == "new_element":
            errors += fm.delta.new_element.WhichOneof("type") == "exception"
        elif kind == "page_not_found":
            raise RuntimeError(f"page not found: {page}")
        elif kind == "script_finished":
            return errors


async def visitor():
    async with websockets.connect(URL, subprotocols=["streamlit"], max_size=None) as ws:
        return sum([await run_page(ws, page) for page in PAGES])


async def main(proc, sessions):
    rss = lambda: proc.memory_info().rss / 2**20
    cpu = lambda: sum(proc.cpu_times()[:2])
    for _ in range(60):
        try:
            async with websockets.connect(URL, subprotocols=["streamlit"]):
                break
        except OSError:
            await asyncio.sleep(1)
    print(f"server idle: RSS {rss():.0f} MB")
    if sessions == 0:
        for kind in ["cold", "warm"]:
            c0, t0 = cpu(), time.time()
            await visitor()
            print(f"{kind} visitor: {cpu() - c0:.1f} CPU-s, {time.time() - t0:.1f} s, RSS {rss():.0f} MB")
        return
    t0 = time.time()
    errors = await asyncio.gather(*(visitor() for _ in range(sessions)))
    print(f"{sessions} visitor(s) x {len(PAGES)} pages in {time.time() - t0:.0f} s, exceptions {sum(errors)}, "
          f"RSS {rss():.0f} MB")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    env = dict(os.environ, **dict(kv.split("=", 1) for kv in sys.argv[2:]))
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                               "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        asyncio.run(main(psutil.Process(server.pid), n))
    finally:
        server.terminate()
        server.wait()
