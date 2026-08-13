"""
causal_runner.py  --  configure and batch-launch causal_29_12.py

Edit the CONFIG section below, then run:
    python causal_runner.py

A timestamped log file is saved to the logs/ folder after every run.
"""

import datetime
import os
import subprocess
import sys
import time

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Folder that contains the window CSV files
WINDOWS_DIR = r'C:\Eminds pros\E-minds projects\fuhrlander\causal-data-engine\pipeline\windows'

# Filename prefix before the 3-digit window index
PATTERN = 'wt80_seg0_win'

# Window range (inclusive on both ends)
WIN_FROM = 11
WIN_TO   = 27

LMIN  = 3      # minimum aggregation factor
LMAX  = 15     # maximum aggregation factor
LINC  = 1      # step between L values

ALPHA    = 0.1   # SET_LAMBDA — ICA sparsity regularisation scale
ICA_REGU = 0.05  # ICA regularisation strength

# ─────────────────────────────────────────────────────────────────────────────

HERE   = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(HERE, "causal_29_12_runner.py")
logs_dir = os.path.join(HERE, "logs")
os.makedirs(logs_dir, exist_ok=True)

run_start_dt = datetime.datetime.now()
log_filename  = f"run_{run_start_dt.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
log_path      = os.path.join(logs_dir, log_filename)

# ── helpers ───────────────────────────────────────────────────────────────────

_log_lines = []

def log(line=""):
    """Print to console and buffer for the log file."""
    print(line)
    _log_lines.append(line)

def write_log():
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")
    print(f"\nLog saved to: {log_path}")

# ── header ────────────────────────────────────────────────────────────────────

log(f"\n{'='*65}")
log(f"  causal_runner.py")
log(f"  started       : {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  windows dir   : {WINDOWS_DIR}")
log(f"  pattern       : {PATTERN}  [{WIN_FROM:03d} -> {WIN_TO:03d}]  ({WIN_TO - WIN_FROM + 1} windows)")
log(f"  L range       : {LMIN} to {LMAX} step {LINC}  ({len(range(LMIN, LMAX+1, LINC))} iterations per window)")
log(f"  alpha         : {ALPHA}")
log(f"  ica_regu      : {ICA_REGU}")
log(f"{'='*65}")

# ── main loop ─────────────────────────────────────────────────────────────────

batch_start = time.perf_counter()
results = []   # [(win_id, win_name, status, elapsed)]

for i in range(WIN_FROM, WIN_TO + 1):
    win_name = f"{PATTERN}{i:03d}"
    filepath = os.path.join(WINDOWS_DIR, win_name + ".csv")

    log(f"\n{'='*65}")
    log(f"  Window {i:03d} / {WIN_TO:03d}  --  {win_name}.csv")
    log(f"  started : {datetime.datetime.now().strftime('%H:%M:%S')}")
    log(f"{'='*65}")

    if not os.path.exists(filepath):
        log(f"  [SKIP] File not found: {filepath}")
        results.append((i, win_name, "MISSING", 0.0))
        continue

    win_start = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable, script,
            "--file",     filepath,
            "--lmin",     str(LMIN),
            "--lmax",     str(LMAX),
            "--linc",     str(LINC),
            "--alpha",    str(ALPHA),
            "--ica-regu", str(ICA_REGU),
        ],
        cwd=HERE,
    )
    elapsed = time.perf_counter() - win_start
    status  = "OK" if proc.returncode == 0 else f"ERROR({proc.returncode})"
    results.append((i, win_name, status, elapsed))
    log(f"\n  [{status}] Window {i:03d} finished")
    log(f"  finished : {datetime.datetime.now().strftime('%H:%M:%S')}")
    log(f"  time     : {elapsed:.1f}s  ({elapsed/60:.2f} min)")

# ── batch summary ─────────────────────────────────────────────────────────────

batch_total   = time.perf_counter() - batch_start
run_end_dt    = datetime.datetime.now()

log(f"\n{'='*65}")
log(f"  BATCH SUMMARY")
log(f"{'='*65}")
log(f"  Run started  : {run_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Run finished : {run_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"")
log(f"  Config")
log(f"  ------")
log(f"  Windows dir  : {WINDOWS_DIR}")
log(f"  Pattern      : {PATTERN}  [{WIN_FROM:03d} -> {WIN_TO:03d}]")
log(f"  L range      : {LMIN} to {LMAX} step {LINC}")
log(f"  alpha        : {ALPHA}    ica_regu: {ICA_REGU}")
log(f"")
log(f"  Per-window timings")
log(f"  ------------------")
log(f"  {'Win':>4}  {'Name':<22}  {'Status':>12}  {'seconds':>10}  {'minutes':>8}")
log(f"  {'-'*62}")
for win_id, win_name, status, elapsed in results:
    log(f"  {win_id:>4}  {win_name:<22}  {status:>12}  {elapsed:>10.1f}  {elapsed/60:>8.2f}")
log(f"  {'-'*62}")
log(f"  {'':>4}  {'TOTAL':<22}  {'':>12}  {batch_total:>10.1f}  {batch_total/60:>8.2f}")
log(f"{'='*65}")

ok  = sum(1 for *_, s, _ in results if s == "OK")
err = sum(1 for *_, s, _ in results if s not in ("OK", "MISSING"))
skp = sum(1 for *_, s, _ in results if s == "MISSING")
log(f"")
log(f"  {ok} succeeded  |  {err} failed  |  {skp} skipped (missing file)")
log(f"  Total wall time : {batch_total:.1f}s  ({batch_total/60:.2f} min)")
log(f"  Log file        : {log_path}")

write_log()
