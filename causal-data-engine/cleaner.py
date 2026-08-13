"""
cleaner.py  --  extract pattern-matching Excel files from causal analysis results

Walks SOURCE_DIR recursively, copies every .xlsx whose filename contains
FILE_PATTERN into OUTPUT_DIR (flat, no sub-folders).

Edit the three variables in the CONFIG block below, then run:
    python cleaner.py
"""

import os
import shutil

# ─── CONFIG ──────────────────────────────────────────────────────────────────

SOURCE_DIR  = r'C:\Eminds pros\E-minds projects\fuhrlander\causal-data-engine\WT_80_Causal_Analysis_Results'

# Only .xlsx files whose name contains this string are copied.
# Change this to target a different segment or pattern.
FILE_PATTERN = 'wt80_seg0'

OUTPUT_DIR  = r'C:\Eminds pros\E-minds projects\fuhrlander\causal-data-engine\extracted_matrices'

# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied = 0
    skipped = 0

    for root, _, files in os.walk(SOURCE_DIR):
        for fname in files:
            if not fname.endswith('.xlsx'):
                continue
            if FILE_PATTERN not in fname:
                continue
            if fname.startswith('scaled_'):
                continue

            src = os.path.join(root, fname)
            dst = os.path.join(OUTPUT_DIR, fname)

            # Avoid silent overwrites if two windows produce the same filename
            if os.path.exists(dst):
                base, ext = os.path.splitext(fname)
                # append the parent folder name to disambiguate
                parent = os.path.basename(root)
                dst = os.path.join(OUTPUT_DIR, f"{base}__{parent}{ext}")

            shutil.copy2(src, dst)
            print(f"  copied  {fname}")
            copied += 1

    print(f"\nDone — {copied} file(s) copied to:\n  {OUTPUT_DIR}")
    if skipped:
        print(f"  {skipped} skipped (no pattern match)")


if __name__ == '__main__':
    main()
