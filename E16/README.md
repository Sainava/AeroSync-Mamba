1. Upload `e16-audio-representation.ipynb` to Kaggle; Settings → Accelerator **GPU**, Internet **ON** (the notebook clones the TII dataset itself, or uses an attached copy with `Dataset/0..4`).
2. Optional: attach `version 27/` as a Kaggle dataset so v27's inner-validation rule runs; otherwise the fixed-seed-2026 fallback runs (printed either way).
3. First run `SMOKE = True` (top cell: R0 + R5, fold 1, seed 42, 1 epoch; the first session also downloads ~4.4 GB and builds the parse cache, so allow ~20–40 min) to check the pipeline, then set `SMOKE = False`.
4. "Save Version" → "Save & Run All (Commit)": 9 representations × 5 day-out folds × 3 seeds, ~2–3.5 h, resumable (finished runs in `e16_results.csv` are skipped).
5. Results: `e16_summary.txt` (verdict from the pre-registered rule), `e16_table_*.csv`, `figures/`, all bundled in `/kaggle/working/e16_outputs.zip`.
