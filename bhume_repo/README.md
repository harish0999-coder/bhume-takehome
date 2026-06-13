# BhuMe Take-Home — Harish Nalajala

## Approach

### Problem framing
Official land plot boundaries drift from real fields due to imperfect georeferencing of
old hand-drawn maps. Two kinds of wrong: **placement** (fixable by shifting) and **area**
(geometry disagrees with record — moving won't help).

### Algorithm (solution.py)

1. **Area ratio check** — `map_area ÷ recorded_area`. Far from 1.0× → likely area
   problem → flag. Near 1.0× → placement candidate → attempt correction.

2. **Global median shift** — estimates one coherent dx/dy offset from the 6 example
   truths using centroid differences, applies it to all plots. This alone gets median
   IoU from 0.61 → 0.71.

3. **Per-plot cross-correlation refinement** — for each plot:
   - Crop satellite patch around the globally-shifted boundary (±30m padding)
   - Compute Sobel edge map from RGB imagery
   - Burn polygon boundary as a binary mask
   - Cross-correlate edge map vs mask boundary within ±20px search window
   - Apply the best (dy, dx) offset if it improves edge alignment

4. **Calibrated confidence** — multi-signal, NOT flat:
   - Area ratio deviation from 1.0 (closer = more confident)
   - Cross-correlation peak improvement ratio (stronger signal = more confident)
   - Residual shift magnitude (large = less confident)
   Final range: 0.10–0.92 (mean ~0.75)

5. **Restraint** — area problems with low confidence are flagged. Tiny residuals
   with weak cross-corr signal are left as global-shift-only (protecting calibration).

### Results (self-scored on example truths)

| Village | Median IoU | vs Official | Improved | AUC |
|---|---|---|---|---|
| Vadnerbhairav | 0.749 | +0.137 | 66.7% | 0.688 |
| Malatavadi | 0.724 | +0.214 | 66.7% | 1.000 |

### Running

```bash
pip install geopandas rasterio shapely scipy numpy pyproj
python solution.py
# or single village:
python solution.py data/34855_vadnerbhairav_chandavad_nashik
```

Data files (not in repo — download from hiring.bhume.in/start):
- `data/<village>/input.geojson`
- `data/<village>/imagery.tif`
- `data/<village>/boundaries.tif`
- `data/<village>/example_truths.geojson`
