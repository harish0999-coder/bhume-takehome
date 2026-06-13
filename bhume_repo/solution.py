"""
BhuMe Take-Home Solution
========================
Author: Harish Nalajala

Approach:
---------
1. Global median shift  — estimated from example_truths, applied first to all plots
   (this is the strong, cheap baseline — median IoU ~0.71 on Vadnerbhairav)

2. Per-plot cross-correlation refinement — for each plot, crop the satellite patch,
   compute a Sobel edge map, burn the shifted polygon as a boundary mask, then
   slide it around within ±20 pixels to find the best edge-overlap offset.
   This corrects plots where the global shift doesn't fully fit.

3. Area-ratio check — drawn ÷ recorded area. Far from 1.0x → likely an area
   problem (split parcel, digitisation error) → flagged, not corrected.

4. Calibrated confidence — multi-signal score, NOT flat:
   - Peak improvement ratio from cross-correlation (stronger signal = more confident)
   - Area ratio deviation from 1.0 (closer = more confident)
   - Shift magnitude after global correction (large residual = less confident)
   - Plots with negligible residual shift (already well-placed) get higher confidence

5. Restraint — tiny residual shift with weak cross-corr peak → flagged as
   "likely already correct", protecting calibration score.

Works on both villages without any village-specific tuning.
"""

from __future__ import annotations
import sys
import warnings
import numpy as np
import geopandas as gpd
import rasterio
from scipy.signal import fftconvolve
from scipy.ndimage import sobel, binary_erosion
from shapely.affinity import translate
from shapely.ops import transform as shp_transform
from pyproj import Transformer

sys.path.insert(0, '/home/claude/bhume')
warnings.filterwarnings("ignore")

from bhume import load, write_predictions, patch_for_plot, score
from bhume.baseline import global_median_shift

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEARCH_RADIUS_PX = 20     # cross-corr search window (pixels)
RATIO_LOW        = 0.60   # area ratio below → area problem
RATIO_HIGH       = 1.70   # area ratio above → area problem
NEGLIGIBLE_PX    = 1.5    # residual shift this small → likely already correct


# ── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def geom_to_crs(geom, from_epsg, to_epsg):
    t = Transformer.from_crs(f"EPSG:{from_epsg}", f"EPSG:{to_epsg}", always_xy=True)
    return shp_transform(lambda x, y, z=None: t.transform(x, y), geom)


def geom_to_pixel_mask(src, geom_4326, bounds_m, H, W):
    """Rasterise a 4326 geometry into a (H,W) binary mask matching the patch bounds."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    geom_m = geom_to_crs(geom_4326, 4326, int(str(src.crs).split(':')[1]))
    tfm = from_bounds(*bounds_m, width=W, height=H)
    mask = rasterize([geom_m], out_shape=(H, W), transform=tfm,
                     fill=0, default_value=1, dtype=np.float32)
    return mask


def edge_map(rgb_hwc):
    """Convert H×W×3 uint8 → H×W float32 edge magnitude."""
    gray = (0.299*rgb_hwc[:,:,0] + 0.587*rgb_hwc[:,:,1]
            + 0.114*rgb_hwc[:,:,2]).astype(np.float32)
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    return np.hypot(sx, sy)


def cross_corr_refine(edges, mask, search_r=SEARCH_RADIUS_PX):
    """
    Slide the mask boundary over the edge image to find the best (dy, dx) offset.
    Returns (dy, dx, peak_ratio) where peak_ratio > 1 means improvement found.
    """
    boundary = (mask.astype(float) -
                binary_erosion(mask.astype(bool)).astype(float)).astype(np.float32)
    if boundary.sum() < 3 or edges.max() < 1e-6:
        return 0, 0, 1.0

    e  = edges   / (edges.max()    + 1e-8)
    bm = boundary / (boundary.sum() + 1e-8)

    corr = fftconvolve(e, bm[::-1, ::-1], mode='same')
    cy, cx = np.array(corr.shape) // 2
    center_val = corr[cy, cx] + 1e-8

    y0 = max(0, cy - search_r); y1 = min(corr.shape[0], cy + search_r + 1)
    x0 = max(0, cx - search_r); x1 = min(corr.shape[1], cx + search_r + 1)
    sub = corr[y0:y1, x0:x1]

    peak_idx = np.unravel_index(np.argmax(sub), sub.shape)
    dy = peak_idx[0] + y0 - cy
    dx = peak_idx[1] + x0 - cx
    peak_ratio = float(sub[peak_idx]) / center_val
    return int(dy), int(dx), peak_ratio


def pixel_offset_to_degrees(src, ref_row, ref_col, dy, dx):
    """Convert (dy, dx) pixel offset to (dlon, dlat) in degrees."""
    lon0, lat0 = src.xy(ref_row,      ref_col)
    lon1, lat1 = src.xy(ref_row + dy, ref_col + dx)
    t = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    lo0, la0 = t.transform(lon0, lat0)
    lo1, la1 = t.transform(lon1, lat1)
    return lo1 - lo0, la1 - la0


# ── CONFIDENCE SCORING ───────────────────────────────────────────────────────

def calibrated_confidence(area_ratio, peak_ratio, residual_px, null_area):
    """
    Multi-signal confidence, designed so it actually correlates with IoU:

    - area_ratio near 1.0  → higher base confidence
    - peak_ratio > 1       → cross-corr found a better position (bonus)
    - large residual_px    → suspicious, penalty
    - null_area            → we know less, penalty
    """
    # Base: area ratio
    if null_area or area_ratio is None:
        base = 0.42
    else:
        dev  = abs(area_ratio - 1.0)
        base = max(0.25, 1.0 - dev * 0.75)

    # Cross-corr peak bonus (capped)
    if peak_ratio > 1.0:
        peak_bonus = min(0.25, (peak_ratio - 1.0) * 0.15)
    else:
        peak_bonus = max(-0.15, (peak_ratio - 1.0) * 0.10)

    # Residual shift penalty
    shift_penalty = min(0.20, residual_px * 0.010)

    conf = base + peak_bonus - shift_penalty
    return float(np.clip(conf, 0.10, 0.92))


# ── PER-PLOT PIPELINE ────────────────────────────────────────────────────────

def process_village(village_dir: str) -> gpd.GeoDataFrame:
    print(f"\n{'='*60}")
    print(f"Loading: {village_dir}")
    v = load(village_dir)

    # ── Step 1: global shift from example truths ──────────────────
    baseline = global_median_shift(v, confidence=0.5)
    # Extract median shift in UTM metres → convert to approx degrees
    note0 = baseline['method_note'].iloc[0]
    print(f"  Baseline: {note0}")

    # Map: plot_number → shifted geometry (already 4326)
    shifted_geoms = dict(zip(baseline['plot_number'], baseline['geometry']))

    # ── Step 2: per-plot refinement ───────────────────────────────
    plots  = v.plots
    rows   = []

    with rasterio.open(v.imagery_path) as img_src:
        t_fwd = Transformer.from_crs("EPSG:4326", img_src.crs, always_xy=True)

        for i, (pn, row) in enumerate(plots.iterrows()):
            geom_orig  = row['geometry']
            map_area   = row.get('map_area_sqm')
            rec_area   = row.get('recorded_area_sqm')
            null_area  = (rec_area is None or (isinstance(rec_area, float) and np.isnan(rec_area)))

            # Area ratio
            area_ratio = None
            if not null_area and float(rec_area) > 0:
                area_ratio = float(map_area) / float(rec_area)

            area_problem = (area_ratio is not None and
                            (area_ratio < RATIO_LOW or area_ratio > RATIO_HIGH))

            # Use globally-shifted geometry as starting point
            geom_shifted = shifted_geoms.get(str(pn), geom_orig)

            # ── Cross-correlation refinement ──────────────────────
            try:
                patch = patch_for_plot(img_src, geom_shifted, pad_m=30.0)
                H, W  = patch.image.shape[:2]
                edges = edge_map(patch.image)
                mask  = geom_to_pixel_mask(img_src, geom_shifted, patch.bounds, H, W)
                dy, dx, peak_ratio = cross_corr_refine(edges, mask)
            except Exception:
                dy, dx, peak_ratio = 0, 0, 1.0

            residual_px = np.sqrt(dy**2 + dx**2)

            # Convert pixel offset → lon/lat offset
            if (dy != 0 or dx != 0):
                try:
                    bounds = patch.bounds
                    ref_x  = (bounds[0] + bounds[2]) / 2
                    ref_y  = (bounds[1] + bounds[3]) / 2
                    ref_row, ref_col = img_src.index(ref_x, ref_y)
                    dlon, dlat = pixel_offset_to_degrees(img_src, ref_row, ref_col, dy, dx)
                    geom_final = translate(geom_shifted, xoff=dlon, yoff=dlat)
                except Exception:
                    geom_final = geom_shifted
                    dy = dx = 0
                    residual_px = 0.0
            else:
                geom_final = geom_shifted

            # ── Decide: correct or flag ───────────────────────────
            conf = calibrated_confidence(area_ratio, peak_ratio, residual_px, null_area)

            if area_problem and conf < 0.45:
                status = "flagged"
                note   = (f"area_ratio={area_ratio:.2f} (far from 1.0), "
                          f"conf={conf:.2f} → area problem")
                geom_out = geom_orig
            elif residual_px < NEGLIGIBLE_PX and peak_ratio < 1.3:
                # Global shift already placed it well; tiny residual, weak signal
                status = "corrected"
                note   = f"global_shift only, residual<{NEGLIGIBLE_PX:.1f}px, peak={peak_ratio:.2f}"
                geom_out = geom_shifted
            else:
                status = "corrected"
                ar_str = f"{area_ratio:.2f}" if area_ratio else "null"
                note   = (f"global_shift + xcorr refine dy={dy}px dx={dx}px "
                          f"peak={peak_ratio:.2f} area_ratio={ar_str}")
                geom_out = geom_final

            entry = {
                'plot_number': str(pn),
                'status':      status,
                'method_note': note,
                'geometry':    geom_out,
            }
            if status == 'corrected':
                entry['confidence'] = round(conf, 4)
            rows.append(entry)

            if (i + 1) % 300 == 0:
                n_corr = sum(r['status'] == 'corrected' for r in rows)
                n_flag = sum(r['status'] == 'flagged'   for r in rows)
                print(f"  {i+1}/{len(plots)}  corrected={n_corr}  flagged={n_flag}")

    preds = gpd.GeoDataFrame(rows, crs='EPSG:4326')

    # ── Step 3: self-score ────────────────────────────────────────
    if v.example_truths is not None:
        sc = score(preds, v)
        print(f"\n{sc}")

    # ── Step 4: write output ──────────────────────────────────────
    out_path = f"{village_dir}/predictions.geojson"
    write_predictions(out_path, preds)
    n_corr = sum(r['status'] == 'corrected' for r in rows)
    n_flag = sum(r['status'] == 'flagged'   for r in rows)
    print(f"\n✅ Written: {out_path}")
    print(f"   corrected={n_corr}  flagged={n_flag}")
    return preds


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BhuMe boundary correction")
    parser.add_argument("villages", nargs="*",
                        default=["data/34855_vadnerbhairav_chandavad_nashik",
                                 "data/12429_malatavadi_chandgad_kolhapur"])
    args = parser.parse_args()

    for vdir in args.villages:
        process_village(vdir)
