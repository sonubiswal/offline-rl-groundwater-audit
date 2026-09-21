"""
visualize_rainfall_gw.py

Plots a side‑by‑side comparison of:
  - Left: Aligned CHIRPS rainfall map (raster).
  - Right: CGWB groundwater depth points (scatter), averaged per unique well.

Usage:
    python scripts/visualize_rainfall_gw.py --month 2015-01-01
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from pathlib import Path

def plot_rainfall_gw(month: str = "2015-01-01", output: str = "rainfall_gw_comparison.png"):
    """Generate a side‑by‑side comparison plot."""

    # --- 1. Define file paths ---
    rain_path = Path(f"data/interim/chirps/{month}.tif")
    wells_path = Path("data/interim/cgwb_wells.csv")

    # --- 2. Load wells and average per well (unique ID) ---
    if not wells_path.exists():
        print(f"Error: Well file not found at {wells_path}")
        return

    df = pd.read_csv(wells_path)

    # If you want to filter by the specific month, uncomment the line below:
    # df = df[df['date'] == month]

    # Average depth per unique well (to avoid over‑plotting hundreds of readings)
    unique_wells = df.groupby(['well_id', 'lat', 'lon'], as_index=False).agg({
        'depth_m': 'mean'
    })

    print(f"Loaded {len(unique_wells)} unique wells (averaged depth).")

    # --- 3. Create the figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 4. Left: Rainfall map ---
    if rain_path.exists():
        with rasterio.open(rain_path) as src:
            show(src, ax=ax1, cmap='Blues')
            ax1.set_title(f'CHIRPS Rainfall – {month}')
            # Store bounds to align the right plot
            left, bottom, right, top = src.bounds
    else:
        ax1.set_title(f'Rainfall data not found for {month}')
        left, bottom, right, top = 72.6, 15.6, 80.9, 22.0  # fallback bbox

    # --- 5. Right: Groundwater depth (scatter) ---
    sc = ax2.scatter(
        unique_wells['lon'],
        unique_wells['lat'],
        c=unique_wells['depth_m'],
        cmap='RdYlGn_r',   # Red (deep) → Yellow → Green (shallow)
        s=20,              # point size
        alpha=0.7,         # transparency
        edgecolor='k',     # black outline
        linewidth=0.3
    )
    # Align the axis extent to match the rainfall map
    ax2.set_xlim(left, right)
    ax2.set_ylim(bottom, top)
    ax2.set_title('Groundwater Depth (avg per well)')
    cbar = plt.colorbar(sc, ax=ax2, label='Depth to Water (m)')
    cbar.ax.invert_yaxis()  # optional: deeper wells at bottom of colour bar

    # --- 6. Save and show ---
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Plot saved as {output}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise rainfall vs groundwater depth.")
    parser.add_argument("--month", default="2015-01-01", help="Month in YYYY-MM-DD format.")
    parser.add_argument("--output", default="rainfall_gw_comparison.png", help="Output PNG file.")
    args = parser.parse_args()

    plot_rainfall_gw(args.month, args.output)