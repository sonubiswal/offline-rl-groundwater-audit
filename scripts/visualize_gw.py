"""
visualize_gw.py

Plots CGWB groundwater depth points on a map of Maharashtra.

Usage:
    # For a specific month:
    python scripts/visualize_gw.py --month 2015-01-01

    # For average depth across all months (per well):
    python scripts/visualize_gw.py
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_groundwater(month: str = None, output: str = "groundwater_map.png"):
    """Plot groundwater depth wells for a specific month or averaged over all months."""
    
    # --- 1. Load the well data ---
    wells_path = Path("data/interim/cgwb_wells.csv")
    if not wells_path.exists():
        print(f"Error: Well file not found at {wells_path}")
        return

    df = pd.read_csv(wells_path)
    print(f"Loaded {len(df)} total readings.")

    # --- 2. Filter by month if provided ---
    if month:
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.strftime('%Y-%m-%d') == month]
        if df.empty:
            print(f"No wells found for month {month}. Showing average instead.")
            return plot_groundwater(month=None, output=output)
        print(f"Filtered to {len(df)} readings for {month}.")
        # Average if there are multiple readings per well in that month (unlikely, but safe)
        df = df.groupby(['well_id', 'lat', 'lon'], as_index=False).agg({'depth_m': 'mean'})
    else:
        # Average depth per unique well over all months
        df = df.groupby(['well_id', 'lat', 'lon'], as_index=False).agg({'depth_m': 'mean'})
        print(f"Averaged to {len(df)} unique wells.")

    # --- 3. Plot the wells ---
    # Maharashtra bounding box from your config
    bbox = [72.6, 15.6, 80.9, 22.0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sc = ax.scatter(
        df['lon'],
        df['lat'],
        c=df['depth_m'],
        cmap='RdYlGn_r',    # Red (deep) → Yellow → Green (shallow)
        s=30,
        edgecolor='k',
        linewidth=0.5,
        alpha=0.8
    )
    
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    title = f'Groundwater Depth – {month}' if month else 'Groundwater Depth (Average per Well)'
    ax.set_title(title)
    
    # Colorbar with reversed axis (deep at bottom)
    cbar = plt.colorbar(sc, label='Depth to Water (m)')
    cbar.ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Plot saved as {output}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise groundwater depth wells.")
    parser.add_argument("--month", default=None, help="Month in YYYY-MM-DD format (e.g., 2015-01-01). Omit to plot average.")
    parser.add_argument("--output", default="groundwater_map.png", help="Output PNG file.")
    args = parser.parse_args()
    
    plot_groundwater(args.month, args.output)