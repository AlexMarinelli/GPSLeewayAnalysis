"""
Complete Sailing Data Analysis Pipeline
========================================
This script combines data preprocessing and leeway analysis into a single workflow.

WORKFLOW:
1. Load raw CSV data from boat instruments
2. Filter by time window (AEDT timezone)
3. Apply moving averages and filters
4. Extract upwind sailing periods
5. Perform leeway analysis with configurable lag
6. Generate comprehensive visualizations

Author: Sailing Performance Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import pytz
import os

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Data Files
RAW_DATA_FILE_1 = 'files/2025-03-09 Yandoo.csv'
RAW_DATA_FILE_2 = 'files/2025-03-09 Lazarus Capital Partners.csv'  # Set to path for second dataset or None

# Time Window (AEDT)
DATE_STR = "2025-03-09"
START_TIME_AEDT = "14:53"
END_TIME_AEDT = "16:15"

# Course Parameters
COURSE_AXIS = 65
COURSE_AXIS_ADJ = 0

# Speed Filters
UPWIND_MIN_SPEED = 6
UPWIND_MAX_SPEED = 13

# Analysis Parameters
MA_WINDOW = 4           # Moving average window
LAG_SECONDS = 1         # Lag for leeway analysis

# Output Directory
OUTPUT_DIR = 'outputs'

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def adjust_angle(angle, adjustment):
    """Adjust angle and wrap to 0-360 range"""
    adjusted = angle + adjustment
    if adjusted > 360:
        adjusted -= 360
    elif adjusted < 0:
        adjusted += 360
    return adjusted

def check_upwind(cog, course_axis):
    """Check if sailing upwind based on COG and course axis"""
    diff = abs((cog - course_axis + 180) % 360 - 180)
    return diff <= 90

def filter_by_time_window(df, start_time_str, end_time_str, date_str, timezone_str='Australia/Sydney'):
    """Filter dataframe by time window in local timezone"""
    df['timestamp_utc'] = pd.to_datetime(df['ISODateTimeUTC'])
    local_tz = pytz.timezone(timezone_str)
    df['timestamp_local'] = df['timestamp_utc'].dt.tz_convert(local_tz)
    
    start_dt = local_tz.localize(
        datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %H:%M")
    )
    end_dt = local_tz.localize(
        datetime.strptime(f"{date_str} {end_time_str}", "%Y-%m-%d %H:%M")
    )
    
    mask = (df['timestamp_local'] >= start_dt) & (df['timestamp_local'] <= end_dt)
    return df[mask].copy()

def preprocess_sailing_data(input_file, start_time, end_time, date_str,
                            course_axis, course_axis_adj,
                            upwind_min_speed, upwind_max_speed, ma_window):
    """
    Preprocess raw sailing data
    Returns processed dataframe with filtered upwind data
    """
    print(f"\nPreprocessing: {os.path.basename(input_file)}")
    
    # Load and filter
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} records")
    
    df_filtered = filter_by_time_window(df, start_time, end_time, date_str)
    print(f"  Filtered to {len(df_filtered)} records in time window")
    
    if len(df_filtered) == 0:
        print("  ERROR: No data in time window!")
        return None
    
    # Extract columns
    df_proc = pd.DataFrame()
    df_proc['timestamp'] = pd.to_datetime(df_filtered['ISODateTimeUTC'])
    df_proc['latitude'] = df_filtered['Lat']
    df_proc['longitude'] = df_filtered['Lon']
    df_proc['sog_raw'] = df_filtered['SOG']
    df_proc['cog_raw'] = df_filtered['COG']
    df_proc['hdg_raw'] = df_filtered['Heading']
    
    # Apply adjustments
    df_proc['cog_adj'] = df_proc['cog_raw'].apply(lambda x: adjust_angle(x, course_axis_adj))
    df_proc['hdg_adj'] = df_proc['hdg_raw'].apply(lambda x: adjust_angle(x, course_axis_adj))
    
    # Moving averages
    df_proc['SOG_MA'] = df_proc['sog_raw'].rolling(window=ma_window, min_periods=1).mean()
    df_proc['COG_MA'] = df_proc['cog_adj'].rolling(window=ma_window, min_periods=1).mean()
    df_proc['HDG_MA'] = df_proc['hdg_adj'].rolling(window=ma_window, min_periods=1).mean()
    
    # Determine upwind and apply speed filter
    df_proc['is_upwind'] = df_proc['COG_MA'].apply(lambda x: check_upwind(x, course_axis))
    df_proc['upwind_speed_ok'] = ((df_proc['SOG_MA'] > upwind_min_speed) & 
                                   (df_proc['SOG_MA'] < upwind_max_speed))
    
    # Final output
    df_proc['SOG'] = 0.0
    df_proc['COG'] = 0.0
    df_proc['HDG'] = 0.0
    
    mask = df_proc['is_upwind'] & df_proc['upwind_speed_ok']
    df_proc.loc[mask, 'SOG'] = df_proc.loc[mask, 'SOG_MA']
    df_proc.loc[mask, 'COG'] = df_proc.loc[mask, 'COG_MA']
    df_proc.loc[mask, 'HDG'] = df_proc.loc[mask, 'HDG_MA']
    
    # Round
    output_df = df_proc[['timestamp', 'latitude', 'longitude', 'SOG', 'COG', 'HDG']].copy()
    output_df['latitude'] = output_df['latitude'].round(6)
    output_df['longitude'] = output_df['longitude'].round(6)
    output_df['SOG'] = output_df['SOG'].round(2)
    output_df['COG'] = output_df['COG'].round(2)
    output_df['HDG'] = output_df['HDG'].round(2)
    
    valid_count = (output_df['SOG'] > 0).sum()
    print(f"  Valid upwind records: {valid_count} ({100*valid_count/len(output_df):.1f}%)")
    
    return output_df

# ============================================================================
# LEEWAY ANALYSIS FUNCTIONS
# ============================================================================

def process_dataset_for_analysis(df, dataset_name, lag_seconds):
    """Process preprocessed data for leeway analysis"""
    
    # Set timestamp as index and resample to 1-second intervals
    df_indexed = df.set_index('timestamp')
    df_1sec = df_indexed.resample('1S').mean()
    
    df_1sec['SOG'] = df_1sec['SOG'].round(2)
    df_1sec['COG'] = df_1sec['COG'].round(2)
    df_1sec['HDG'] = df_1sec['HDG'].round(2)
    
    if 'latitude' in df_1sec.columns:
        df_1sec['latitude'] = df_1sec['latitude'].round(6)
    if 'longitude' in df_1sec.columns:
        df_1sec['longitude'] = df_1sec['longitude'].round(6)
    
    # Calculate leeway and derivatives
    df_1sec['Leeway'] = (df_1sec['HDG'] - df_1sec['COG']).abs().round(2)
    df_1sec['SOG_Change'] = df_1sec['SOG'].diff().round(3)
    df_1sec['Leeway_Lag'] = df_1sec['Leeway'].shift(lag_seconds).round(2)
    df_1sec = df_1sec.reset_index()
    
    # Identify upwind periods
    df_1sec['is_upwind'] = (df_1sec['SOG'] > 0) & (df_1sec['COG'] > 0) & (df_1sec['HDG'] > 0)
    df_1sec['period_change'] = df_1sec['is_upwind'].astype(int).diff().fillna(0)
    df_1sec['period_id'] = (df_1sec['period_change'] != 0).cumsum()
    
    # Get top 3 longest periods
    upwind_periods = df_1sec[df_1sec['is_upwind']].groupby('period_id').agg({
        'timestamp': ['first', 'last', 'count']
    })
    upwind_periods.columns = ['_'.join(col).strip() for col in upwind_periods.columns.values]
    upwind_periods = upwind_periods.reset_index()
    upwind_periods['duration_seconds'] = (
        upwind_periods['timestamp_last'] - upwind_periods['timestamp_first']
    ).dt.total_seconds()
    
    top_3_periods = upwind_periods.nlargest(3, 'duration_seconds')
    top_3_period_ids = top_3_periods['period_id'].tolist()
    
    # Filter to top 3 periods
    top3_data = df_1sec[df_1sec['period_id'].isin(top_3_period_ids) & df_1sec['is_upwind']].copy()
    top3_data = top3_data.dropna(subset=['SOG_Change', 'Leeway_Lag'])
    
    # Create buckets
    def categorize_leeway(leeway):
        if leeway < 2:
            return '0-2°'
        elif leeway < 4:
            return '2-4°'
        elif leeway < 6:
            return '4-6°'
        elif leeway < 8:
            return '6-8°'
        elif leeway < 10:
            return '8-10°'
        elif leeway < 12:
            return '10-12°'
        else:
            return '12+°'
    
    top3_data['Leeway_Bucket'] = top3_data['Leeway_Lag'].apply(categorize_leeway)
    
    # Calculate correlations
    corr_accel = stats.pearsonr(top3_data['Leeway_Lag'], top3_data['SOG_Change'])
    corr_sog = stats.pearsonr(top3_data['Leeway_Lag'], top3_data['SOG'])
    
    return top3_data, corr_accel, corr_sog, top_3_period_ids, dataset_name

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_analysis_plots(datasets_data, lag_seconds, output_dir):
    """Create comprehensive analysis plots"""
    
    bucket_names = ['0-2°', '2-4°', '4-6°', '6-8°', '8-10°', '10-12°', '12+°']
    bucket_colors = ['darkgreen', 'limegreen', 'yellowgreen', 'yellow', 'orange', 'orangered', 'red']
    
    num_datasets = len(datasets_data)
    
    # Main comparison figure
    fig = plt.figure(figsize=(24, 6 * num_datasets))
    fig.suptitle(f'Leeway Analysis ({lag_seconds} sec lag)', fontsize=20, fontweight='bold')
    
    gs = fig.add_gridspec(num_datasets, 3, hspace=0.3, wspace=0.3)
    
    for idx, (data, corr_accel, corr_sog, period_ids, name) in enumerate(datasets_data):
        
        # Get bucketed data
        buckets = []
        for bucket_name in bucket_names:
            bucket_data = data[data['Leeway_Bucket'] == bucket_name]
            buckets.append(bucket_data)
        
        # Plot 1: Mean acceleration by bucket
        ax1 = fig.add_subplot(gs[idx, 0])
        means_accel = [bucket['SOG_Change'].mean() if len(bucket) > 0 else 0 for bucket in buckets]
        bars = ax1.bar(bucket_names, means_accel, color=bucket_colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax1.set_ylabel('Mean Acceleration (kt/s)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Leeway Bucket', fontsize=12, fontweight='bold')
        ax1.set_title(f'{name}: Mean Acceleration by Leeway', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        for bar, mean in zip(bars, means_accel):
            if mean != 0:
                height = bar.get_height()
                y_pos = height + 0.005 if height > 0 else height - 0.01
                ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{mean:.4f}', ha='center', va='bottom' if height > 0 else 'top', 
                        fontsize=9, fontweight='bold')
        
        # Plot 2: Mean SOG by bucket
        ax2 = fig.add_subplot(gs[idx, 1])
        means_sog = [bucket['SOG'].mean() if len(bucket) > 0 else 0 for bucket in buckets]
        bars = ax2.bar(bucket_names, means_sog, color=bucket_colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Mean SOG (knots)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Leeway Bucket', fontsize=12, fontweight='bold')
        ax2.set_title(f'{name}: Mean SOG by Leeway', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        for bar, mean in zip(bars, means_sog):
            if mean != 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.2f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        # Plot 3: SOG vs Leeway scatter
        ax3 = fig.add_subplot(gs[idx, 2])
        colors_period = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for pidx, period_id in enumerate(period_ids):
            period_data = data[data['period_id'] == period_id]
            ax3.scatter(period_data['Leeway_Lag'], period_data['SOG'],
                      color=colors_period[pidx], alpha=0.4, s=20, label=f'Period {pidx+1}',
                      edgecolors='black', linewidth=0.3)
        
        z = np.polyfit(data['Leeway_Lag'], data['SOG'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data['Leeway_Lag'].min(), data['Leeway_Lag'].max(), 100)
        ax3.plot(x_trend, p(x_trend), "red", linewidth=2, alpha=0.8, label=f'Trend (slope={z[0]:.3f})')
        ax3.set_xlabel(f'Leeway {lag_seconds}s Ago (degrees)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('SOG (knots)', fontsize=12, fontweight='bold')
        ax3.set_title(f'{name}: SOG vs Leeway\nr = {corr_sog[0]:.4f}, p = {corr_sog[1]:.6f}', 
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/leeway_analysis_lag{lag_seconds}s.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: leeway_analysis_lag{lag_seconds}s.png")
    
    # SOG Distribution figure
    fig2, axes2 = plt.subplots(1, min(3, num_datasets + 1), figsize=(18, 5))
    if not isinstance(axes2, np.ndarray):
        axes2 = [axes2]
    else:
        axes2 = axes2.flatten()
    fig2.suptitle(f'SOG Distribution Analysis ({lag_seconds} sec lag)', fontsize=18, fontweight='bold')
    
    for idx, (data, _, _, _, name) in enumerate(datasets_data):
        ax = axes2[idx]
        color = ['blue', 'red', 'green'][idx % 3]
        ax.hist(data['SOG'].dropna(), bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('SOG (knots)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Add median and quartiles to title
        median_sog = data['SOG'].median()
        q1_sog = data['SOG'].quantile(0.25)
        q3_sog = data['SOG'].quantile(0.75)
        
        ax.set_title(f'{name}: SOG Distribution\n'
                     f'Median={median_sog:.2f} kt, Q1={q1_sog:.2f} kt, Q3={q3_sog:.2f} kt', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add vertical lines for median and quartiles
        ax.axvline(median_sog, color='red', linestyle='--', linewidth=2, label=f'Median: {median_sog:.2f}')
        ax.axvline(q1_sog, color='orange', linestyle=':', linewidth=1.5, label=f'Q1: {q1_sog:.2f}')
        ax.axvline(q3_sog, color='orange', linestyle=':', linewidth=1.5, label=f'Q3: {q3_sog:.2f}')
        ax.legend(fontsize=9)
    
    if num_datasets > 1 and len(axes2) > num_datasets:
        ax = axes2[num_datasets]
        for idx, (data, _, _, _, name) in enumerate(datasets_data):
            color = ['blue', 'red', 'green'][idx % 3]
            ax.hist(data['SOG'].dropna(), bins=30, color=color, alpha=0.5, edgecolor='black', label=name)
        ax.set_xlabel('SOG (knots)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('SOG Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sog_histogram_lag{lag_seconds}s.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: sog_histogram_lag{lag_seconds}s.png")
    
    # Leeway Distribution figure
    fig3, axes3 = plt.subplots(1, min(3, num_datasets + 1), figsize=(18, 5))
    if not isinstance(axes3, np.ndarray):
        axes3 = [axes3]
    else:
        axes3 = axes3.flatten()
    fig3.suptitle(f'Leeway Distribution Analysis ({lag_seconds} sec lag)', fontsize=18, fontweight='bold')
    
    for idx, (data, _, _, _, name) in enumerate(datasets_data):
        ax = axes3[idx]
        color = ['blue', 'red', 'green'][idx % 3]
        ax.hist(data['Leeway'].dropna(), bins=40, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Leeway (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Add median and quartiles to title
        median_leeway = data['Leeway'].median()
        q1_leeway = data['Leeway'].quantile(0.25)
        q3_leeway = data['Leeway'].quantile(0.75)
        
        ax.set_title(f'{name}: Leeway Distribution\n'
                     f'Median={median_leeway:.2f}°, Q1={q1_leeway:.2f}°, Q3={q3_leeway:.2f}°', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add vertical lines for median and quartiles
        ax.axvline(median_leeway, color='red', linestyle='--', linewidth=2, label=f'Median: {median_leeway:.2f}°')
        ax.axvline(q1_leeway, color='orange', linestyle=':', linewidth=1.5, label=f'Q1: {q1_leeway:.2f}°')
        ax.axvline(q3_leeway, color='orange', linestyle=':', linewidth=1.5, label=f'Q3: {q3_leeway:.2f}°')
        ax.legend(fontsize=9)
    
    if num_datasets > 1 and len(axes3) > num_datasets:
        ax = axes3[num_datasets]
        for idx, (data, _, _, _, name) in enumerate(datasets_data):
            color = ['blue', 'red', 'green'][idx % 3]
            ax.hist(data['Leeway'].dropna(), bins=40, color=color, alpha=0.5, edgecolor='black', label=name)
        ax.set_xlabel('Leeway (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Leeway Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/leeway_histogram_lag{lag_seconds}s.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: leeway_histogram_lag{lag_seconds}s.png")
    
    plt.close()

def create_gps_track_map(datasets_data, output_dir, processed_datasets=None, dataset_names=None):
    """Create GPS track map with upwind sections highlighted"""
    
    num_datasets = len(datasets_data)
    
    # Create figure with one map per dataset
    fig, axes = plt.subplots(1, num_datasets, figsize=(10 * num_datasets, 8))
    if num_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('GPS Track Map - All Upwind Data (Top 3 Periods Highlighted)', fontsize=18, fontweight='bold')
    
    for idx, (analyzed_data, _, _, period_ids, name) in enumerate(datasets_data):
        ax = axes[idx]
        
        # Get the full processed dataset to show ALL upwind sections
        if processed_datasets is not None and idx < len(processed_datasets):
            full_data = processed_datasets[idx].copy()
            full_data = full_data.sort_values('timestamp')
            
            # Separate upwind (SOG > 0) from non-upwind (SOG = 0)
            upwind_mask = (full_data['SOG'] > 0)
            all_upwind_data = full_data[upwind_mask].copy()
            
            # Identify continuous sections (gaps > 2 seconds = new section)
            if len(all_upwind_data) > 0:
                all_upwind_data['time_diff'] = all_upwind_data['timestamp'].diff().dt.total_seconds()
                all_upwind_data['section_id'] = (all_upwind_data['time_diff'] > 2).cumsum()
                
                # Plot each continuous section separately (don't connect gaps)
                for section_id in all_upwind_data['section_id'].unique():
                    section_data = all_upwind_data[all_upwind_data['section_id'] == section_id]
                    if len(section_data) > 1:  # Only plot if more than 1 point
                        ax.plot(section_data['longitude'], section_data['latitude'], 
                               color='darkblue', linewidth=2, alpha=0.6, zorder=1)
                        ax.scatter(section_data['longitude'], section_data['latitude'], 
                                  color='darkblue', s=3, alpha=0.3, zorder=1)
                
                # Add legend entry for all upwind data
                ax.plot([], [], color='darkblue', linewidth=2, alpha=0.6, label='All Upwind Data')
        
        # Get all points from the analyzed upwind data
        all_lats = analyzed_data['latitude'].dropna()
        all_lons = analyzed_data['longitude'].dropna()
        
        # Create a background track using all available data
        # Sort by timestamp to ensure proper track order
        track_data = analyzed_data.sort_values('timestamp').copy()
        
        # Separate by period for color coding
        colors_period = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Plot each upwind period with different colors
        for pidx, period_id in enumerate(period_ids):
            period_data = track_data[track_data['period_id'] == period_id].copy()
            if len(period_data) > 0:
                # Sort by timestamp
                period_data = period_data.sort_values('timestamp')
                
                ax.plot(period_data['longitude'], period_data['latitude'], 
                       color=colors_period[pidx], linewidth=3, alpha=0.8, 
                       label=f'Upwind Period {pidx+1}', zorder=2)
                
                # Mark start (circle) and end (square) of period
                start_point = period_data.iloc[0]
                end_point = period_data.iloc[-1]
                ax.scatter(start_point['longitude'], start_point['latitude'], 
                          color=colors_period[pidx], s=150, marker='o', 
                          edgecolors='black', linewidth=2.5, zorder=3, label=f'Start P{pidx+1}')
                ax.scatter(end_point['longitude'], end_point['latitude'], 
                          color=colors_period[pidx], s=150, marker='s', 
                          edgecolors='black', linewidth=2.5, zorder=3, label=f'End P{pidx+1}')
        
        # Set labels and title
        ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (°S)', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}\nTop 3 Upwind Periods', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set aspect ratio to be more accurate for latitude/longitude
        # At Sydney latitude (~33°S), 1° longitude ≈ 0.85 × 1° latitude
        lat_center = all_lats.mean()
        lon_lat_ratio = np.cos(np.radians(abs(lat_center)))
        ax.set_aspect(1/lon_lat_ratio)
        
        # Add compass rose (simple N arrow)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Position in top-right corner
        arrow_x = xlim[1] - 0.12 * x_range
        arrow_y = ylim[1] - 0.12 * y_range
        arrow_length = 0.06 * y_range
        
        ax.annotate('N', xy=(arrow_x, arrow_y + arrow_length), 
                   xytext=(arrow_x, arrow_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                   fontsize=16, fontweight='bold', ha='center')
        
        # Add statistics box
        valid_data = analyzed_data[analyzed_data['SOG'] > 0]
        
        if processed_datasets is not None and idx < len(processed_datasets):
            full_data = processed_datasets[idx]
            all_upwind_count = (full_data['SOG'] > 0).sum()
            top3_pct = 100 * len(valid_data) / all_upwind_count if all_upwind_count > 0 else 0
            stats_text = f'All Upwind: {all_upwind_count} records\n'
            stats_text += f'Top 3 Periods: {len(valid_data)} ({top3_pct:.1f}%)\n'
        else:
            stats_text = f'Upwind Records: {len(valid_data)}\n'
        
        stats_text += f'Mean SOG: {valid_data["SOG"].mean():.2f} kt\n'
        stats_text += f'Mean Leeway: {valid_data["Leeway"].mean():.2f}°\n'
        stats_text += f'Median Leeway: {valid_data["Leeway"].median():.2f}°\n'
        stats_text += f'Distance: ~{len(valid_data) * valid_data["SOG"].mean() * 0.000278:.2f} nm'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5))
        
        # Add scale reference (approximate)
        scale_length = 0.001  # degrees longitude
        scale_x = xlim[0] + 0.05 * x_range
        scale_y = ylim[0] + 0.05 * y_range
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
               'k-', linewidth=3)
        # At Sydney latitude, 0.001° ≈ 92 meters
        scale_meters = int(scale_length * 111000 * lon_lat_ratio)
        ax.text(scale_x + scale_length/2, scale_y - 0.01 * y_range, 
               f'{scale_meters}m', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gps_track_map.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: gps_track_map.png")
    
    plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline():
    """Execute complete analysis pipeline"""
    
    print("=" * 80)
    print("COMPLETE SAILING DATA ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Preprocess data
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)
    
    processed_datasets = []
    dataset_names = []
    
    if RAW_DATA_FILE_1 and os.path.exists(RAW_DATA_FILE_1):
        df1 = preprocess_sailing_data(
            RAW_DATA_FILE_1, START_TIME_AEDT, END_TIME_AEDT, DATE_STR,
            COURSE_AXIS, COURSE_AXIS_ADJ,
            UPWIND_MIN_SPEED, UPWIND_MAX_SPEED, MA_WINDOW
        )
        if df1 is not None:
            processed_datasets.append(df1)
            dataset_names.append(os.path.splitext(os.path.basename(RAW_DATA_FILE_1))[0])
            
            # Save preprocessed data
            output_file = f"{OUTPUT_DIR}/{dataset_names[-1]}_processed.csv"
            df1.to_csv(output_file, index=False)
            print(f"  Saved preprocessed data: {output_file}")
    
    if RAW_DATA_FILE_2 and os.path.exists(RAW_DATA_FILE_2):
        df2 = preprocess_sailing_data(
            RAW_DATA_FILE_2, START_TIME_AEDT, END_TIME_AEDT, DATE_STR,
            COURSE_AXIS, COURSE_AXIS_ADJ,
            UPWIND_MIN_SPEED, UPWIND_MAX_SPEED, MA_WINDOW
        )
        if df2 is not None:
            processed_datasets.append(df2)
            dataset_names.append(os.path.splitext(os.path.basename(RAW_DATA_FILE_2))[0])
            
            # Save preprocessed data
            output_file = f"{OUTPUT_DIR}/{dataset_names[-1]}_processed.csv"
            df2.to_csv(output_file, index=False)
            print(f"  Saved preprocessed data: {output_file}")
    
    if len(processed_datasets) == 0:
        print("\nERROR: No datasets to process!")
        return
    
    # Step 2: Analyze data
    print("\n" + "=" * 80)
    print("STEP 2: LEEWAY ANALYSIS")
    print("=" * 80)
    
    analyzed_datasets = []
    for df, name in zip(processed_datasets, dataset_names):
        print(f"\nAnalyzing: {name}")
        result = process_dataset_for_analysis(df, name, LAG_SECONDS)
        analyzed_datasets.append(result)
        
        data, corr_accel, corr_sog, _, _ = result
        print(f"  Records in analysis: {len(data)}")
        print(f"  Leeway vs Acceleration: r = {corr_accel[0]:.4f}, p = {corr_accel[1]:.6f}")
        print(f"  Leeway vs SOG: r = {corr_sog[0]:.4f}, p = {corr_sog[1]:.6f}")
    
    # Step 3: Create visualizations
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    create_analysis_plots(analyzed_datasets, LAG_SECONDS, OUTPUT_DIR)
    # Pass both analyzed data and full processed datasets for GPS map
    create_gps_track_map(analyzed_datasets, OUTPUT_DIR, processed_datasets, dataset_names)
    
    # Step 4: Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    for data, corr_accel, corr_sog, _, name in analyzed_datasets:
        print(f"\n{name.upper()}:")
        print(f"  Records: {len(data)}")
        
        # SOG statistics with median and quartiles
        print(f"  SOG Statistics:")
        print(f"    Median: {data['SOG'].median():.2f} kt")
        print(f"    Mean: {data['SOG'].mean():.2f} kt")
        print(f"    Q1 (25th percentile): {data['SOG'].quantile(0.25):.2f} kt")
        print(f"    Q3 (75th percentile): {data['SOG'].quantile(0.75):.2f} kt")
        print(f"    Std Dev: {data['SOG'].std():.2f} kt")
        print(f"    Min: {data['SOG'].min():.2f} kt")
        print(f"    Max: {data['SOG'].max():.2f} kt")
        
        # Leeway statistics with median and quartiles
        print(f"  Leeway Statistics:")
        print(f"    Median: {data['Leeway'].median():.2f}°")
        print(f"    Mean: {data['Leeway'].mean():.2f}°")
        print(f"    Q1 (25th percentile): {data['Leeway'].quantile(0.25):.2f}°")
        print(f"    Q3 (75th percentile): {data['Leeway'].quantile(0.75):.2f}°")
        print(f"    Std Dev: {data['Leeway'].std():.2f}°")
        print(f"    Min: {data['Leeway'].min():.2f}°")
        print(f"    Max: {data['Leeway'].max():.2f}°")
        
        # Correlations
        print(f"  Correlations:")
        print(f"    Leeway vs Acceleration: r={corr_accel[0]:.4f}, p={corr_accel[1]:.6f}")
        print(f"    Leeway vs SOG: r={corr_sog[0]:.4f}, p={corr_sog[1]:.6f}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    run_complete_pipeline()