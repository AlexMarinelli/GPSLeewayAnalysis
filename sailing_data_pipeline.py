import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pytz

# ============================================================================
# CONFIGURATION PARAMETERS - ADJUST THESE AS NEEDED
# ============================================================================

# Time window (AEDT timezone)
START_TIME_AEDT = "14:53"  # Format: "HH:MM"
END_TIME_AEDT = "16:15"    # Format: "HH:MM"
DATE_STR = "2025-03-09"     # Date of the data

# Course parameters
COURSE_AXIS = 65            # Target course axis in degrees
COURSE_AXIS_ADJ = 0         # Adjustment to course axis

# Speed filters
UPWIND_MIN_SPEED = 6        # Minimum speed for upwind (knots)
UPWIND_MAX_SPEED = 13       # Maximum speed for upwind (knots)

# Moving average window (for smoothing)
MA_WINDOW = 4               # Number of samples for moving average

# File paths
INPUT_FILE = '/mnt/user-data/uploads/1764753121610_2025-03-09_Yandoo.csv'
OUTPUT_FILE = '/mnt/user-data/outputs/Yandoo_processed.csv'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def time_to_seconds(time_str):
    """Convert HH:MM time string to seconds since midnight"""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60

def filter_by_time_window(df, start_time_str, end_time_str, date_str, timezone_str='Australia/Sydney'):
    """
    Filter dataframe by time window in local timezone
    
    Parameters:
    - df: DataFrame with 'ISODateTimeUTC' column
    - start_time_str: Start time in "HH:MM" format
    - end_time_str: End time in "HH:MM" format  
    - date_str: Date string in "YYYY-MM-DD" format
    - timezone_str: Local timezone (default: Australia/Sydney for AEDT)
    """
    # Parse UTC timestamps
    df['timestamp_utc'] = pd.to_datetime(df['ISODateTimeUTC'])
    
    # Convert to local timezone
    local_tz = pytz.timezone(timezone_str)
    df['timestamp_local'] = df['timestamp_utc'].dt.tz_convert(local_tz)
    
    # Create start and end datetime objects in local timezone
    start_dt = local_tz.localize(
        datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %H:%M")
    )
    end_dt = local_tz.localize(
        datetime.strptime(f"{date_str} {end_time_str}", "%Y-%m-%d %H:%M")
    )
    
    # Filter by time window
    mask = (df['timestamp_local'] >= start_dt) & (df['timestamp_local'] <= end_dt)
    
    return df[mask].copy()

def calculate_moving_average(series, window):
    """Calculate moving average with specified window"""
    return series.rolling(window=window, min_periods=1).mean()

def adjust_angle(angle, adjustment):
    """Adjust angle and wrap to 0-360 range"""
    adjusted = angle + adjustment
    if adjusted > 360:
        adjusted -= 360
    elif adjusted < 0:
        adjusted += 360
    return adjusted

def check_upwind(cog, course_axis):
    """
    Check if sailing upwind based on COG and course axis
    Returns True if angle difference from course axis is <= 90 degrees
    """
    # Calculate shortest angle difference
    diff = abs((cog - course_axis + 180) % 360 - 180)
    return diff <= 90

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_sailing_data(input_file, output_file, 
                         start_time, end_time, date_str,
                         course_axis, course_axis_adj,
                         upwind_min_speed, upwind_max_speed,
                         ma_window):
    """
    Complete pipeline to process raw sailing data
    """
    
    print("=" * 80)
    print("SAILING DATA PROCESSING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    print(f"  Date: {date_str}")
    print(f"  Time window (AEDT): {start_time} to {end_time}")
    print(f"  Course axis: {course_axis}°")
    print(f"  Course axis adjustment: {course_axis_adj}°")
    print(f"  Upwind speed range: {upwind_min_speed}-{upwind_max_speed} knots")
    print(f"  Moving average window: {ma_window} samples")
    
    # Step 1: Load raw data
    print("\nStep 1: Loading raw data...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} records")
    
    # Step 2: Filter by time window
    print("\nStep 2: Filtering by time window...")
    df_filtered = filter_by_time_window(df, start_time, end_time, date_str)
    print(f"  Retained {len(df_filtered)} records within time window")
    
    if len(df_filtered) == 0:
        print("ERROR: No data within specified time window!")
        return None
    
    # Step 3: Extract and rename relevant columns
    print("\nStep 3: Extracting relevant columns...")
    df_proc = pd.DataFrame()
    df_proc['timestamp'] = df_filtered['ISODateTimeUTC']
    df_proc['latitude'] = df_filtered['Lat']
    df_proc['longitude'] = df_filtered['Lon']
    df_proc['sog_raw'] = df_filtered['SOG']
    df_proc['cog_raw'] = df_filtered['COG']
    df_proc['hdg_raw'] = df_filtered['Heading']  # True heading
    
    # Step 4: Apply adjustments
    print("\nStep 4: Applying course adjustments...")
    df_proc['cog_adj'] = df_proc['cog_raw'].apply(
        lambda x: adjust_angle(x, course_axis_adj)
    )
    df_proc['hdg_adj'] = df_proc['hdg_raw'].apply(
        lambda x: adjust_angle(x, course_axis_adj)
    )
    
    # Step 5: Calculate moving averages
    print("\nStep 5: Calculating moving averages...")
    df_proc['SOG_MA'] = calculate_moving_average(df_proc['sog_raw'], ma_window)
    df_proc['COG_MA'] = calculate_moving_average(df_proc['cog_adj'], ma_window)
    df_proc['HDG_MA'] = calculate_moving_average(df_proc['hdg_adj'], ma_window)
    
    # Step 6: Determine upwind periods
    print("\nStep 6: Identifying upwind periods...")
    df_proc['is_upwind'] = df_proc['COG_MA'].apply(
        lambda x: check_upwind(x, course_axis)
    )
    
    # Step 7: Apply speed filters for upwind
    print("\nStep 7: Applying speed filters...")
    df_proc['upwind_speed_ok'] = (
        (df_proc['SOG_MA'] > upwind_min_speed) & 
        (df_proc['SOG_MA'] < upwind_max_speed)
    )
    
    # Step 8: Create final filtered columns
    print("\nStep 8: Creating final filtered output...")
    # Initialize output columns
    df_proc['SOG'] = 0.0
    df_proc['COG'] = 0.0
    df_proc['HDG'] = 0.0
    
    # Apply filters: only include data when upwind AND speed is in range
    mask = df_proc['is_upwind'] & df_proc['upwind_speed_ok']
    df_proc.loc[mask, 'SOG'] = df_proc.loc[mask, 'SOG_MA']
    df_proc.loc[mask, 'COG'] = df_proc.loc[mask, 'COG_MA']
    df_proc.loc[mask, 'HDG'] = df_proc.loc[mask, 'HDG_MA']
    
    # Step 9: Create final output
    print("\nStep 9: Preparing final output...")
    output_df = df_proc[['timestamp', 'latitude', 'longitude', 'SOG', 'COG', 'HDG']].copy()
    
    # Round to appropriate precision
    output_df['latitude'] = output_df['latitude'].round(6)
    output_df['longitude'] = output_df['longitude'].round(6)
    output_df['SOG'] = output_df['SOG'].round(2)
    output_df['COG'] = output_df['COG'].round(2)
    output_df['HDG'] = output_df['HDG'].round(2)
    
    # Step 10: Save output
    print("\nStep 10: Saving output...")
    output_df.to_csv(output_file, index=False)
    print(f"  Saved {len(output_df)} records to {output_file}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("PROCESSING STATISTICS")
    print("=" * 80)
    
    non_zero_records = (output_df['SOG'] > 0).sum()
    print(f"\nTotal records: {len(output_df)}")
    print(f"Records with valid upwind data: {non_zero_records}")
    print(f"Records filtered out: {len(output_df) - non_zero_records}")
    print(f"Upwind percentage: {100 * non_zero_records / len(output_df):.1f}%")
    
    if non_zero_records > 0:
        valid_data = output_df[output_df['SOG'] > 0]
        print(f"\nValid data statistics:")
        print(f"  SOG - Mean: {valid_data['SOG'].mean():.2f} kt, "
              f"Min: {valid_data['SOG'].min():.2f} kt, "
              f"Max: {valid_data['SOG'].max():.2f} kt")
        print(f"  COG - Mean: {valid_data['COG'].mean():.2f}°, "
              f"Min: {valid_data['COG'].min():.2f}°, "
              f"Max: {valid_data['COG'].max():.2f}°")
        print(f"  HDG - Mean: {valid_data['HDG'].mean():.2f}°, "
              f"Min: {valid_data['HDG'].min():.2f}°, "
              f"Max: {valid_data['HDG'].max():.2f}°")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    return output_df

# ============================================================================
# RUN PIPELINE
# ============================================================================

if __name__ == "__main__":
    result = process_sailing_data(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        start_time=START_TIME_AEDT,
        end_time=END_TIME_AEDT,
        date_str=DATE_STR,
        course_axis=COURSE_AXIS,
        course_axis_adj=COURSE_AXIS_ADJ,
        upwind_min_speed=UPWIND_MIN_SPEED,
        upwind_max_speed=UPWIND_MAX_SPEED,
        ma_window=MA_WINDOW
    )
    
    if result is not None:
        print(f"\nOutput file ready at: {OUTPUT_FILE}")
        print("You can now use this file with your leeway analysis script!")
