"""
Configuration File for Sailing Data Analysis Pipeline
======================================================

Edit the parameters below and run:
    python3 complete_sailing_pipeline.py

All configuration is in one place for easy access.
"""

# ============================================================================
# FILE PATHS
# ============================================================================

# Path to raw CSV file from boat instruments
# Format: /path/to/your/file.csv
RAW_DATA_FILE_1 = 'data/raw/2025-03-09 Yandoo.csv'

# Optional: Add a second dataset for comparison
# Set to None if you only have one dataset
RAW_DATA_FILE_2 = 'data/raw/2025-03-09 Lazarus Capital Partners.csv'

# Output directory for processed files and charts
OUTPUT_DIR = 'data/processed'

# ============================================================================
# TIME WINDOW
# ============================================================================

# Date of data collection (YYYY-MM-DD format)
DATE_STR = "2025-03-09"

# Time range in AEDT timezone (HH:MM format)
# Only data between these times will be analyzed
START_TIME_AEDT = "14:53"
END_TIME_AEDT = "16:15"

# ============================================================================
# COURSE PARAMETERS
# ============================================================================

# Target course axis in degrees (0-360)
# This is your intended upwind course
COURSE_AXIS = 65

# Adjustment to apply to all course values (usually 0)
# Use this if your compass has a systematic error
COURSE_AXIS_ADJ = 0

# ============================================================================
# SPEED FILTERS
# ============================================================================

# Minimum speed for valid upwind data (knots)
# Filters out tacks and very light air
UPWIND_MIN_SPEED = 6

# Maximum speed for valid upwind data (knots)
# Filters out reaching/running and very heavy air
UPWIND_MAX_SPEED = 13

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Moving average window size (number of samples)
# Larger values = smoother data, but less responsive
# Recommended: 3-5 samples
MA_WINDOW = 4

# Lag for leeway analysis (seconds)
# How many seconds back to look when analyzing effect of leeway
# Try: 1, 2, 5 seconds
LAG_SECONDS = 1

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Uncomment one of these presets for common scenarios:

# # LIGHT AIR (5-10 knots TWS)
# UPWIND_MIN_SPEED = 4
# UPWIND_MAX_SPEED = 8
# MA_WINDOW = 5

# # MEDIUM AIR (10-15 knots TWS)
# UPWIND_MIN_SPEED = 6
# UPWIND_MAX_SPEED = 13
# MA_WINDOW = 4

# # HEAVY AIR (15+ knots TWS)
# UPWIND_MIN_SPEED = 8
# UPWIND_MAX_SPEED = 15
# MA_WINDOW = 3

# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

# Example 1: Compare two different sails
# RAW_DATA_FILE_1 = '/path/to/jib_session.csv'
# RAW_DATA_FILE_2 = '/path/to/genoa_session.csv'

# Example 2: Morning vs afternoon conditions
# # Morning session:
# START_TIME_AEDT = "09:00"
# END_TIME_AEDT = "11:00"
# # Then run again with:
# # START_TIME_AEDT = "14:00"
# # END_TIME_AEDT = "16:00"

# Example 3: Test different lag periods
# Run the script multiple times with LAG_SECONDS = 1, 2, 5
# Compare results to see optimal lag period

# ============================================================================
# NOTES
# ============================================================================

# Time Zones:
#   - Input CSV file uses UTC timestamps
#   - START_TIME and END_TIME use AEDT (Australia/Sydney)
#   - Pipeline handles conversion automatically

# Course Axis:
#   - Should match your target upwind course
#   - Data within ±90° of course axis is considered "upwind"
#   - Adjust based on wind direction during session

# Speed Filters:
#   - Too wide: Includes tacks and maneuvers (noisy data)
#   - Too narrow: Not enough data for analysis
#   - Adjust based on conditions during session

# Moving Average:
#   - Smooths noisy GPS/instrument data
#   - 4 samples ≈ 2 seconds of data at 2Hz
#   - Larger window = smoother but less detail

# Output Files:
#   - *_processed.csv: Filtered data ready for analysis
#   - leeway_analysis_*.png: Main analysis charts
#   - sog_histogram_*.png: Speed distributions
#   - leeway_histogram_*.png: Leeway distributions
