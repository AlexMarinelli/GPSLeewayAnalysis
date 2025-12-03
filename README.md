# Sailing Data Analysis Pipeline

Complete Python pipeline for processing raw sailing instrument data and performing leeway analysis.

## Overview

This pipeline replaces your Excel preprocessing workflow with a fully automated Python solution that:
1. Loads raw CSV data from boat instruments
2. Filters by time window (AEDT timezone)
3. Applies moving averages and course adjustments
4. Extracts upwind sailing periods
5. Performs leeway analysis with configurable lag
6. Generates comprehensive visualizations

## Files

### Scripts

1. **`sailing_data_pipeline.py`** - Standalone preprocessing script
   - Converts raw instrument CSV to filtered upwind data
   - Configurable time windows, course axis, and speed filters
   - Outputs clean CSV for analysis

2. **`complete_sailing_pipeline.py`** - Integrated end-to-end pipeline
   - Combines preprocessing and analysis in one workflow
   - Generates all visualizations automatically
   - **RECOMMENDED** for most use cases

### Output Files

- `*_processed.csv` - Preprocessed data (timestamp, lat, lon, SOG, COG, HDG)
- `leeway_analysis_lag*s.png` - Main analysis charts
- `sog_histogram_lag*s.png` - Speed distribution charts
- `leeway_histogram_lag*s.png` - Leeway distribution charts

## Quick Start

### Option 1: Complete Pipeline (Recommended)

```python
# Edit configuration at top of complete_sailing_pipeline.py:

RAW_DATA_FILE_1 = '/path/to/your/raw_data.csv'
DATE_STR = "2025-03-09"
START_TIME_AEDT = "14:53"
END_TIME_AEDT = "16:15"
COURSE_AXIS = 65
LAG_SECONDS = 1

# Run:
python3 complete_sailing_pipeline.py
```

### Option 2: Two-Step Process

```python
# Step 1: Preprocess data
python3 sailing_data_pipeline.py
# Creates: Yandoo_processed.csv

# Step 2: Run your existing analysis on processed data
python3 your_leeway_analysis.py
```

## Configuration Parameters

### Time Window
```python
DATE_STR = "2025-03-09"        # Date of data collection
START_TIME_AEDT = "14:53"      # Start time in AEDT
END_TIME_AEDT = "16:15"        # End time in AEDT
```

### Course Parameters
```python
COURSE_AXIS = 65               # Target course axis (degrees)
COURSE_AXIS_ADJ = 0            # Adjustment to apply
```

### Speed Filters
```python
UPWIND_MIN_SPEED = 6           # Minimum speed for upwind (knots)
UPWIND_MAX_SPEED = 13          # Maximum speed for upwind (knots)
```

### Analysis Parameters
```python
MA_WINDOW = 4                  # Moving average window (samples)
LAG_SECONDS = 1                # Lag for leeway analysis (seconds)
```

## Input Data Format

Raw CSV files should have these columns:
- `ISODateTimeUTC` - Timestamp in ISO format (UTC)
- `Lat` - Latitude
- `Lon` - Longitude
- `SOG` - Speed Over Ground (knots)
- `COG` - Course Over Ground (degrees)
- `Heading` - True heading (degrees)

Example:
```
ISODateTimeUTC,Lat,Lon,SOG,COG,Heading,...
2025-03-09T01:05:44.046000Z,-33.873422,151.242442,0.1,203.7,1.328525,...
```

## Output Data Format

Processed CSV files contain:
```
timestamp,latitude,longitude,SOG,COG,HDG
2025-03-09T03:53:56.060000+0000,-33.864187,151.247145,0,0,0
2025-03-09T03:53:56.561000+0000,-33.864173,151.247143,6.35,354.95,265.88
```

Where:
- Values are 0 when NOT in valid upwind period
- Non-zero values indicate valid upwind sailing with speed in range

## How It Works

### 1. Time Filtering
- Converts UTC timestamps to AEDT (Australia/Sydney timezone)
- Filters data to specified time window
- Handles timezone conversions automatically

### 2. Data Preprocessing
- Applies course axis adjustments to COG and Heading
- Calculates moving averages (default 4-sample window)
- Smooths noisy instrument data

### 3. Upwind Detection
- Determines if boat is sailing upwind based on COG relative to course axis
- Uses ±90° window around course axis
- Filters out off-wind periods

### 4. Speed Filtering
- Applies minimum/maximum speed thresholds
- Excludes tacking, maneuvers, and very light/heavy air
- Focuses analysis on target upwind conditions

### 5. Leeway Analysis
- Calculates leeway (difference between heading and COG)
- Bins leeway into 2-degree buckets
- Analyzes correlation with acceleration and boat speed
- Identifies top 3 longest upwind periods

## Customization

### Adding a Second Dataset

```python
# In complete_sailing_pipeline.py:
RAW_DATA_FILE_1 = '/path/to/first_dataset.csv'
RAW_DATA_FILE_2 = '/path/to/second_dataset.csv'  # Add second file

# Pipeline will automatically:
# - Process both datasets
# - Create comparison visualizations
# - Generate side-by-side statistics
```

### Changing Lag Period

```python
LAG_SECONDS = 2  # Test different lag values (1, 2, 5, etc.)
```

The lag determines how many seconds back to look when analyzing the effect of leeway on acceleration.

### Adjusting Speed Filters

```python
# For heavier air:
UPWIND_MIN_SPEED = 8
UPWIND_MAX_SPEED = 15

# For lighter air:
UPWIND_MIN_SPEED = 4
UPWIND_MAX_SPEED = 10
```

## Troubleshooting

### No Data in Time Window
```
ERROR: No data in time window!
```
**Solution**: Check that:
- Date is correct (YYYY-MM-DD format)
- Times are in AEDT (not UTC)
- Raw data file contains data for that date

### Low Upwind Percentage
```
Valid upwind records: 523 (5.3%)
```
**Solution**: Try:
- Adjusting course axis to match actual wind direction
- Widening speed filter range
- Checking if tacks are being excluded correctly

### Dependencies Missing
```
ModuleNotFoundError: No module named 'pytz'
```
**Solution**:
```bash
pip install pandas numpy matplotlib scipy pytz --break-system-packages
```

## Output Interpretation

### Leeway Analysis Charts
- **Mean Acceleration by Leeway**: Shows how boat speed changes with different leeway angles
- **Mean SOG by Leeway**: Average boat speed at each leeway angle
- **SOG vs Leeway Scatter**: Individual data points showing relationship

### Correlation Values
- **r value**: Strength of relationship (-1 to +1)
  - r > 0.7: Strong correlation
  - r = 0.3-0.7: Moderate correlation
  - r < 0.3: Weak correlation
- **p value**: Statistical significance
  - p < 0.05: Statistically significant
  - p < 0.01: Highly significant

### Expected Results
For good upwind performance:
- Lower leeway (0-4°) should correlate with higher boat speed
- Negative correlation between leeway and acceleration
- Tighter leeway distribution (smaller standard deviation)

## Advanced Usage

### Batch Processing Multiple Sessions
```python
dates = ["2025-03-09", "2025-03-10", "2025-03-11"]
for date in dates:
    DATE_STR = date
    run_complete_pipeline()
```

### Exporting Statistics to CSV
```python
# Add after analysis:
stats_df = pd.DataFrame({
    'Dataset': [name for _, _, _, _, name in analyzed_datasets],
    'Mean_SOG': [data['SOG'].mean() for data, _, _, _, _ in analyzed_datasets],
    'Mean_Leeway': [data['Leeway'].mean() for data, _, _, _, _ in analyzed_datasets],
    'Correlation_r': [corr_sog[0] for _, _, corr_sog, _, _ in analyzed_datasets]
})
stats_df.to_csv('summary_statistics.csv', index=False)
```

## Technical Details

### Moving Average Implementation
Uses pandas rolling window with forward-fill for missing values:
```python
df['SOG_MA'] = df['sog_raw'].rolling(window=4, min_periods=1).mean()
```

### Angle Wrapping
Handles 360° wrapping correctly:
```python
def adjust_angle(angle, adjustment):
    adjusted = angle + adjustment
    if adjusted > 360:
        adjusted -= 360
    elif adjusted < 0:
        adjusted += 360
    return adjusted
```

### Upwind Detection
Calculates shortest angular distance:
```python
def check_upwind(cog, course_axis):
    diff = abs((cog - course_axis + 180) % 360 - 180)
    return diff <= 90
```

## Support

For issues or questions:
1. Check configuration parameters are correct
2. Verify input file format matches expected structure
3. Review console output for specific error messages
4. Ensure all dependencies are installed

## Version History

- **v1.0** (2025-12-03): Initial release
  - Complete preprocessing pipeline
  - Integrated analysis workflow
  - Multi-dataset comparison support

## License

This software is provided as-is for sailing performance analysis purposes.
# GPSLeewayAnalysis
