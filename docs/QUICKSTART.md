# Quick Start Guide

## 3-Step Setup

### Step 1: Edit Configuration
Open `config.py` and set your parameters:

```python
RAW_DATA_FILE_1 = '/path/to/your/data.csv'
DATE_STR = "2025-03-09"
START_TIME_AEDT = "14:53"
END_TIME_AEDT = "16:15"
COURSE_AXIS = 65
```

### Step 2: Run Analysis
```bash
python3 run_analysis.py
```

### Step 3: View Results
Check the `outputs` folder for:
- `*_processed.csv` - Clean data
- `leeway_analysis_*.png` - Analysis charts
- Histogram charts

## Common Adjustments

### Different Time Window
```python
START_TIME_AEDT = "10:00"  # Change start
END_TIME_AEDT = "12:00"    # Change end
```

### Different Speed Range
```python
UPWIND_MIN_SPEED = 8   # Heavier air
UPWIND_MAX_SPEED = 15  # Heavier air
```

### Test Different Lags
```python
LAG_SECONDS = 2  # Try 1, 2, 5
```

## File Structure

```
outputs/
├── config.py                    ← Edit this
├── run_analysis.py              ← Run this
├── complete_sailing_pipeline.py ← Full pipeline
├── sailing_data_pipeline.py     ← Just preprocessing
├── README.md                    ← Full documentation
├── QUICKSTART.md               ← This file
│
└── Generated files:
    ├── *_processed.csv
    ├── leeway_analysis_*.png
    ├── sog_histogram_*.png
    └── leeway_histogram_*.png
```

## Troubleshooting

**"No data in time window"**
- Check date format: YYYY-MM-DD
- Times are in AEDT, not UTC
- Verify data file has correct date

**"Low upwind percentage"**
- Adjust COURSE_AXIS to match wind
- Widen speed range
- Check if tacks are being excluded

**"Module not found"**
```bash
pip install pandas numpy matplotlib scipy pytz --break-system-packages
```

## Next Steps

1. **Compare datasets**: Set `RAW_DATA_FILE_2` in config
2. **Batch process**: Run multiple times with different dates
3. **Export stats**: See README for CSV export code
4. **Customize charts**: Edit visualization functions

## Support

See `README.md` for complete documentation including:
- Detailed parameter explanations
- Advanced usage examples
- Technical details
- Configuration presets
