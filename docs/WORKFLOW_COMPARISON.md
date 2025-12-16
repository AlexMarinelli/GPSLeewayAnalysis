# Workflow Comparison: Excel vs Python Pipeline

## Your Previous Excel Workflow

```
1. Open raw CSV in Excel
2. Manually filter by time range
3. Create calculated columns:
   - adj_COG = IF((COG+adjustment)>360, (COG+adjustment)-360, COG+adjustment)
   - adj_HDG = IF((HDG+adjustment)>360, (HDG+adjustment)-360, HDG+adjustment)
   - SOG_MA = AVERAGE(SOG range)
   - COG_MA = AVERAGE(adj_COG range)
   - HDG_MA = AVERAGE(adj_HDG range)
   - Upwind = ABS(MOD(COG_MA-course_axis+180,360)-180)<=90
   - Filtered SOG = IF(AND(Upwind, SOG_MA>min, SOG_MA<max), SOG_MA, 0)
   - Filtered COG = IF(AND(Upwind, SOG_MA>min, SOG_MA<max), COG_MA, 0)
   - Filtered HDG = IF(AND(Upwind, SOG_MA>min, SOG_MA<max), HDG_MA, 0)
4. Copy columns X, Y, Z to new CSV
5. Load CSV into Python analysis script
6. Run analysis
7. Repeat for each session...
```

**Time per session: ~15-30 minutes**
**Error-prone**: Manual time filtering, copy-paste mistakes
**Hard to reproduce**: Different people might filter differently

---

## New Python Pipeline Workflow

```
1. Edit config.py:
   - Set file path
   - Set time window
   - Set parameters

2. Run: python3 run_analysis.py

3. Done!
```

**Time per session: ~30 seconds**
**Reproducible**: Same config = same results
**Automated**: No manual steps

---

## Side-by-Side Comparison

| Task | Excel Method | Python Method |
|------|--------------|---------------|
| Load data | Open in Excel | Automatic |
| Filter by time | Manual selection + formulas | Set START/END time in config |
| Convert UTC to AEDT | Manual calculation | Automatic |
| Apply course adjustment | Complex IF formulas | One parameter |
| Calculate moving average | AVERAGE formulas | One parameter |
| Detect upwind periods | Nested IF formulas | Automatic |
| Apply speed filters | More IF formulas | Two parameters |
| Export filtered data | Copy-paste columns | Automatic |
| Run analysis | Separate script | Integrated |
| Generate charts | Manual or separate | Automatic |
| Process multiple sessions | Repeat everything | Change date, re-run |
| Compare datasets | Very manual | Set second file path |

---

## What Excel Did (Hidden Complexity)

Your Excel file had these key operations:

### 1. Time Window Filter
**Excel**: Manually scroll, select rows between timestamps  
**Python**: `START_TIME_AEDT = "14:53"` and `END_TIME_AEDT = "16:15"`

### 2. Timezone Conversion
**Excel**: Mental calculation of UTC + 11 hours  
**Python**: Automatic with pytz library

### 3. Course Adjustment
**Excel**: 
```
=IF((F6+$H$4)>360, (F6+$H$4)-360, F6+$H$4)
```
**Python**: 
```python
adjust_angle(cog, COURSE_AXIS_ADJ)
```

### 4. Moving Average
**Excel**: 
```
=AVERAGE(D6:D9)  # Must adjust range for each row
```
**Python**: 
```python
df['SOG_MA'] = df['sog_raw'].rolling(window=MA_WINDOW).mean()
```

### 5. Upwind Detection
**Excel**: 
```
=ABS(MOD(L6-65+180,360)-180)<=90
```
**Python**: 
```python
check_upwind(cog, COURSE_AXIS)
```

### 6. Conditional Filtering
**Excel**: 
```
=IF(AND(O6, K6>$H$2, K6<$H$1), K6, 0)
```
**Python**: 
```python
mask = is_upwind & (SOG_MA > min_speed) & (SOG_MA < max_speed)
df.loc[mask, 'SOG'] = df.loc[mask, 'SOG_MA']
```

---

## Configuration Mapping

Your Excel parameters → Python config:

| Excel Cell | Description | Python Config |
|------------|-------------|---------------|
| H1 | upwind max | `UPWIND_MAX_SPEED = 13` |
| H2 | upwind min | `UPWIND_MIN_SPEED = 6` |
| H3 | course axis | `COURSE_AXIS = 65` |
| H4 | adj value | `COURSE_AXIS_ADJ = 0` |
| Manual | Time window | `START_TIME_AEDT = "14:53"` |
| Manual | Time window | `END_TIME_AEDT = "16:15"` |
| 4 samples | MA window | `MA_WINDOW = 4` |

---

## Advantages of Python Pipeline

### 1. **Speed**
- Excel: 15-30 minutes per session
- Python: 30 seconds per session
- **20-60x faster**

### 2. **Accuracy**
- Excel: Manual steps prone to errors
- Python: Same input = same output every time

### 3. **Scalability**
- Excel: One session at a time
- Python: Batch process entire season
```python
for date in ["2025-03-09", "2025-03-10", "2025-03-11"]:
    DATE_STR = date
    run_analysis()
```

### 4. **Comparison**
- Excel: Very manual to compare datasets
- Python: Just set `RAW_DATA_FILE_2`

### 5. **Documentation**
- Excel: Formulas hidden in cells
- Python: All logic visible and commented

### 6. **Version Control**
- Excel: "final_v2_FINAL_use_this.xlsx"
- Python: Git commit history

### 7. **Collaboration**
- Excel: Email files back and forth
- Python: Share config, everyone gets same results

---

## Example: Processing 10 Sessions

### Excel Method
```
1. Open raw CSV #1
2. Set up formulas (15 min)
3. Filter and export (5 min)
4. Run analysis (2 min)
5. Repeat 9 more times...

Total: ~220 minutes (3.7 hours)
```

### Python Method
```python
sessions = [
    ("2025-03-09", "14:53", "16:15"),
    ("2025-03-10", "10:00", "12:00"),
    # ... 8 more sessions
]

for date, start, end in sessions:
    DATE_STR = date
    START_TIME_AEDT = start
    END_TIME_AEDT = end
    run_analysis()

Total: ~5 minutes
```

**40x faster for batch processing**

---

## Migration Checklist

✅ **You have**:
- [x] Raw instrument CSV files
- [x] Time windows in AEDT
- [x] Course axis settings
- [x] Speed filter ranges
- [x] Python scripts ready

✅ **To do**:
1. [ ] Edit `config.py` with your parameters
2. [ ] Run `python3 run_analysis.py`
3. [ ] Verify results match Excel output
4. [ ] Archive Excel workflow (you won't need it!)

---

## Still Need Excel?

You can still use Excel for:
- Quick manual inspection of raw data
- Ad-hoc calculations not in pipeline
- Creating presentation tables
- Importing processed CSVs for custom charts

But preprocessing is now fully automated!

---

## Questions?

See:
- `QUICKSTART.md` - Get started quickly
- `README.md` - Full documentation
- `config.py` - All adjustable parameters
