# Sailing Data Analysis Pipeline - File Index

## 📋 Start Here

**New user?** → Read `QUICKSTART.md` (2 minutes)  
**Want details?** → Read `README.md` (10 minutes)  
**Coming from Excel?** → Read `WORKFLOW_COMPARISON.md` (5 minutes)

---

## 🚀 Quick Start (30 seconds)

1. Edit `config.py` - Set your file path and parameters
2. Run `python3 run_analysis.py`
3. View your charts!

---

## 📁 File Descriptions

### Python Scripts

| File | Purpose | When to Use |
|------|---------|-------------|
| **`run_analysis.py`** | Main script - uses config.py | **USE THIS** - Run every time |
| `config.py` | Configuration file | Edit once per session |
| `complete_sailing_pipeline.py` | Full pipeline (standalone) | If you want parameters in one file |
| `sailing_data_pipeline.py` | Just preprocessing | If you only need clean CSV |

### Documentation

| File | Content | Read Time |
|------|---------|-----------|
| `QUICKSTART.md` | Get started fast | 2 min |
| `README.md` | Complete documentation | 10 min |
| `WORKFLOW_COMPARISON.md` | Excel vs Python comparison | 5 min |
| `INDEX.md` | This file | 1 min |

### Generated Files

| Pattern | Description |
|---------|-------------|
| `*_processed.csv` | Clean data ready for analysis |
| `leeway_analysis_lag*s.png` | Main analysis charts (3 panels) |
| `sog_histogram_lag*s.png` | Speed distribution charts |
| `leeway_histogram_lag*s.png` | Leeway distribution charts |

---

## 🎯 Common Tasks

### First Time Setup
```bash
1. Read QUICKSTART.md
2. Edit config.py
3. python3 run_analysis.py
```

### Analyze New Session
```bash
1. Edit config.py (change DATE, START_TIME, END_TIME)
2. python3 run_analysis.py
```

### Compare Two Datasets
```python
# In config.py:
RAW_DATA_FILE_1 = 'session1.csv'
RAW_DATA_FILE_2 = 'session2.csv'
```

### Change Analysis Parameters
```python
# In config.py:
LAG_SECONDS = 2        # Try different lag
UPWIND_MIN_SPEED = 8   # Adjust speed range
COURSE_AXIS = 70       # Update course axis
```

---

## 📊 What You Get

### Input
```
raw_data.csv (UTC timestamps, all sensor data)
  └─ 33,195 records
```

### Processing
```
1. Filter to time window (AEDT)
   └─ 9,840 records
2. Apply moving average
3. Detect upwind periods
4. Filter by speed range
   └─ 5,357 valid records (54%)
```

### Output
```
Processed CSV:
  - timestamp, lat, lon, SOG, COG, HDG
  - 0 when not valid upwind
  - Values when valid upwind

Charts (PNG):
  - Acceleration by leeway bucket
  - Speed by leeway bucket  
  - SOG vs Leeway scatter plot
  - Distribution histograms
```

---

## 🔧 Configuration Parameters

Quick reference (see `config.py` for full details):

```python
# Required
RAW_DATA_FILE_1 = '/path/to/data.csv'
DATE_STR = "2025-03-09"
START_TIME_AEDT = "14:53"
END_TIME_AEDT = "16:15"

# Course
COURSE_AXIS = 65
COURSE_AXIS_ADJ = 0

# Filters
UPWIND_MIN_SPEED = 6
UPWIND_MAX_SPEED = 13
MA_WINDOW = 4

# Analysis
LAG_SECONDS = 1
```

---

## 🎓 Learning Path

### Beginner
1. `QUICKSTART.md` - Get it working
2. Run with your data
3. Understand the charts

### Intermediate
1. `README.md` - Learn all parameters
2. `WORKFLOW_COMPARISON.md` - Understand the logic
3. Try different configurations

### Advanced
1. Read `complete_sailing_pipeline.py` code
2. Modify visualization functions
3. Add custom analysis
4. Batch process entire season

---

## 💡 Tips

**Starting out?**
- Use default parameters first
- Make one change at a time
- Compare results with Excel (if migrating)

**Getting good results?**
- Try different LAG_SECONDS (1, 2, 5)
- Narrow speed range for specific conditions
- Compare port vs starboard tacks (add second file)

**Want more?**
- Export statistics to CSV (see README)
- Batch process multiple sessions
- Create custom charts
- Integrate with other tools

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| No data in time window | Check date format and timezone |
| Low upwind percentage | Adjust course axis and speed range |
| Module not found | `pip install pandas numpy matplotlib scipy pytz --break-system-packages` |
| Charts don't show trends | Widen time window or adjust filters |

See `README.md` Troubleshooting section for more.

---

## 📈 Expected Results

Good upwind sailing shows:
- **Lower leeway** (0-4°) correlates with **higher speed**
- **Negative correlation** between leeway and acceleration
- **Mean leeway** around 3-6° in optimal conditions
- **Clear trend** in SOG vs Leeway scatter (negative slope)

---

## 🔄 Update History

**Current Version**: v1.0 (2025-12-03)
- Complete preprocessing pipeline
- Integrated analysis workflow  
- Multi-dataset comparison
- Comprehensive documentation

---

## 📝 Next Steps

After your first successful run:

1. ✅ Verify output matches expectations
2. ✅ Archive your Excel workflow (you won't need it!)
3. ✅ Process your entire season's data
4. ✅ Compare different conditions/sails/crew
5. ✅ Share with your sailing team

---

## 📞 Quick Reference

**Just want to run it?**
```bash
python3 run_analysis.py
```

**Just want to change parameters?**
```bash
nano config.py  # or your favorite editor
```

**Just want clean data (no analysis)?**
```bash
python3 sailing_data_pipeline.py
```

**Want everything in one file?**
```bash
python3 complete_sailing_pipeline.py
```

---

## 🎉 You're Ready!

Pick your path:
- **Quick start** → `QUICKSTART.md`
- **Full details** → `README.md`
- **Compare workflows** → `WORKFLOW_COMPARISON.md`
- **Just run it** → `python3 run_analysis.py`

Happy sailing! 🛥️
