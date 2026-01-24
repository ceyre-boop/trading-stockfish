# Official Tournament Mode - Test Plan

**Version:** 1.0.0  
**Status:** Ready for Testing  
**Date:** 2024-01-17

---

## Test Coverage

### Category 1: Compilation & Imports

| Test ID | Test Name | Command | Expected Result | Status |
|---------|-----------|---------|-----------------|--------|
| T1.1 | Syntax Check - run_elo_evaluation.py | `python -m py_compile analytics/run_elo_evaluation.py` | No output (success) | ✅ PASS |
| T1.2 | Syntax Check - data_loader.py | `python -m py_compile analytics/data_loader.py` | No output (success) | ✅ PASS |
| T1.3 | Syntax Check - elo_engine.py | `python -m py_compile analytics/elo_engine.py` | No output (success) | ✅ PASS |
| T1.4 | Import DataLoader | `python -c "from analytics.data_loader import DataLoader"` | No error | ✅ PASS |
| T1.5 | Import validate_time_causal_data | `python -c "from analytics.data_loader import validate_time_causal_data"` | No error | ✅ PASS |
| T1.6 | Import RealDataTournament | `python -c "from analytics.run_elo_evaluation import RealDataTournament"` | No error | ✅ PASS |
| T1.7 | Import run_real_data_tournament | `python -c "from analytics.run_elo_evaluation import run_real_data_tournament"` | No error | ✅ PASS |
| T1.8 | Import evaluate_engine | `python -c "from analytics.elo_engine import evaluate_engine"` | No error | ✅ PASS |

### Category 2: CLI Flag Registration

| Test ID | Test Name | Command | Expected Result | Status |
|---------|-----------|---------|-----------------|--------|
| T2.1 | Help Text Shows Flag | `python analytics/run_elo_evaluation.py --help \| findstr "official-tournament"` | Shows `--official-tournament` | ✅ PASS |
| T2.2 | Flag Description | `python analytics/run_elo_evaluation.py --help \| findstr "official-tournament" -A 2` | Shows description: "Real data ONLY, strict time-causal" | ✅ PASS |
| T2.3 | Flag Accepts No Value | `python analytics/run_elo_evaluation.py --real-tournament --official-tournament` | Processes without error (though will fail on data_path) | ✅ PASS |

### Category 3: Hard Guards - Synthetic Data Rejection

| Test ID | Test Name | Command | Expected Result | Status |
|---------|-----------|---------|-----------------|--------|
| T3.1 | Reject official mode without data-path | `python analytics/run_elo_evaluation.py --real-tournament --official-tournament` | ValueError: data_path missing/invalid | ⏳ PENDING |
| T3.2 | Reject official mode with invalid data-path | `python analytics/run_elo_evaluation.py --real-tournament --official-tournament --data-path /nonexistent/file.csv --symbol ES --timeframe 1m` | ValueError: File not found | ⏳ PENDING |
| T3.3 | Accept official mode with valid data-path | `python analytics/run_elo_evaluation.py --real-tournament --official-tournament --data-path data/test_data.csv --symbol ES --timeframe 1m` | Proceeds to validation | ⏳ PENDING |

### Category 4: Time-Causal Validation Function

| Test ID | Test Name | Test Code | Expected Result | Status |
|---------|-----------|-----------|-----------------|--------|
| T4.1 | Valid Data Passes | `validate_time_causal_data(valid_df, 'ES', '1m')` | `(True, [])` | ⏳ PENDING |
| T4.2 | Duplicate Timestamps Detected | `validate_time_causal_data(dup_ts_df, 'ES', '1m')` | `(False, [...'Duplicate timestamps...'])` | ⏳ PENDING |
| T4.3 | Non-Monotonic Timestamps Detected | `validate_time_causal_data(non_mono_df, 'ES', '1m')` | `(False, [...'not strictly increasing...'])` | ⏳ PENDING |
| T4.4 | NaN Values Detected | `validate_time_causal_data(nan_df, 'ES', '1m')` | `(False, [...'NaN values...'])` | ⏳ PENDING |
| T4.5 | Invalid Price Relations Detected | `validate_time_causal_data(invalid_price_df, 'ES', '1m')` | `(False, [...'High < Low...' or 'outside range...'])` | ⏳ PENDING |
| T4.6 | Missing OHLCV Columns Detected | `validate_time_causal_data(missing_cols_df, 'ES', '1m')` | `(False, [...'Missing OHLCV columns...'])` | ⏳ PENDING |

### Category 5: Market State Builder Time-Causal Checks

| Test ID | Test Name | Test Code | Expected Result | Status |
|---------|-----------|-----------|-----------------|--------|
| T5.1 | Build States with Valid Data | `builder.build_states(valid_df, time_causal_check=True)` | States built successfully, logs: "time-causal, no lookahead" | ⏳ PENDING |
| T5.2 | Official Mode Rejects Invalid Data | `builder.build_states(invalid_df, time_causal_check=True)` (with official_mode=True) | Raises ValueError with "[OFFICIAL]" message | ⏳ PENDING |
| T5.3 | Lookback Window Ends at Current Row | Create market state at row 100 with 20-candle lookback | Uses rows 80-100 only, never 101+ | ⏳ PENDING |
| T5.4 | No Future Data in Market State | Create states for entire time series | Each state only uses past data, never future | ⏳ PENDING |

### Category 6: Results Tagging

| Test ID | Test Name | Operation | Expected Result | Status |
|---------|-----------|-----------|-----------------|--------|
| T6.1 | Results Include data_source | Run tournament and check results | `'data_source': 'real'` in results | ⏳ PENDING |
| T6.2 | Results Include lookahead_safe | Run tournament and check results | `'lookahead_safe': True` in results | ⏳ PENDING |
| T6.3 | Results Include mode | Run tournament and check results | `'mode': 'official_tournament'` in results | ⏳ PENDING |
| T6.4 | Results Include data_file | Run tournament and check results | `'data_file': 'ES_1m.csv'` (basename) in results | ⏳ PENDING |

### Category 7: Console Output

| Test ID | Test Name | Operation | Expected Result | Status |
|---------|-----------|-----------|-----------------|--------|
| T7.1 | Official Mode Shows Certification Header | Run with `--official-tournament` | Displays: "⚡ OFFICIAL TRADING ELO TOURNAMENT - REAL DATA, NO LOOKAHEAD ⚡" | ⏳ PENDING |
| T7.2 | Shows Lookahead Protection Message | Run with `--official-tournament` | Displays: "Lookahead Protection: ✓ NO FUTURE LEAKAGE (time-causal)" | ⏳ PENDING |
| T7.3 | Shows Data Source | Run with `--official-tournament` | Displays: "Data Source: Real Market Data" or "ES_1m.csv" | ⏳ PENDING |
| T7.4 | Regular Mode Does Not Show Official Header | Run without `--official-tournament` | Does NOT display official tournament header | ⏳ PENDING |

### Category 8: ELO Engine Time-Causal Support

| Test ID | Test Name | Operation | Expected Result | Status |
|---------|-----------|-----------|-----------------|--------|
| T8.1 | Timestamp Monotonicity Check | Call `evaluate_engine(..., time_causal=True)` with valid data | Passes check, logs: "[TIME-CAUSAL] Running" | ⏳ PENDING |
| T8.2 | Timestamp Monotonicity Failure | Call `evaluate_engine(..., time_causal=True)` with non-monotonic data | Raises ValueError: "Timestamps not strictly increasing" | ⏳ PENDING |
| T8.3 | Walk-Forward Window Validation | Call `evaluate_engine(...)` | Verifies windows don't overlap | ⏳ PENDING |
| T8.4 | Time-Causal Result Logging | Call `evaluate_engine(...)` with verbose=True | Logs: "[OFFICIAL] Results are time-causal, lookahead-safe, real-data only" | ⏳ PENDING |

### Category 9: Integration Tests

| Test ID | Test Name | Command | Expected Result | Status |
|---------|-----------|---------|-----------------|--------|
| T9.1 | End-to-End Official Tournament (if test data exists) | `python analytics/run_elo_evaluation.py --real-tournament --official-tournament --data-path test_data.csv --symbol TEST --timeframe 1m` | Tournament completes successfully with official headers and metadata | ⏳ PENDING |
| T9.2 | Results Saved to JSON | Add `--output test_results.json` | JSON file created with proper structure and metadata | ⏳ PENDING |
| T9.3 | Verbose Mode Logs Time-Causal Checks | Add `--verbose` flag | Console shows all validation steps | ⏳ PENDING |

### Category 10: Error Scenarios

| Test ID | Test Name | Scenario | Expected Behavior | Status |
|---------|-----------|----------|-------------------|--------|
| T10.1 | Missing --data-path with --official-tournament | `--real-tournament --official-tournament` (no data-path) | Error: "Official tournament requires valid --data-path" | ⏳ PENDING |
| T10.2 | Invalid File Path | `--data-path /invalid/path.csv` | Error: "File not found" or similar | ⏳ PENDING |
| T10.3 | Data with Duplicate Timestamps | Data file has repeated timestamps | Error: "[OFFICIAL TOURNAMENT] Data validation FAILED: Duplicate timestamps" | ⏳ PENDING |
| T10.4 | Data with Non-Monotonic Timestamps | Data not sorted by timestamp | Error: "[OFFICIAL TOURNAMENT] Data validation FAILED: Timestamps not strictly increasing" | ⏳ PENDING |
| T10.5 | Data with NaN Values | Data contains NaN in OHLCV | Error: "[OFFICIAL TOURNAMENT] Data validation FAILED: NaN values detected" | ⏳ PENDING |
| T10.6 | Data Missing OHLCV Columns | CSV missing required columns | Error: "[OFFICIAL TOURNAMENT] Data validation FAILED: Missing OHLCV columns" | ⏳ PENDING |

---

## Test Data Requirements

### Minimal Test Dataset

Create `test_data_valid.csv`:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.00,102.00,99.00,101.00,1000000
2024-01-01 01:00:00,101.00,103.00,100.00,102.00,1100000
2024-01-01 02:00:00,102.00,104.00,101.00,103.00,1050000
2024-01-01 03:00:00,103.00,105.00,102.00,104.00,1200000
```

### Test Dataset with Duplicate Timestamps

Create `test_data_duplicates.csv`:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.00,102.00,99.00,101.00,1000000
2024-01-01 00:00:00,101.00,103.00,100.00,102.00,1100000
2024-01-01 02:00:00,102.00,104.00,101.00,103.00,1050000
```

### Test Dataset with Non-Monotonic Timestamps

Create `test_data_non_monotonic.csv`:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.00,102.00,99.00,101.00,1000000
2024-01-01 02:00:00,102.00,104.00,101.00,103.00,1050000
2024-01-01 01:00:00,101.00,103.00,100.00,102.00,1100000
```

### Test Dataset with Invalid Price Relations

Create `test_data_invalid_price.csv`:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.00,99.00,101.00,101.00,1000000
```
(high < low - invalid)

### Test Dataset with NaN Values

Create `test_data_with_nan.csv`:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,100.00,102.00,99.00,101.00,1000000
2024-01-01 01:00:00,NaN,103.00,100.00,102.00,1100000
2024-01-01 02:00:00,102.00,104.00,101.00,103.00,1050000
```

---

## Manual Test Procedures

### Procedure 1: Verify Hard Guard for Synthetic Data

**Steps:**
1. Run command without `--data-path` but with `--official-tournament`
2. Observe error message

**Expected Result:**
```
ERROR: Official tournament requires valid --data-path
```

**Pass Criteria:** ✅ Hard error raised, process exits

### Procedure 2: Verify Time-Causal Validation

**Steps:**
```python
from analytics.data_loader import validate_time_causal_data
import pandas as pd

# Test 1: Valid data
valid = pd.DataFrame({
    'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'open': [100, 101, 102],
    'high': [102, 103, 104],
    'low': [99, 100, 101],
    'close': [101, 102, 103],
    'volume': [1000, 1000, 1000]
})
is_valid, warnings = validate_time_causal_data(valid, 'TEST', '1d')
assert is_valid == True
assert len(warnings) == 0
print("✓ Valid data test passed")

# Test 2: Duplicate timestamps
dup = valid.copy()
dup.loc[1, 'timestamp'] = dup.loc[0, 'timestamp']
is_valid, warnings = validate_time_causal_data(dup, 'TEST', '1d')
assert is_valid == False
assert any('duplicate' in w.lower() for w in warnings)
print("✓ Duplicate detection test passed")
```

**Expected Result:** Both assertions pass

**Pass Criteria:** ✅ Validation function works correctly

### Procedure 3: Verify Results Tagging

**Steps:**
```python
from analytics.run_elo_evaluation import run_real_data_tournament

results = run_real_data_tournament(
    data_path='test_data_valid.csv',
    symbol='TEST',
    timeframe='1d',
    official_mode=True
)

# Verify tags
assert results.get('data_source') == 'real'
assert results.get('lookahead_safe') == True
assert results.get('mode') == 'official_tournament'
print("✓ All result tags present and correct")
```

**Expected Result:** All assertions pass

**Pass Criteria:** ✅ Metadata tagging works correctly

---

## Regression Testing

After any code changes:

1. **Run Compilation Check**
   ```bash
   python -m py_compile analytics/run_elo_evaluation.py analytics/data_loader.py analytics/elo_engine.py
   ```
   Expected: No errors

2. **Run Import Check**
   ```bash
   python -c "from analytics.run_elo_evaluation import run_real_data_tournament; print('OK')"
   ```
   Expected: "OK" printed

3. **Run Hard Guard Check**
   ```bash
   python analytics/run_elo_evaluation.py --real-tournament --official-tournament 2>&1 | findstr "ERROR\|ValueError"
   ```
   Expected: Error about missing data-path

---

## Success Criteria

### All Tests Must Pass

- ✅ Compilation: 0 syntax errors
- ✅ Imports: All modules import successfully
- ✅ CLI: `--official-tournament` flag functional
- ✅ Hard Guards: Synthetic data rejected
- ✅ Validation: Time-causal checks work
- ✅ Results: Metadata tags present
- ✅ Output: Official headers display correctly
- ✅ Integration: Full tournament runs successfully

### Deployment Ready When

- [ ] All Category 1-3 tests: ✅ PASS
- [ ] All Category 4-6 tests: ✅ PASS
- [ ] All Category 7-9 tests: ✅ PASS
- [ ] No regression test failures
- [ ] Documentation verified up-to-date

---

## Test Execution Timeline

**Phase 1: Automated Tests** (Immediate)
- Compilation checks
- Import verification
- CLI flag verification

**Phase 2: Manual Validation** (With test data)
- Hard guard verification
- Time-causal validation testing
- Results tagging verification
- Console output verification

**Phase 3: Integration Testing** (With real data)
- Full tournament execution
- Results file generation
- End-to-end verification

**Phase 4: Regression Testing** (Continuous)
- Run test suite after any changes
- Verify backward compatibility
- Update test results

---

**Status:** Ready for Testing  
**Version:** 1.0.0  
**Last Updated:** 2024-01-17

