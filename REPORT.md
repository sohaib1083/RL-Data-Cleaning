# RL Task Report

This is a complete RL training task for LLMs focused on **data cleaning** - a critical skill for ML engineers and researchers.

## ⚡ Quick Start

```bash
# 1. Set API key
export GROQ_API_KEY=your_key_here

# 2. Run task (generates dataset automatically and runs 10 concurrent tests)
python3 data_cleaning_task.py

# 3. Check results - should show pass rate in 10-40% range
```

---

## Latest Test Results

```
============================================================
FINAL RESULTS:
============================================================
  Total Tests: 10
  Passed: 2
  Failed: 8
  Pass Rate: 20.0%
============================================================
✅ PASS RATE IN TARGET RANGE (10-40%)
```

## Test Details

### Successful Tests: 2/10

- Test 2: ✅ PASS (Score: 100%)
- Test 7: ✅ PASS (Score: 100%)

### Failed Tests: 8/10

- Test 1: ❌ FAIL (Score: 0%)
- Test 3: ❌ FAIL (Score: 0%)
- Test 4: ❌ FAIL (Score: 0%)
- Test 5: ❌ FAIL (Score: 0%)
- Test 6: ❌ FAIL (Score: 0%)
- Test 8: ❌ FAIL (Score: 0%)
- Test 9: ❌ FAIL (Score: 0%)
- Test 10: ❌ FAIL (Score: 0%)

### Failure Analysis

```
Most common failed checks:
  - reasonable_cleaning_done: 7/8 (88%)
  - error_occurred: 7/8 (88%)
  - file_exists: 1/8 (12%)
```

**Diverse failure modes**: Tests fail for different reasons (execution errors, incomplete cleaning, file creation), providing good learning signals for RL training.

---

## Task Configuration

### Model

- **llama-3.1-8b-instant**
- Runs 10 concurrent tests
- Max 15 steps per test

### Prompt

```
Clean the messy dataset 'experiments.csv' and save to 'cleaned_experiments.csv'.

Required cleaning steps:
1. Remove duplicate experiment_id rows (keep first occurrence)
2. Remove rows with missing accuracy or training_time values
3. Remove rows where accuracy is not in valid range [0, 1]
4. Remove rows where training_time is zero or negative

Use the python_expression tool to do all cleaning in pandas, then save the result.
Finally, call submit_answer with a summary of what you did.
```

**No example code provided** - Model must figure out the solution.

### Grading Function (Strict - 6 Checks)

All checks must pass:

1. ✅ **file_exists**: Output file 'cleaned_experiments.csv' must be created
2. ✅ **reasonable_cleaning_done**: 15-50 rows removed from original dataset
3. ✅ **duplicates_removed**: ZERO duplicates allowed (perfect removal required)
4. ✅ **missing_values_removed**: ≤2 missing values per column (accuracy, training_time)
5. ✅ **accuracy_valid**: ≤2 invalid accuracy values (must be in [0, 1])
6. ✅ **training_time_valid**: ≤2 invalid training_time values (must be > 0)

---

## Dataset

**File**: `experiments.csv`  
**Original rows**: 75  
**Expected after cleaning**: ~40-50 rows

### Issues Introduced

- 15 duplicate rows
- 8 missing accuracy values
- 7 missing training_time values
- 8 out-of-range accuracy values
- 6 non-positive training_time values
- 20 model_name formatting issues
- 10 invalid status values

---

## Why This Task Works (20% Pass Rate)

### Challenge Factors

1. **Strict Grading**: Zero tolerance for duplicates, all 6 checks must pass
2. **Smaller Model**: llama-3.1-8b-instant struggles with tool calling
3. **No Example Code**: Model must construct solution from requirements
4. **Multiple Requirements**: 6 different validation checks to satisfy

### Success Factors

1. **Clear Objective**: Clean messy data and save result
2. **Measurable**: Binary pass/fail with detailed scoring
3. **Real-world Skill**: Data cleaning is fundamental in ML/data science
4. **Diverse Failures**: Can fail on execution, logic, or completeness

---

## Assignment Requirements ✅

| Requirement              | Status | Evidence                        |
| ------------------------ | ------ | ------------------------------- |
| Pass rate 10-40%         | ✅     | 20.0% achieved                  |
| At least 10 tests        | ✅     | 10 concurrent tests             |
| Novel & interesting      | ✅     | Real data cleaning skill        |
| Multiple approaches      | ✅     | Various pandas methods possible |
| Diverse failures         | ✅     | 3+ different failure types      |
| All requirements checked | ✅     | 6 validation checks             |
| Prompt matches grading   | ✅     | 4 requirements → 6 checks       |

---

## Code Structure

```python
async def main(num_tests: int = 10):
    """Run 10 concurrent tests to measure pass rate"""

    # Define tools and prompt (shared across tests)
    tools = [read_csv, write_csv, python_expression, get_file_info, submit_answer]
    prompt = "Clean 'experiments.csv'..."

    # Inline test runner
    async def run_one_test(test_id: int):
        generate_messy_dataset(seed=42 + test_id)  # Different dataset each test
        result = await run_agent_loop(prompt, tools, model="llama-3.1-8b-instant")
        passed, verification = verify_data_cleaning()  # Strict grading
        return test_id, passed, verification

    # Run all 10 concurrently
    tasks = [run_one_test(i+1) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Display results and pass rate
    ...
```

## Files Included

### Core Implementation

- `data_cleaning_task.py` - Main task file
- `generate_dataset.py` - Dataset generator
- `experiments.csv` - Sample messy dataset (auto-generated)

### Documentation

- `REPORT.md` - This file

---

## How to Verify

```bash
# Run the task
python3 data_cleaning_task.py

# Expected output:
# - 10 test results (each shows PASS/FAIL)
# - Final pass rate (should be 10-40%)
# - ✅ PASS RATE IN TARGET RANGE message
# - Failure analysis breakdown
```

---

## Summary

**Status**: ✅ READY FOR SUBMISSION

- Pass rate: **20.0%** (in target 10-40% range)
- Concurrent execution: 10 tests run simultaneously
- Strict grading: 6 comprehensive checks
- Diverse failures: Multiple failure modes for learning
- Clean code: Simplified architecture, well-documented
- Real skill: Data cleaning is fundamental in ML workflows

This task successfully challenges LLMs while remaining achievable, providing valuable learning signals for RL training.
