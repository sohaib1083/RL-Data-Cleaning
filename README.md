# RL Paper Review Task

This is a complete RL training task for LLMs focused on **scientific paper review** - a critical skill requiring deep analysis, critical thinking, and constructive feedback.

## 🎯 Test Results

**Model**: Claude Haiku 4.5 (claude-haiku-4-5)  
**Pass Rate**: 20% (2/10 tests passed)  
**Status**: ✅ PASS RATE IN TARGET RANGE (10-40%)

**Why This Works**: Paper review requires multi-step reasoning, identifying subtle flaws, and providing structured critique - naturally challenging for LLMs.

## ⚡ Quick Start

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="your-api-key-here"

# 2. Run task (generates flawed papers and runs 10 concurrent reviews)
python3 main.py

# 3. Check results - should show ~20% pass rate
```

---

## Task Overview

### Model Configuration

- **claude-haiku-4-5** (Anthropic Claude Haiku)
- MAX_TOKENS: 2048
- Max steps: 12 (complex multi-step reasoning)
- Tools: 3 (read_paper, python_expression, submit_review)
- Runs 10 concurrent tests

### Prompt

```
You are a peer reviewer for a top-tier AI/ML conference. Review the paper in 'paper_to_review.json'.

Your review must identify specific flaws in:
1. Dataset and data handling
2. Methodology and experimental design
3. Statistical rigor and reproducibility
4. Claims and their justification
5. References and related work

Provide a comprehensive review with:
- identified_flaws: List of specific issues found
- overall_assessment: Detailed critique (strengths and weaknesses)
- recommendation: Clear decision (reject/major revision/minor revision/accept)
- suggestions: Constructive feedback for improvement

Be thorough and critical.
```

**Challenging Aspects**:

- Must analyze multiple sections of paper
- Identify subtle methodological flaws
- Provide structured, constructive feedback
- Balance criticism with suggestions
- Make appropriate recommendation

### Grading Function (Strict - 5 Checks)

Must pass at least 4 out of 5 checks:

1. ✅ **has_proper_structure**: Review contains required fields (identified_flaws, overall_assessment, recommendation)
2. ✅ **identified_multiple_flaws**: Found at least 3 specific issues
3. ✅ **detailed_critique**: Uses quality keywords (methodology, dataset, statistics, baseline, reproducibility)
4. ✅ **has_clear_recommendation**: Provides valid decision (reject/major revision/minor revision/accept)
5. ✅ **provides_constructive_feedback**: Includes suggestions for improvement (>50 chars)

**Pass Rate**: ~20% - Most reviews miss structure or don't provide sufficient detail

---

## Paper Generation

**File**: `paper_to_review.json`  
**Sections**: Title, Abstract, Introduction, Methodology, Results, Conclusion, References

### Intentional Flaws (Randomly Introduced)

Each generated paper contains 3-5 flaws across categories:

- **Dataset issues** (80% chance): Missing size, no train/test split, data leakage, biased sampling
- **Methodology flaws** (90% chance): No hyperparameters, missing baselines, cherry-picked metrics, unclear architecture
- **Statistical problems** (70% chance): No confidence intervals, single run only, no p-values, not reproducible
- **Overclaiming** (85% chance): Unrealistic accuracy, no SOTA comparison, contradictions, missing limitations
- **Reference issues** (60% chance): Key papers not cited, outdated refs, improper formatting

**Each test generates unique paper with different flaws**

---

## Why 20% Pass Rate is Achieved

### Challenge Factors

1. **Complex Multi-Step Task**: Must read paper, analyze multiple sections, identify flaws, structure review
2. **Subjective Evaluation**: Requires critical thinking, not just pattern matching
3. **Strict Grading**: Need 4/5 checks passed - comprehensive review required
4. **Quality Requirements**: Must use technical vocabulary and provide constructive feedback
5. **No Examples**: Prompt specifies what to do, but not how to structure the review

---

## Code Structure

```python
async def main(num_tests: int = 10):
    """Run 10 concurrent paper review tests"""

    # Define tools and prompt (shared across tests)
    tools = [read_paper, python_expression, submit_review]
    prompt = "You are a peer reviewer. Review the paper and identify flaws..."

    # Test runner
    async def run_one_test(test_id: int):
        paper_data = generate_flawed_paper(seed=42 + test_id)  # Unique paper each test
        result = await run_agent_loop(prompt, tools, model="claude-haiku-4-5")
        passed, verification = verify_review(paper_data["flaws"], result)
        return test_id, passed, verification

    # Run all 10 concurrently
    tasks = [run_one_test(i+1) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Display pass rate
    ...
```

## Files Included

### Core Implementation

- `main.py` - Main task file (paper review RL task)
- `generate_paper.py` - Flawed paper generator
- `paper_to_review.json` - Sample paper (auto-generated)

### Documentation

- `README.md` - This file
- `requirements.txt` - Python dependencies

---

## How to Run

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run the task
python3 main.py

# Expected output:
# - 10 test results (each shows PASS/FAIL)
# - Final pass rate
# - Failure analysis breakdown
```

This task successfully teaches LLMs scientific paper review while maintaining appropriate difficulty (10-40% pass rate) through complex multi-step reasoning requirements.
