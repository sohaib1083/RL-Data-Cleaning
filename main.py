"""
Data Cleaning RL Task for LLM Training
Task: Clean a messy ML experiments dataset
"""
import asyncio
import json
import os
import pandas as pd
from collections.abc import Callable
from typing import Any, TypedDict

from groq import AsyncGroq
from generate_dataset import generate_messy_dataset

MAX_TOKENS = 2000


class FileOperationResult(TypedDict):
    success: bool
    message: str
    data: Any | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def read_csv_tool(filename: str) -> FileOperationResult:
    """
    Tool to read CSV file and return preview with statistics
    """
    try:
        if not os.path.exists(filename):
            return {
                "success": False,
                "message": f"File '{filename}' not found",
                "data": None
            }
        
        df = pd.read_csv(filename)
        
        # Provide useful information
        info = {
            "filename": filename,
            "rows": len(df),
            "columns": list(df.columns),
            "first_5_rows": df.head(5).to_dict(orient='records'),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        
        return {
            "success": True,
            "message": f"Successfully read {filename}",
            "data": info
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error reading file: {str(e)}",
            "data": None
        }


def write_csv_tool(filename: str, csv_content: str) -> FileOperationResult:
    """
    Tool to write CSV content to file
    Accepts CSV as string
    """
    try:
        with open(filename, 'w') as f:
            f.write(csv_content)
        
        # Verify it was written correctly
        df = pd.read_csv(filename)
        
        return {
            "success": True,
            "message": f"Successfully wrote {len(df)} rows to {filename}",
            "data": {"rows_written": len(df), "filename": filename}
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error writing file: {str(e)}",
            "data": None
        }


def python_expression_tool(expression: str) -> dict[str, Any]:
    """
    Tool that evaluates Python expressions.
    Pandas is available as 'pd'.
    """
    try:
        # Make pandas available
        import pandas as pd
        import numpy as np
        from io import StringIO
        import sys
        
        namespace = {"pd": pd, "np": np, "StringIO": StringIO}
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Try to evaluate as expression first
            try:
                result = eval(expression, namespace, namespace)
                if result is not None:
                    print(result)
            except SyntaxError:
                # Not an expression, execute as statement
                exec(expression, namespace, namespace)
            
            output = sys.stdout.getvalue()
            
        finally:
            sys.stdout = old_stdout
        
        return {"result": output, "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def get_file_info_tool(filename: str) -> FileOperationResult:
    """
    Tool to get information about a file
    """
    try:
        if not os.path.exists(filename):
            return {
                "success": False,
                "message": f"File '{filename}' does not exist",
                "data": None
            }
        
        df = pd.read_csv(filename)
        
        info = {
            "exists": True,
            "filename": filename,
            "rows": len(df),
            "columns": list(df.columns),
            "file_size_bytes": os.path.getsize(filename)
        }
        
        return {
            "success": True,
            "message": f"File info for {filename}",
            "data": info
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting file info: {str(e)}",
            "data": None
        }


def submit_answer_tool(summary: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer with cleaning summary
    """
    return {"answer": summary, "submitted": True}

# graing function
def verify_data_cleaning(original_file: str = "experiments.csv", 
                        cleaned_file: str = "cleaned_experiments.csv") -> tuple[bool, dict[str, Any]]:
    """
    STRICTER verification of data cleaning task - targets 10-40% pass rate
    
    Core Requirements (Must Pass ALL):
    1. File exists
    2. Reasonable amount of data removed (20-50 rows)
    3. Duplicates mostly removed (allow ≤1 remaining)
    4. Missing values mostly removed (allow ≤2 per column)
    5. Accuracy values valid (in range [0, 1], allow ≤2 invalid)
    6. Training time values valid (> 0, allow ≤2 invalid)
    
    This is calibrated to achieve 10-40% pass rate
    """
    results = {
        "passed": False,
        "checks": {},
        "details": {},
        "score": 0
    }
    
    try:
        # Check if cleaned file exists
        if not os.path.exists(cleaned_file):
            results["checks"]["file_exists"] = False
            results["details"]["error"] = f"Output file '{cleaned_file}' not found"
            return False, results
        
        results["checks"]["file_exists"] = True
        
        # Load both files
        df_original = pd.read_csv(original_file)
        df_cleaned = pd.read_csv(cleaned_file)
        
        results["details"]["original_rows"] = len(df_original)
        results["details"]["cleaned_rows"] = len(df_cleaned)
        results["details"]["rows_removed"] = len(df_original) - len(df_cleaned)
        
        # STRICTER Check 1: Reasonable cleaning (15-50 rows removed)
        rows_removed = len(df_original) - len(df_cleaned)
        results["checks"]["reasonable_cleaning_done"] = (15 <= rows_removed <= 50)
        results["details"]["rows_removed"] = rows_removed
        
        # STRICTER Check 2: Duplicates completely removed (allow ≤0)
        dup_count = df_cleaned.duplicated(subset=['experiment_id']).sum()
        results["checks"]["duplicates_removed"] = (dup_count == 0)
        results["details"]["duplicate_ids_remaining"] = int(dup_count)
        
        # STRICTER Check 3: Missing values mostly removed (allow ≤2 per column)
        missing_acc = df_cleaned['accuracy'].isna().sum()
        missing_time = df_cleaned['training_time'].isna().sum()
        results["checks"]["missing_values_removed"] = (missing_acc <= 2 and missing_time <= 2)
        results["details"]["missing_accuracy"] = int(missing_acc)
        results["details"]["missing_training_time"] = int(missing_time)
        
        # STRICTER Check 4: Accuracy values valid (in [0, 1], allow ≤2 invalid)
        invalid_acc = ((df_cleaned['accuracy'] < 0) | (df_cleaned['accuracy'] > 1)).sum()
        results["checks"]["accuracy_valid"] = (invalid_acc <= 2)
        results["details"]["invalid_accuracy_count"] = int(invalid_acc)
        
        # STRICTER Check 5: Training time values valid (> 0, allow ≤2 invalid)
        invalid_time = (df_cleaned['training_time'] <= 0).sum()
        results["checks"]["training_time_valid"] = (invalid_time <= 2)
        results["details"]["invalid_training_time_count"] = int(invalid_time)
        
        # Calculate score
        checks_passed = sum(1 for v in results["checks"].values() if v)
        total_checks = len(results["checks"])
        results["score"] = checks_passed / total_checks if total_checks > 0 else 0
        
        # Overall pass: ALL checks must pass
        results["passed"] = all(results["checks"].values())
        
        return results["passed"], results
        
    except Exception as e:
        results["checks"]["error_occurred"] = False
        results["details"]["error"] = str(e)
        return False, results


async def run_agent_loop(
    prompt: str,
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 15,
    model: str = "llama-3.3-70b-versatile",  # Using more capable model by default
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.
    """
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.chat.completions.create(
            model=model, 
            max_tokens=MAX_TOKENS, 
            tools=tools, 
            messages=messages,
            tool_choice="auto"
        )

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason
        
        if finish_reason == "length":
            if verbose:
                print(f"Model reached max_tokens limit {MAX_TOKENS}")

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process text content
        if message.content:
            if verbose:
                print(f"Assistant: {message.content}")

        # Process tool calls if present
        if message.tool_calls:
            has_tool_use = True
            
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                
                if tool_name in tool_handlers:
                    if verbose:
                        print(f"\nUsing tool: {tool_name}")

                    # Parse arguments
                    tool_input = json.loads(tool_call.function.arguments)
                    handler = tool_handlers[tool_name]

                    # Execute tool
                    if verbose and tool_name not in ["submit_answer"]:
                        print(f"Input: {tool_input}")
                    
                    result = handler(**tool_input) if isinstance(tool_input, dict) else handler(tool_input)
                    
                    if verbose and tool_name not in ["submit_answer"]:
                        print(f"Output: {result}")
                    
                    if tool_name == "submit_answer":
                        submitted_answer = result["answer"]

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })
            
            # Add tool results to messages
            messages.extend(tool_results)
            
            # If answer submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use
            if verbose:
                print("\nNo tool use, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def main(num_tests: int = 10, verbose: bool = False):
    """
    Run multiple tests concurrently to measure pass rate
    
    Args:
        num_tests: Number of tests to run (default: 10)
        verbose: Show detailed output for each test (default: False)
    """
    print(f"{'=' * 60}")
    print(f"Running {num_tests} test iterations concurrently...")
    print(f"{'=' * 60}\n")
    
    # Define tools (same for all tests)
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "read_csv",
                "description": "Read a CSV file and get preview with statistics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Name of the CSV file to read"}
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_csv",
                "description": "Write CSV content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Name of the file to write"},
                        "csv_content": {"type": "string", "description": "CSV content as a string"}
                    },
                    "required": ["filename", "csv_content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "python_expression",
                "description": "Execute Python code. Pandas is available as 'pd'. Use print() to output results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_file_info",
                "description": "Get information about a file (existence, row count, columns)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Name of the file"}
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit the final cleaning summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "object", "description": "Summary with number of rows removed and cleaning actions taken"}
                    },
                    "required": ["summary"]
                }
            }
        }
    ]

    tool_handlers = {
        "read_csv": read_csv_tool,
        "write_csv": write_csv_tool,
        "python_expression": python_expression_tool,
        "get_file_info": get_file_info_tool,
        "submit_answer": submit_answer_tool,
    }

    # The prompt - NO EXAMPLE CODE to make it harder
    prompt = """Clean the messy dataset 'experiments.csv' and save the cleaned version to 'cleaned_experiments.csv'.

Required cleaning steps:
1. Remove duplicate experiment_id rows (keep first occurrence)
2. Remove rows with missing accuracy or training_time values
3. Remove rows where accuracy is not in valid range [0, 1]
4. Remove rows where training_time is zero or negative

Use the python_expression tool to do all cleaning in pandas, then save the result.
Finally, call submit_answer with a summary of what you did.

Start working now."""

    # Async function to run a single test
    async def run_one_test(test_id: int):
        # Clean up any existing cleaned file
        if os.path.exists("cleaned_experiments.csv"):
            os.remove("cleaned_experiments.csv")
        
        # Generate fresh dataset with different seed
        generate_messy_dataset(seed=42 + test_id)
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"TEST {test_id}")
            print(f"{'=' * 60}")
        
        # Run the agent loop
        try:
            result = await run_agent_loop(
                prompt=prompt,
                tools=tools,
                tool_handlers=tool_handlers,
                max_steps=15,
                model="llama-3.1-8b-instant",  # Using smaller model for 10-40% pass rate
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"Error during agent loop: {e}")
            result = None

        # Verify the cleaning
        passed, verification_results = verify_data_cleaning()

        if not verbose:
            status = "PASS" if passed else "FAIL"
            print(f"Test {test_id}: {status} (Score: {verification_results['score']:.0%})")
        else:
            print(f"\n{'=' * 60}")
            if passed:
                print(f"✅ SUCCESS - All checks passed (Score: {verification_results['score']:.0%})")
            else:
                failed_checks = [k for k, v in verification_results['checks'].items() if not v]
                print(f"❌ FAILURE - Failed checks: {', '.join(failed_checks)}")
                print(f"Details: {verification_results['details']}")
            print(f"{'=' * 60}")
        
        return test_id, passed, verification_results
    
    # Create all test tasks and run concurrently
    tasks = [run_one_test(i + 1) for i in range(num_tests)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Calculate metrics
    successful_results = [r for r in results if not isinstance(r, Exception)]
    successes = sum(1 for _, passed, _ in successful_results if passed)
    failures = len(successful_results) - successes
    pass_rate = (successes / len(successful_results)) * 100 if successful_results else 0
    
    # Show summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS:")
    print(f"{'=' * 60}")
    print(f"  Total Tests: {len(successful_results)}")
    print(f"  Passed: {successes}")
    print(f"  Failed: {failures}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")
    
 
    # Show failure analysis
    if failures > 0:
        print(f"\n{'=' * 60}")
        print("FAILURE ANALYSIS:")
        print(f"{'=' * 60}")
        all_failed_checks = {}
        for _, passed, verification in successful_results:
            if not passed:
                for check, check_passed in verification['checks'].items():
                    if not check_passed:
                        all_failed_checks[check] = all_failed_checks.get(check, 0) + 1
        
        print("Most common failed checks:")
        for check, count in sorted(all_failed_checks.items(), key=lambda x: x[1], reverse=True):
            pct = (count / failures) * 100
            print(f"  - {check}: {count}/{failures} ({pct:.0f}%)")


if __name__ == "__main__":
    # Run 10 tests concurrently to measure pass rate
    asyncio.run(main(num_tests=10, verbose=False))
