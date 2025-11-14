"""
Paper Review RL Task for LLM Training
Task: Review a scientific paper and identify flaws
"""
import asyncio
import json
import os
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from generate_paper import generate_flawed_paper, save_paper_to_file

MAX_TOKENS = 2048


class ToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitReviewToolResult(TypedDict):
    review: dict
    submitted: bool


def read_paper_tool(filename: str) -> ToolResult:
    """Read the paper JSON file"""
    try:
        if not os.path.exists(filename):
            return {"result": None, "error": f"File {filename} not found"}
        
        with open(filename, 'r') as f:
            paper = json.load(f)
        
        return {"result": paper, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def python_expression_tool(expression: str) -> ToolResult:
    """Execute Python code for analysis"""
    try:
        namespace = {"json": json}
        stdout = StringIO()
        
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_review_tool(review: dict) -> SubmitReviewToolResult:
    """Submit the final review"""
    return {"review": review, "submitted": True}


def verify_review(paper_flaws: dict, submitted_review: dict | None) -> tuple[bool, dict[str, Any]]:
    """
    Verify the quality of the paper review
    
    A good review should identify:
    1. Multiple specific flaws (at least 3 out of 5 categories)
    2. Provide constructive feedback
    3. Have clear structure
    4. Rate the paper appropriately
    5. Suggest improvements
    
    This targets 10-40% pass rate
    """
    results = {
        "passed": False,
        "checks": {},
        "details": {},
        "score": 0
    }
    
    if submitted_review is None:
        results["checks"]["review_submitted"] = False
        results["details"]["error"] = "No review submitted"
        return False, results
    
    results["checks"]["review_submitted"] = True
    
    try:
        # Get actual flaws from paper
        actual_flaws = set()
        for category, flaw in paper_flaws.items():
            if flaw:
                actual_flaws.add(category)
        
        # Check 1: Review structure
        required_fields = ["identified_flaws", "overall_assessment", "recommendation"]
        has_structure = all(field in submitted_review for field in required_fields)
        results["checks"]["has_proper_structure"] = has_structure
        results["details"]["missing_fields"] = [f for f in required_fields if f not in submitted_review]
        
        # Check 2: Identified flaws
        identified = submitted_review.get("identified_flaws", [])
        if isinstance(identified, str):
            identified = [identified]
        
        num_identified = len(identified) if isinstance(identified, list) else 0
        results["checks"]["identified_multiple_flaws"] = num_identified >= 3
        results["details"]["flaws_identified_count"] = num_identified
        results["details"]["actual_flaws_count"] = len(actual_flaws)
        
        # Check 3: Quality of critique (check for keywords)
        review_text = json.dumps(submitted_review).lower()
        quality_keywords = ["methodology", "dataset", "statistics", "baseline", "reproducib"]
        quality_count = sum(1 for kw in quality_keywords if kw in review_text)
        results["checks"]["detailed_critique"] = quality_count >= 3
        results["details"]["quality_keywords_found"] = quality_count
        
        # Check 4: Appropriate recommendation
        recommendation = submitted_review.get("recommendation", "").lower()
        valid_recommendations = ["reject", "major revision", "minor revision", "accept"]
        has_recommendation = any(rec in recommendation for rec in valid_recommendations)
        results["checks"]["has_clear_recommendation"] = has_recommendation
        results["details"]["recommendation"] = recommendation
        
        # Check 5: Constructive feedback
        has_feedback = "suggestions" in submitted_review or "improvements" in submitted_review
        feedback_length = len(str(submitted_review.get("suggestions", ""))) + len(str(submitted_review.get("improvements", "")))
        results["checks"]["provides_constructive_feedback"] = has_feedback and feedback_length > 50
        results["details"]["feedback_length"] = feedback_length
        
        # Calculate score
        checks_passed = sum(1 for v in results["checks"].values() if v)
        total_checks = len(results["checks"])
        results["score"] = checks_passed / total_checks if total_checks > 0 else 0
        
        # Overall pass: At least 4 out of 5 checks
        results["passed"] = checks_passed >= 4
        
        return results["passed"], results
        
    except Exception as e:
        results["checks"]["error_occurred"] = False
        results["details"]["error"] = str(e)
        return False, results


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 12,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """Run agent loop for paper review"""
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "max_tokens":
            if verbose:
                print(f"Model reached max_tokens limit {MAX_TOKENS}")

        has_tool_use = False
        tool_results = []
        submitted_review = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text[:200]}...")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"\nUsing tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "submit_review":
                        result = handler(tool_input.get("review", tool_input))
                        submitted_review = result["review"]
                    elif tool_name == "python_expression":
                        result = handler(tool_input["expression"])
                    else:
                        result = handler(**tool_input) if isinstance(tool_input, dict) else handler(tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result),
                    })

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_review is not None:
                if verbose:
                    print(f"\nReview submitted!")
                return submitted_review
        else:
            if verbose:
                print("\nNo tool use, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting review.")
    return None


async def main(num_tests: int = 10, verbose: bool = False):
    """Run paper review tests"""
    print(f"{'=' * 60}")
    print(f"Running {num_tests} paper review test iterations concurrently...")
    print(f"{'=' * 60}\n")
    
    # Define tools
    tools: list[ToolUnionParam] = [
        {
            "name": "read_paper",
            "description": "Read the scientific paper to review (JSON format)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Paper filename"}
                },
                "required": ["filename"]
            }
        },
        {
            "name": "python_expression",
            "description": "Execute Python code for analysis",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["expression"]
            }
        },
        {
            "name": "submit_review",
            "description": "Submit the final paper review",
            "input_schema": {
                "type": "object",
                "properties": {
                    "review": {
                        "type": "object",
                        "description": "Complete review with identified_flaws, overall_assessment, recommendation, and suggestions"
                    }
                },
                "required": ["review"]
            }
        }
    ]

    tool_handlers = {
        "read_paper": read_paper_tool,
        "python_expression": python_expression_tool,
        "submit_review": submit_review_tool,
    }

    prompt = """You are a peer reviewer for a top-tier AI/ML conference. Review the paper in 'paper_to_review.json'.

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

Be thorough and critical. Use the read_paper tool, analyze carefully, then submit_review.

Start your review now."""

    async def run_one_test(test_id: int):
        # Generate unique paper
        paper_data = generate_flawed_paper(seed=42 + test_id)
        save_paper_to_file(paper_data, "paper_to_review.json")
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"TEST {test_id}")
            print(f"{'=' * 60}")
        
        try:
            result = await run_agent_loop(
                prompt=prompt,
                tools=tools,
                tool_handlers=tool_handlers,
                max_steps=12,
                model="claude-haiku-4-5",
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"Error during review: {e}")
            result = None

        # Verify review
        passed, verification_results = verify_review(
            paper_data["flaw_categories"],
            result
        )

        if not verbose:
            status = "PASS" if passed else "FAIL"
            print(f"Test {test_id}: {status} (Score: {verification_results['score']:.0%})")
        else:
            print(f"\n{'=' * 60}")
            if passed:
                print(f"✅ EXCELLENT REVIEW (Score: {verification_results['score']:.0%})")
            else:
                failed_checks = [k for k, v in verification_results['checks'].items() if not v]
                print(f"❌ INSUFFICIENT REVIEW - Failed checks: {', '.join(failed_checks)}")
                print(f"Details: {verification_results['details']}")
            print(f"{'=' * 60}")
        
        return test_id, passed, verification_results
    
    # Run all tests concurrently
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


if __name__ == "__main__":
    asyncio.run(main(num_tests=10, verbose=False))
