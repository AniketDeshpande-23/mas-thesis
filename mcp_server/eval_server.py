# from .mcp_instance import mcp  # re-enable when run_tests is active

# NOTE: run_tests is disabled — evaluate_patch() does not yet exist in
# trulens_evaluator.py. Enable once Docker-based eval is implemented.
#
# @mcp.tool()
# def run_tests(patch: str, instance_id: str) -> dict:
#     from evaluation.trulens_evaluator import evaluate_patch
#     result = evaluate_patch(patch, instance_id)
#     return {
#         "resolved": result.resolved,
#         "fail_to_pass": result.fail_to_pass_passed,
#         "pass_to_pass": result.pass_to_pass_passed,
#         "error": result.error_message,
#     }