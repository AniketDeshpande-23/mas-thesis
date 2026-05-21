from .mcp_instance import mcp
from data_loaders.swebench_pro import SWEBenchPro

dataset = SWEBenchPro()
dataset.load()

@mcp.tool()
def get_instance(instance_id: str) -> dict:
    inst = dataset.get_instance(instance_id)

    if not inst:
        return {"error": "instance not found"}

    return {
        "repo": inst.repo,
        "problem": inst.problem_statement,
        "files": inst.interface,
        "fail_to_pass": inst.fail_to_pass,
        "pass_to_pass": inst.pass_to_pass,
        "base_commit": inst.base_commit,
    }