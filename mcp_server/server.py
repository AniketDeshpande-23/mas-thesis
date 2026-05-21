from .mcp_instance import mcp

# just import modules so decorators register tools
from . import dataset_server
from . import eval_server

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)