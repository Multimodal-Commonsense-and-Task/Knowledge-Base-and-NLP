from simulation.agent_client import AgentClient

BASE_URL = "http://localhost:8100"

TOOL_NAME = "excel_describe_sheets"
ARGUMENTS = {
    "fileAbsolutePath": "/workspace/shared/test_excel.xlsx",
}

def tool_list_test():
    client = AgentClient(base_url=BASE_URL)
    tool_names = client.test_tool_list()
    print("Available tools:")
    for idx, name in enumerate(tool_names):
        print(f"{idx + 1}. {name}")
    print()

def call_tool_test():
    client = AgentClient(base_url=BASE_URL)
    result = client.test_call_tool(
        tool_name=TOOL_NAME,
        arguments=ARGUMENTS,
    )
    print("Tool call result:", result)
    print()

if __name__ == "__main__":
    tool_list_test()
    # call_tool_test()
