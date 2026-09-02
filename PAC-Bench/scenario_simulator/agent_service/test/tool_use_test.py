import os

from tool.mcp_client import MCPClient

async def get_tool_list(
    server_name: str | None = None,
):
    client = MCPClient(
        config_path=os.getenv("MCP_SERVER_CONFIG"),
        tool_ban_list_path=os.getenv("MCP_TOOL_BAN_LIST"),
    )

    tools = await client.get_tools(server_name=server_name)
    tool_names = [t.name for t in tools]

    return tool_names

async def call_tool(
    tool_name: str,
    arguments: dict,
):
    client = MCPClient(
        config_path=os.getenv("MCP_SERVER_CONFIG"),
        tool_ban_list_path=os.getenv("MCP_TOOL_BAN_LIST"),
    )

    tools = await client.get_tools()
    target_tool = None
    for tool in tools:
        if tool.name == tool_name:
            target_tool = tool
            break
    
    if target_tool is None:
        return f"Tool '{tool_name}' not found."

    result = await target_tool.ainvoke(arguments)

    return result

async def main():
    tool_names = await get_tool_list()
    for idx, name in enumerate(tool_names):
        print(name)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())