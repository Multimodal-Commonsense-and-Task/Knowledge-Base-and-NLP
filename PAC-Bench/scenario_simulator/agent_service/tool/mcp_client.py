import os
import json
import re
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

class MCPClient: 
    def __init__(
        self,
        config_path: str,
        tool_ban_list_path: str | None = None,
    ):
        self.config_path = config_path
        self.tool_ban_list_path = tool_ban_list_path

        self.tool_ban_list = self._load_tool_ban_list(self.tool_ban_list_path)
        self.server_config = self._load_server_config(self.config_path)

        self.client = MultiServerMCPClient(self.server_config)
        
    def _resolve_placeholders(self, value, pattern):
            if isinstance(value, dict):
                return {key: self._resolve_placeholders(val, pattern) for key, val in value.items()}
            if isinstance(value, list):
                return [self._resolve_placeholders(item, pattern) for item in value]
            if isinstance(value, str):
                def replace(match):
                    env_key = match.group(1)
                    env_value = os.getenv(env_key)
                    if env_value is None:
                        raise ValueError(
                            f"Missing environment variable '{env_key}' required for MCP server config"
                        )
                    return env_value

                return pattern.sub(replace, value)
            return value
    
    def _load_tool_ban_list(self, path: str | None) -> list:
        if path is None:
            return []
        
        with open(path, "r", encoding="utf-8") as file:
            ban_data = json.load(file)
            
        flattened = []
        for s_name, tools in ban_data.items():
            for t_name in tools:
                flattened.append(f"{s_name}_{t_name}")
        return flattened

    def _load_server_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)

        placeholder_pattern = re.compile(r"{([^}]+)}")

        return self._resolve_placeholders(config, placeholder_pattern)

    async def get_tools(self, server_name: str | None = None) -> list:
        tools_total = []
        
        target_servers = [server_name] if server_name else self.server_config.keys()

        for s_name in target_servers:
            if s_name not in self.server_config:
                continue
                
            tools = await self.client.get_tools(server_name=s_name)
            for tool in tools:
                if not tool.name.lower().startswith(s_name.lower()):
                    tool.name = f"{s_name}_{tool.name}"
                
                if tool.name in self.tool_ban_list:
                    continue
                    
                tools_total.extend([tool])
            
        return tools_total