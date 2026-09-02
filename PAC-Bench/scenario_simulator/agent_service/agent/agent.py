from dotenv import load_dotenv
import os
import json

from tool.mcp_client import MCPClient
from tool.mcp_agent import MCPAgent
from utils.model_factory import ModelFactory

load_dotenv()

class Agent:
    def __init__(
        self,
        agent_name: str,
        agent_context: dict,
        model,
        mcp_client: MCPClient,
        tools: list,
        prompt: str,
        agent_args: dict,
        mcp_agent: MCPAgent,
        _internal_init: bool = False,
    ):
        if not _internal_init:
            raise RuntimeError("Direct instantiation 'Agent(...)' is not allowed. Please use 'await Agent.create(...)' instead.")
        
        self.agent_name = agent_name
        self.agent_context = agent_context
        self.model = model
        self.mcp_client = mcp_client
        self.tools = tools
        self.prompt = prompt
        self.agent_args = agent_args
        self.mcp_agent = mcp_agent

    @classmethod
    async def create(
        cls,
        agent_name: str,
        *,
        model_name: str,
        agent_context: dict = {},
        model_args: dict = {},
        agent_args: dict = {},
        prompt_path: str = None,
        server_config_path: str = None,
        tool_ban_list_path: str = None,
    ):
        if agent_name not in ("agent_a", "agent_b"):
            raise ValueError("agent_name must be either 'agent_a' or 'agent_b'")

        prompt = cls._load_prompt(agent_name, agent_context, prompt_path)
        server_config_path = server_config_path or os.getenv("MCP_SERVER_CONFIG")
        tool_ban_list_path = tool_ban_list_path or os.getenv("MCP_TOOL_BAN_LIST")

        model = ModelFactory.create_model(model_name=model_name, **model_args)
        mcp_client = MCPClient(server_config_path, tool_ban_list_path)
        tools = await mcp_client.get_tools()

        mcp_agent = MCPAgent(
            model=model,
            tools=tools,
            system_prompt=prompt,
            **agent_args,
        )

        return cls(
            agent_name=agent_name,
            agent_context=agent_context,
            model=model,
            mcp_client=mcp_client,
            tools=tools,
            prompt=prompt,
            agent_args=agent_args,
            mcp_agent=mcp_agent,
            _internal_init=True,
        )
    
    @staticmethod
    def _load_prompt(agent_name: str, agent_context: dict, path: str | None) -> str:
        if path is None:
            path = os.getenv("PROMPT_DIR") + f"/{agent_name}.txt"
        
        with open(path, "r", encoding="utf-8") as file:
            prompt = file.read()
        
        if agent_context:
            prompt = prompt.format(**agent_context)

        print(prompt)
        return prompt
    
    def get_conversation_history(self) -> list:
        return self.mcp_agent.get_messages()
    
    def get_last_turn_conversation(self) -> list:
        return self.mcp_agent.get_last_turn_messages()
    
    def get_middleware_messages(self) -> list:
        return self.mcp_agent.middleware_messages
    
    def get_chunks(self) -> list:
        return self.mcp_agent.chunks

    def get_token_usage(self):
        history = self.get_conversation_history()

        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        for message in history:
            usage_metadata = getattr(message, "usage_metadata", {})

            input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
            output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
            total_tokens = usage_metadata.get("total_tokens") or input_tokens + output_tokens
            
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["total_tokens"] += total_tokens

        return totals

    async def run(self, query: str) -> str:
        response = await self.mcp_agent.run(query)
        return response

