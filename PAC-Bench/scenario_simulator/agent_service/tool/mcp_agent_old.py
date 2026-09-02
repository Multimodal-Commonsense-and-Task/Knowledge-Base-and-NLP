from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from copy import deepcopy
import re

load_dotenv()

class MCPAgent:
    def __init__(
        self,
        model: BaseChatModel,
        tools: list,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        max_tool_calls: int | None = None,
        max_tool_retries: int = 2,
        context_length_limit: int | None = 30000,
        print_log: bool = True,
    ):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_tool_retries = max_tool_retries
        self.print_log = print_log
        self.system_prompt = system_prompt
        self.context_length_limit = context_length_limit
        self.memory = InMemorySaver()

        self.messages = []
        self.last_turn_messages = []
        self.middleware_messages = []

        self.agent = self._create_agent()
    
    def _create_middleware(self):
        middleware = []
        if self.max_tool_calls is not None:
            middleware.append(
                ToolCallLimitMiddleware(
                    run_limit=self.max_tool_calls,
                    exit_behavior="continue",
                )
            )
        if self.max_steps is not None:
            middleware.append(
                ModelCallLimitMiddleware(
                    run_limit=self.max_steps,
                    exit_behavior="error",
                )
            )
        if self.max_tool_retries > 0:
            middleware.append(
                ToolRetryMiddleware(
                    max_retries=self.max_tool_retries,
                )
            )
        if self.context_length_limit is not None:
            middleware.append(
                SummarizationMiddleware(
                    model="gpt-5-mini",
                    trigger=("tokens", self.context_length_limit)
                )
            )
        return middleware

    def _create_agent(self):
        middleware = self._create_middleware()
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            middleware=middleware,
            system_prompt=self.system_prompt,
        )
        return agent

    def _print_stream_chunk(self, chunk: dict):
        if self.print_log:
            print("\n================= Stream Chunk =================\n")
            for key, value in chunk.items():
                if not isinstance(value, dict):
                    print(f"key: {key}\n")
                    print(f"value: {value}\n")
                    continue
                messages = value.get("messages", [])
                content = messages[-1].content_blocks[0] if len(messages) > 0 else value

                if isinstance(content, str):
                    content = self._delete_think_chunk(content)

                print(f"key: {key}\n")
                print(f"content: {content}\n")

    def get_messages(self) -> list:
        return self.messages
    
    def get_last_turn_messages(self) -> list:
        return self.last_turn_messages

    def _delete_think_chunk(self, message: str) -> str:
        if not isinstance(message, str) or not message:
            return message

        # Remove any <think>...</think> blocks (possibly spanning multiple lines).
        # Also handles multiple occurrences.
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", message, flags=re.IGNORECASE)
        return cleaned.strip()

    async def run(self, query: str) -> str:
        human_message = HumanMessage(content=query)
        self.messages.append(human_message)
        self.last_turn_messages = [human_message]

        async for chunk in self.agent.astream({"messages": deepcopy(self.messages)}, stream_mode="updates"):
            for key, value in chunk.items():
                if "middleware" in key.lower():
                    if not value:
                        value = "No middleware messages."
                    self.middleware_messages.append(value)
                if isinstance(value, dict):
                    messages = value.get("messages", [])
                    self.messages.extend(messages)
                    self.last_turn_messages.extend(messages)
            self._print_stream_chunk(chunk)

        final_content = self.messages[-1].content
        if isinstance(final_content, str):
            return self._delete_think_chunk(final_content)
        
        
        return final_content
        
