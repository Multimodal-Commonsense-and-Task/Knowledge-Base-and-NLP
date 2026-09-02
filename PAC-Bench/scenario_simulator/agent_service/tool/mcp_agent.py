from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import uuid
import re
import json

from utils.model_factory import ModelFactory

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
        self.memory_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        self.messages = []
        self.last_turn_messages = []
        self.middleware_messages = []
        self.chunks = []

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
                    model=ModelFactory.create_model("gpt-5-mini"),
                    trigger=("tokens", self.context_length_limit),
                    keep=("tokens", int(self.context_length_limit * 0.8))
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
            checkpointer=self.memory
        )
        return agent

    def _print_stream_chunk(self, chunk: dict):
        is_printed = False
        if self.print_log:
            try: 
                for key, value in chunk.items():
                    if not isinstance(value, dict):
                        continue

                    messages = value.get("messages", [])
                    if len(messages) > 0:
                        message = messages[-1]
                        content_blocks = getattr(message, "content_blocks", [])
                        if len(content_blocks) > 0:
                            content = content_blocks[0]
                            if not is_printed:
                                print("\n================= Stream Chunk =================\n")
                                is_printed = True
                            print(f"key: {key}\n")
                            print(f"content: {content}\n")
                    else:
                        if not is_printed:
                            print("\n================= Stream Chunk =================\n")
                            is_printed = True
                        print(f"key: {key}\n")
                        print(f"value: {json.dumps(value, indent=2, ensure_ascii=False)}\n")

            except Exception as e:
                print(f"Error printing stream chunk: {e}")

    def get_messages(self) -> list:
        return self.messages
    
    def get_last_turn_messages(self) -> list:
        return self.last_turn_messages

    def _delete_think_chunk(self, message: str) -> str:
        if not isinstance(message, str) or not message:
            return message

        # Remove any <think>...</think> blocks (possibly spanning multiple lines).
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", message, flags=re.IGNORECASE)
        return cleaned.strip()

    async def run(self, query: str) -> str:
        # Clear last turn messages
        self.last_turn_messages = []

        # Prepare input messages
        input_messages = {"messages": [HumanMessage(content=query)]}

        # initialize final content
        final_content = ""

        # Stream the agent steps

        self.chunks.append({
            "type": "start",
            "query": query,
        })

        async for chunk in self.agent.astream(
            input_messages, 
            config=self.memory_config,
            stream_mode="updates"
        ):
            self.chunks.append(chunk)
            self._print_stream_chunk(chunk)

            for node_name, values in chunk.items():
                if not isinstance(values, dict):
                    continue
                
                # Extract new messages
                new_messages = values.get("messages", [])
                if not isinstance(new_messages, list):
                    new_messages = [new_messages]

                for message in new_messages:
                    # Update overall and last turn messages
                    self.messages.append(message)
                    self.last_turn_messages.append(message)

                    # Collect middleware messages
                    if node_name not in ["model", "tools"]:
                        self.middleware_messages.append(message)
                    
                    # Extract agent response from model node
                    if isinstance(message, AIMessage):
                        new_content = str(getattr(message, "content", "")).strip()
                        if len(new_content) > 0:
                            final_content = new_content

        if not final_content:
            raise ValueError("Error occured in agent response: response is empty.")
        
        return self._delete_think_chunk(final_content)
        
