from fastapi import FastAPI
import os
import traceback
from typing import Any

from agent.agent import Agent
from test.tool_use_test import get_tool_list, call_tool

app = FastAPI()
agent = None


def _error_response(endpoint: str, exc: Exception, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "endpoint": endpoint,
        "trace": traceback.format_exc(),
    }

    if extra:
        payload.update(extra)

    return payload

@app.get("/health")
def health():
    try:
        return {"status": "ok"}
    except Exception as e:
        return _error_response("/health", e)

@app.post("/agent/initialize")
async def initialize_agent(body: dict):
    global agent

    agent_context = body.get("agent_context", {})
    model_name = body.get("model_name", "gpt-5.2")
    model_args = body.get("model_args", {})
    agent_args = body.get("agent_args", {})
    prompt_path = body.get("prompt_path", None)
    server_config_path = body.get("server_config_path", None)
    tool_ban_list_path = body.get("tool_ban_list_path", None)

    try:
        agent_name = os.getenv("AGENT_ID")

        agent = await Agent.create(
            agent_name=agent_name,
            agent_context=agent_context,
            model_name=model_name,
            model_args=model_args,
            agent_args=agent_args,
            prompt_path=prompt_path,
            server_config_path=server_config_path,
            tool_ban_list_path=tool_ban_list_path,
        )

        return {"status": "agent initialized"}
    except Exception as e:
        return _error_response("/agent/initialize", e)

@app.post("/agent/run")
async def run_agent(body: dict):
    # Special handling: treat all failures (including uninitialized agent) as exceptions
    # so we always return a full traceback.
    try:
        if agent is None:
            raise RuntimeError("agent not initialized")

        query = body.get("query", "")
        response = await agent.run(query)
        return {"response": response}
    except Exception as e:
        return _error_response("/agent/run", e)

@app.get("/agent/conversation_history")
def get_conversation_history():
    try:
        if agent is None:
            raise RuntimeError("agent not initialized")
        return agent.get_conversation_history()
    except Exception as e:
        return _error_response("/agent/conversation_history", e)

@app.get("/agent/last_turn_conversation")
def get_last_turn_conversation():
    try:
        if agent is None:
            raise RuntimeError("agent not initialized")
        return agent.get_last_turn_conversation()
    except Exception as e:
        return _error_response("/agent/last_turn_conversation", e)

@app.get("/agent/middleware_messages")
def get_middleware_messages():
    try:
        if agent is None:
            raise RuntimeError("agent not initialized")
        return agent.get_middleware_messages()
    except Exception as e:
        return _error_response("/agent/middleware_messages", e)

@app.get("/agent/prompt")
def get_prompt():
    try:
        if agent is None:
            raise RuntimeError("agent not initialized")
        return agent.prompt
    except Exception as e:
        return _error_response("/agent/prompt", e)

@app.get("/agent/token_usage")
def get_token_usage():
    try:
        if agent is None:
            raise RuntimeError("agent not initialized")
        return agent.get_token_usage()
    except Exception as e:
        return _error_response("/agent/token_usage", e)
    
@app.get("/agent/chunks")
def get_chunks():
    try:
        if agent is None:
            raise RuntimeError("agent not initialized")
        return agent.get_chunks()
    except Exception as e:
        return _error_response("/agent/chunks", e)

@app.get("/test/tool_list")
async def test_tool_list(server_name: str | None = None):
    try:
        return await get_tool_list(server_name)
    except Exception as e:
        return _error_response("/test/tool_list", e)

@app.post("/test/call_tool")
async def test_call_tool(body: dict):
    try:
        tool_name = body.get("tool_name")
        arguments = body.get("arguments", {})
        return await call_tool(tool_name, arguments)
    except Exception as e:
        return _error_response("/test/call_tool", e)