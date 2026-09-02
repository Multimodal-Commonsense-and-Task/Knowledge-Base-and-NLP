import os
import json

from agent.agent import Agent

async def multi_turn_test():
    agent = await Agent.create(
        agent_name="agent_a",
        model_name="qwen3-8b",
        prompt_path="prompt/agent_a_test.txt",
        agent_context={},
        agent_args={
            "max_steps": 10,
            "max_tool_calls": 5,
        },
        model_args={}
    )

    response = await agent.run(
        "show me your available tool list."
    )
    print("\n\nAgent response:", response)

    messages = agent.get_conversation_history()
    messages = [dict(m) for m in messages]

    last_turn = agent.get_last_turn_conversation()
    last_turn = [dict(m) for m in last_turn]

    file_dict = {
        "messages": messages,
        "last_turn": last_turn,
        "token_usage": agent.get_token_usage(),
    }
    with open("test/message1.json", "w", encoding="utf-8") as f:
        json.dump(file_dict, indent=2, ensure_ascii=False, fp=f)
    
    response = await agent.run(
        "1. Briefly tell me what request was made last time. 2. Read the contents of /workspace/shared/sample.txt again and briefly tell me if there are any changes."
    )
    print("\n\nAgent response:", response)

    messages = agent.get_conversation_history()
    messages = [dict(m) for m in messages]

    last_turn = agent.get_last_turn_conversation()
    last_turn = [dict(m) for m in last_turn]

    file_dict = {
        "messages": messages,
        "last_turn": last_turn,
        "token_usage": agent.get_token_usage(),
    }
    with open("test/message2.json", "w", encoding="utf-8") as f:
        json.dump(file_dict, indent=2, ensure_ascii=False, fp=f)

if __name__ == "__main__":
    import asyncio
    asyncio.run(multi_turn_test())