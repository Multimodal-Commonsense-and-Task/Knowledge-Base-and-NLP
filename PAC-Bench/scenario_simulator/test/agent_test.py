from simulation.agent_client import AgentClient

BASE_URL = "http://localhost:8100"

def single_request_test():
    client = AgentClient(base_url=BASE_URL)
    
    client.initialize_agent(
        agent_context={},
        model_name="gpt-5-mini",
        prompt_path="prompt/agent_a_test.txt",
    )

    response = client.run_sync(
        "Use the excel tool to create /workspace/shared/test_excel.xlsx and write 'Hello, World!' in cell A1"
    )

    print("\n\nConversation history:", client.get_conversation_history())
    # print("\n\nTool usage:", client.get_tool_usage_history())
    # print("\n\nToken usage:", client.get_token_usage())
    print("\n\nAgent response:", response)


if __name__ == "__main__":
    single_request_test()