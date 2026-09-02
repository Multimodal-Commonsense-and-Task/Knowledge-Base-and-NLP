from simulation.agent_client import AgentClient
from simulation.history import ResponseHistory, ToolUsageHistory
import re

from utils.workspace_snapshotter import WorkspaceSnapshotter


class Simulator:
    def __init__(
        self,
        client_a: AgentClient,
        client_b: AgentClient,
        max_turns: int = 20,
        workspace_snapshotter: WorkspaceSnapshotter | None = None,
    ):
        self.client_a = client_a
        self.client_b = client_b

        self.response_history = ResponseHistory()
        self.tool_usage_a = ToolUsageHistory()
        self.tool_usage_b = ToolUsageHistory()

        self.max_turns = max_turns
        self.current_turn = 1
        self.finished = False

        self.workspace_snapshotter = workspace_snapshotter

        self.milestone_completion = {
            "agent_a": [],
            "agent_b": [],
        }

        self.results = []

    def _check_finished(self):
        history = self.response_history.get_history()

        if len(history) < 2:
            self.finished = False
            return

        last_two = history[-2:]
        self.finished = all("[fin]" in msg["message"].lower() for msg in last_two)
    
    def _extract_milestone_completion(self, message: str):
        """
        Extracts milestone numbers from a string.
        Pattern: [MILE: <num>] (Case-insensitive, flexible spacing)
        """
        pattern = r"\[[Mm][Ii][Ll][Ee]:\s*(\d+)\s*\]"
        
        # Find all matches and convert the captured groups to integers
        matches = re.findall(pattern, message)
        
        return [int(num) for num in matches]
    
    def get_response_history(self) -> list:
        return self.response_history.get_history()
    
    def get_tool_usage_history(self) -> dict:
        return {
            "agent_a": self.tool_usage_a.get_tool_usage_history(),
            "agent_b": self.tool_usage_b.get_tool_usage_history(),
        }
    
    def get_middleware_messages(self) -> dict:
        return {
            "agent_a": self.client_a.get_middleware_messages(),
            "agent_b": self.client_b.get_middleware_messages(),
        }

    def get_milestone_completion(self) -> dict:
        return self.milestone_completion

    def get_results(self) -> list:
        return self.results

    def get_chunks(self) -> dict:
        return {
            "agent_a": self.client_a.get_chunks(),
            "agent_b": self.client_b.get_chunks(),
        }
    
    def initialize_agent(
        self,
        agent_name: str,
        agent_context: dict,
        model_name: str,
        model_args: dict = {},
        agent_args: dict = {},
        prompt_path: str = None,
        server_config_path: str = None,
    ):
        if agent_name not in ("agent_a", "agent_b"):
            raise ValueError("agent_name must be either 'agent_a' or 'agent_b'")
        
        client = self.client_a if agent_name == "agent_a" else self.client_b

        client.initialize_agent(
            agent_context=agent_context,
            model_name=model_name,
            model_args=model_args,
            agent_args=agent_args,
            prompt_path=prompt_path,
            server_config_path=server_config_path,
        )

    def step(self):
        if self.current_turn > self.max_turns:
            return {
                "agent": None,
                "response": "Max turns reached. Simulation ended.",
                "tool_usage": [],
            }
        if self.finished:
            return {
                "agent": None,
                "response": "Simulation has already finished.",
                "tool_usage": [],
            }

        turn_num = self.current_turn

        history = self.response_history.get_history()
        last_history = history[-1] if history else {}

        is_last_agent_a = last_history.get("agent", "") == "agent_a"
        current_agent_name = "agent_b" if is_last_agent_a else "agent_a"
        current_client = self.client_b if is_last_agent_a else self.client_a
        tool_usage = self.tool_usage_b if is_last_agent_a else self.tool_usage_a
        tool_usage_len = len(tool_usage.tool_usages)
    
        query = last_history.get("message", "Now, start the collaboration.")
        try:
            response = current_client.run_sync(query)
            completed_milestones = self._extract_milestone_completion(response)

            for milestone_num in completed_milestones:
                self.milestone_completion[current_agent_name].append({
                    "num": milestone_num,
                    "turn": self.current_turn,
                })

            self.response_history.append(response, agent_name=current_agent_name)
            tool_usage.append_from_conversation_history(
                current_client.get_last_turn_conversation()
            )

            self._check_finished()

            self.current_turn += 1

            return {
                "agent": current_agent_name,
                "response": response,
                "tool_usage": tool_usage.get_tool_usage_history()[tool_usage_len:],
            }

        except Exception as e:
            response = f"Error during agent run: {str(e)}"

            self.response_history.append(response, agent_name=current_agent_name)
            tool_usage.append_from_conversation_history(
                current_client.get_last_turn_conversation()
            )

            raise e

        finally:
            if self.workspace_snapshotter is not None:
                try:
                    self.workspace_snapshotter.snapshot(turn_num)
                except Exception:
                    # Never let snapshot failures override the simulation outcome.
                    pass

    def run(self):
        if self.current_turn > 1:
            raise RuntimeError(
                "Simulation has already started by calling step(). Use step() for further steps."
            )

        self.results = []

        while self.current_turn <= self.max_turns:
            if self.finished:
                break
            step_result = self.step()
            self.results.append(step_result)

        return self.results
        

        


