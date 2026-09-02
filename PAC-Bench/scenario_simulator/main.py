import argparse
import asyncio
import traceback
from pathlib import Path

from dotenv import load_dotenv

from simulation.agent_client import AgentClient
from simulation.simulator import Simulator
from utils.cleaner import Cleaner
from utils.io_utils import list_json_files
from utils.result_writer import SimulationResultWriter
from utils.scenario_manager import ScenarioManager
from utils.parse_utils import parse_interval, parse_range
from utils.workspace_snapshotter import WorkspaceSnapshotter

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent simulations across scenarios.")
    parser.add_argument("--scenario-dir", required=True, help="Directory containing scenario JSON files.")
    parser.add_argument("--result-dir", required=True, help="Directory where simulation outputs will be written.")
    parser.add_argument("--session-range", required=True, help="Session id range, e.g. '[0, 20)' or '[0, 20]'.")
    parser.add_argument(
        "--port-a-range",
        required=True,
        help="Agent A port range, same format as session range (e.g. '[8100, 8120)').",
    )
    parser.add_argument(
        "--port-b-range",
        required=True,
        help="Agent B port range, same format as session range (e.g. '[8101, 8121)').",
    )
    parser.add_argument("--model-agent-a", required=True, help="Model name for agent_a.")
    parser.add_argument("--model-agent-b", required=True, help="Model name for agent_b.")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum turns per simulation run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum LLM calls for each agent per simulation turn.")
    parser.add_argument("--max-tool-calls", type=int, default=None, help="Maximum tool calls for each agent per simulation turn.")
    return parser.parse_args()


def _validate_arguments(
    scenario_dir: Path,
    session_ids: list[int],
    port_a_list: list[int],
    port_b_list: list[int],
) -> None:
    if not scenario_dir.exists() or not scenario_dir.is_dir():
        raise ValueError(f"Scenario directory does not exist: {scenario_dir}")

    if len(session_ids) != len(port_a_list) or len(session_ids) != len(port_b_list):
        raise ValueError(
            "session-range, port-a-range, port-b-range must produce the same number of values"
        )


def _print_output(run_results: list) -> None:
    errors = [result for result in run_results if isinstance(result, Exception)]
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return

    for scenario_name, output_path, status in run_results:  # type: ignore[misc]
        print(f"{status.upper()} {scenario_name} -> {output_path}")


async def _run_single_scenario(
    scenario_path: Path,
    session_queue: "asyncio.Queue[int]",
    port_by_session: dict[int, tuple[int, int]],
    result_writer: SimulationResultWriter,
    model_agent_a: str,
    model_agent_b: str,
    max_turns: int,
    max_steps: int,
    max_tool_calls: int,
) -> tuple[str, Path]:

    session_id = await session_queue.get()
    cleaner = Cleaner(session_id=session_id)

    port_a, port_b = port_by_session[session_id]

    try:
        cleaner.clean_all()

        results = []
        response_history = []
        tool_usage_history = {"agent_a": [], "agent_b": []}
        token_usage_by_agent = {"agent_a": {}, "agent_b": {}}

        client_a = None
        client_b = None

        try:
            scenario_manager = ScenarioManager(
                scenario_path=str(scenario_path),
                max_turns=max_turns,
                max_tool_calls=max_tool_calls,
            )
            context_a = scenario_manager.get_agent_context("agent_a")
            context_b = scenario_manager.get_agent_context("agent_b")

            client_a = AgentClient(base_url=f"http://localhost:{port_a}")
            client_b = AgentClient(base_url=f"http://localhost:{port_b}")

            scenario_dir = result_writer.base_result_dir / scenario_path.stem
            scenario_dir.mkdir(parents=True, exist_ok=True)
            workspace_snapshotter = WorkspaceSnapshotter(
                session_id=session_id,
                output_dir=scenario_dir,
            )

            simulator = Simulator(
                client_a=client_a,
                client_b=client_b,
                max_turns=max_turns,
                workspace_snapshotter=workspace_snapshotter,
            )

            simulator.initialize_agent(
                agent_name="agent_a",
                agent_context=context_a,
                model_name=model_agent_a,
                model_args={},
                agent_args={"max_steps": max_steps, "max_tool_calls": max_tool_calls},
            )
            simulator.initialize_agent(
                agent_name="agent_b",
                agent_context=context_b,
                model_name=model_agent_b,
                model_args={},
                agent_args={"max_steps": max_steps, "max_tool_calls": max_tool_calls},
            )

            agent_prompts = {
                "agent_a": client_a.get_prompt(),
                "agent_b": client_b.get_prompt(),
            }

            results = simulator.run()
            response_history = simulator.get_response_history()
            tool_usage_history = simulator.get_tool_usage_history()
            milestone_completion = simulator.get_milestone_completion()

            token_usage_by_agent = {
                "agent_a": client_a.get_token_usage(),
                "agent_b": client_b.get_token_usage(),
            }

            scenario_output = result_writer.save(
                scenario_path=scenario_path,
                scenario_name=scenario_path.stem,
                session_id=session_id,
                results=results,
                milestone_completion=milestone_completion,
                response_history=response_history,
                tool_usage_history=tool_usage_history,
                token_usage_by_agent=token_usage_by_agent,
                agent_prompts=agent_prompts,
            )

            return (scenario_path.name, scenario_output, "success")
        except Exception as exc:
            if client_a is not None:
                token_usage_by_agent["agent_a"] = client_a.get_token_usage()
            if client_b is not None:
                token_usage_by_agent["agent_b"] = client_b.get_token_usage()

            error_info = {
                "type": type(exc).__name__,
                "message": str(exc),
                "repr": repr(exc),
                "traceback": traceback.format_exc(),
                "chunks": simulator.get_chunks(),
            }

            response_history = simulator.get_response_history()
            tool_usage_history = simulator.get_tool_usage_history()
            results = simulator.get_results()
            milestone_completion = simulator.get_milestone_completion()

            scenario_output = result_writer.save(
                scenario_path=scenario_path,
                scenario_name=scenario_path.stem,
                session_id=session_id,
                milestone_completion=milestone_completion,
                response_history=response_history,
                tool_usage_history=tool_usage_history,
                token_usage_by_agent=token_usage_by_agent,
                agent_prompts={
                    "agent_a": client_a.get_prompt() if client_a is not None else "",
                    "agent_b": client_b.get_prompt() if client_b is not None else "",
                },
                error=error_info,
                results=results,
            )
            return (scenario_path.name, scenario_output, "error")
    finally:
        cleaner.clean_all()
        session_queue.put_nowait(session_id)

async def _run_all_scenarios(args: argparse.Namespace) -> None:
    scenario_dir = Path(args.scenario_dir)
    session_ids = parse_range(args.session_range)
    port_a_list = parse_interval(args.port_a_range)
    port_b_list = parse_interval(args.port_b_range)
    result_writer = SimulationResultWriter(args.result_dir)

    _validate_arguments(scenario_dir, session_ids, port_a_list, port_b_list)

    port_by_session = {
        session_id: (port_a, port_b)
        for session_id, port_a, port_b in zip(session_ids, port_a_list, port_b_list)
    }

    scenario_files = list_json_files(scenario_dir)
    if not scenario_files:
        raise ValueError(f"No scenario files found in {scenario_dir}.")

    session_queue: asyncio.Queue[int] = asyncio.Queue()
    for session_id in session_ids:
        session_queue.put_nowait(session_id)

    tasks = [
        asyncio.create_task(
            _run_single_scenario(
                scenario_path=scenario_path,
                session_queue=session_queue,
                port_by_session=port_by_session,
                result_writer=result_writer,
                model_agent_a=args.model_agent_a,
                model_agent_b=args.model_agent_b,
                max_turns=args.max_turns,
                max_steps=args.max_steps,
                max_tool_calls=args.max_tool_calls,
            )
        )
        for scenario_path in scenario_files
    ]

    run_results = await asyncio.gather(*tasks, return_exceptions=True)
    _print_output(run_results)


def main() -> None:
    args = parse_args()
    print('hello')
    print(args.model_agent_a, args.model_agent_b, args.max_turns, args.max_steps, args.max_tool_calls)
    asyncio.run(_run_all_scenarios(args))


if __name__ == "__main__":
    main()