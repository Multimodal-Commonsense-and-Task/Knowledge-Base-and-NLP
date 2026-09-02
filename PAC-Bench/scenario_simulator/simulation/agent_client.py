from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AgentClientConfig:
    base_url: str
    timeout_s: float = 60.0


class AgentClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        url: str | None = None,
        endpoint: str | None = None,
        error_type: str | None = None,
        trace: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.url = url
        self.endpoint = endpoint
        self.error_type = error_type
        self.trace = trace


class AgentClient:
    """HTTP client for the agent_service FastAPI app.

    Endpoints are defined in agent_service/app/main.py.
    """

    def __init__(self, base_url: str, timeout_s: float = 300.0):
        self._config = AgentClientConfig(
            base_url=base_url.rstrip("/"),
            timeout_s=timeout_s,
        )

    def _make_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._config.base_url}{path}"

    def _request_json(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any | None = None,
    ) -> Any:
        url = self._make_url(path)
        if params:
            encoded = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None},
                doseq=True,
            )
            if encoded:
                url = f"{url}?{encoded}"

        data: bytes | None
        headers = {
            "Accept": "application/json",
        }

        if body is None:
            data = None
        else:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url=url, data=data, method=method, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            raw = (e.read() or b"").decode("utf-8", errors="replace")
            # Best-effort: preserve structured error fields if server returned JSON.
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict) and parsed.get("error"):
                raise AgentClientError(
                    AgentClient._format_server_error_message(parsed),
                    status_code=int(e.code),
                    response_body=raw,
                    url=url,
                    endpoint=str(parsed.get("endpoint") or "" or path),
                    error_type=str(parsed.get("error_type") or ""),
                    trace=str(parsed.get("trace") or ""),
                ) from e

            raise AgentClientError(
                f"Agent service HTTP error: {e.code}",
                status_code=int(e.code),
                response_body=raw,
                url=url,
            ) from e
        except urllib.error.URLError as e:
            raise AgentClientError(
                f"Agent service connection error: {e}",
                url=url,
            ) from e
        except socket.timeout as e:
            raise AgentClientError(
                "Agent service request timed out",
                url=url,
            ) from e
        except json.JSONDecodeError as e:
            raise AgentClientError(
                "Agent service returned non-JSON response",
                url=url,
            ) from e

    @staticmethod
    def _format_server_error_message(payload: dict[str, Any]) -> str:
        endpoint = payload.get("endpoint")
        error_type = payload.get("error_type")
        error = payload.get("error")
        trace = payload.get("trace")

        header_parts = [p for p in [endpoint, error_type] if p]
        header = " / ".join(str(p) for p in header_parts)
        if header:
            msg = f"{header}: {error}"
        else:
            msg = str(error)

        if trace:
            msg = f"{msg}\n\n{trace}"

        return msg

    @staticmethod
    def _raise_if_error(payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("error"):
            raise AgentClientError(
                AgentClient._format_server_error_message(payload),
                endpoint=str(payload.get("endpoint") or ""),
                error_type=str(payload.get("error_type") or ""),
                trace=str(payload.get("trace") or ""),
                response_body=json.dumps(payload, ensure_ascii=False),
            )

    # --------- Endpoints ---------

    def health(self) -> dict:
        payload = self._request_json("GET", "/health")
        if not isinstance(payload, dict):
            raise AgentClientError("Unexpected /health response")
        return payload

    def initialize_agent(
        self,
        *,
        agent_context: dict,
        model_name: str,
        model_args: dict | None = None,
        agent_args: dict | None = None,
        prompt_path: str | None = None,
        server_config_path: str | None = None,
    ) -> dict:
        payload = self._request_json(
            "POST",
            "/agent/initialize",
            body={
                "model_name": model_name,
                "prompt_path": prompt_path,
                "server_config_path": server_config_path,
                "agent_context": agent_context,
                "model_args": model_args or {},
                "agent_args": agent_args or {},
            },
        )
        self._raise_if_error(payload)
        if not isinstance(payload, dict):
            raise AgentClientError("Unexpected /agent/initialize response")
        return payload

    def run_sync(self, query: str) -> str:
        # FastAPI signature: run_agent(query: str)
        # query is treated as a query parameter unless the server uses Body(...).
        payload = self._request_json("POST", "/agent/run", body={"query": query})
        self._raise_if_error(payload)
        if not isinstance(payload, dict) or "response" not in payload:
            raise AgentClientError("Unexpected /agent/run response")
        return str(payload["response"])

    async def run(self, query: str) -> str:
        return await asyncio.to_thread(self.run_sync, query)

    def get_conversation_history(self) -> list:
        payload = self._request_json("GET", "/agent/conversation_history")
        if isinstance(payload, dict) and "error" in payload:
            self._raise_if_error(payload)
        if not isinstance(payload, list):
            raise AgentClientError("Unexpected /agent/conversation_history response")
        return payload

    def get_last_turn_conversation(self) -> list:
        payload = self._request_json("GET", "/agent/last_turn_conversation")
        if isinstance(payload, dict) and "error" in payload:
            self._raise_if_error(payload)
        if not isinstance(payload, list):
            raise AgentClientError("Unexpected /agent/last_turn_conversation response")
        return payload

    def get_token_usage(self) -> dict:
        payload = self._request_json("GET", "/agent/token_usage")
        self._raise_if_error(payload)
        if not isinstance(payload, dict):
            raise AgentClientError("Unexpected /agent/token_usage response")
        return payload

    def get_prompt(self) -> str:
        payload = self._request_json("GET", "/agent/prompt")
        self._raise_if_error(payload)
        return "" if payload is None else str(payload)

    def get_chunks(self) -> list:
        payload = self._request_json("GET", "/agent/chunks")
        if isinstance(payload, dict) and "error" in payload:
            self._raise_if_error(payload)
        if not isinstance(payload, list):
            raise AgentClientError("Unexpected /agent/chunks response")
        return payload

    def test_tool_list(self, server_name: str | None = None) -> Any:
        payload = self._request_json(
            "GET",
            "/test/tool_list",
            params={"server_name": server_name},
        )
        self._raise_if_error(payload)
        return payload

    def test_call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> Any:
        payload = self._request_json(
            "POST",
            "/test/call_tool",
            body={
                "arguments": arguments,
                "tool_name": tool_name,
            },
        )
        self._raise_if_error(payload)
        return payload

if __name__ == "__main__":
    client = AgentClient(base_url="http://localhost:8200")

    print("Health:", client.health())
    print("Initializing agent...")
    init_response = client.initialize_agent(
        agent_context={},
        model_name="gpt-5.2",
        agent_args={"max_tool_calls": 5},
        prompt_path="prompt/agent_a_test.txt",
    )
    print("Init response:", init_response)