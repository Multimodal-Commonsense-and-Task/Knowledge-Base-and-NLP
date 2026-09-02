from typing import Any
import re

class ResponseHistory:
    def __init__(self):
        self.messages = []
        self.current_turn = 0

    def append(self, message: str, agent_name: str):
        self.current_turn += 1
        self.messages.append(
            {
                "turn": self.current_turn,
                "agent": agent_name,
                "message": message,
            }
        )
    
    def get_history(self) -> list:
        return self.messages

class ToolUsageHistory:
    def __init__(self):
        self.tool_usages = []

    def append(self, tool_name: str, arguments: dict, result: Any):
        self.tool_usages.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

    def append_last_turn_from_conversation_history(self, conversation_history: list):
        if not conversation_history:
            return

        last_human_idx = -1
        for idx in range(len(conversation_history) - 1, -1, -1):
            msg = conversation_history[idx]
            msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
            if msg_type == "human":
                last_human_idx = idx
                break

        self._append_tool_usages_from_messages(conversation_history[last_human_idx + 1 :])

    def append_from_conversation_history(self, conversation_history: list):
        self._append_tool_usages_from_messages(conversation_history)

    def _append_tool_usages_from_messages(self, messages: list):
        pending_calls: dict[str, dict] = {}

        for msg in messages:
            if isinstance(msg, dict):
                msg_type = msg.get("type")
                tool_calls = msg.get("tool_calls")
            else:
                msg_type = getattr(msg, "type", None)
                tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
                for call in tool_calls:
                    if isinstance(call, dict):
                        call_id = call.get("id")
                        tool_name = call.get("name")
                        arguments = call.get("args", {})
                    else:
                        call_id = getattr(call, "id", None)
                        tool_name = getattr(call, "name", None)
                        arguments = getattr(call, "args", {})

                    if call_id and tool_name:
                        pending_calls[str(call_id)] = {
                            "tool_name": tool_name,
                            "arguments": arguments or {},
                        }

            if msg_type == "tool":
                if isinstance(msg, dict):
                    tool_call_id = msg.get("tool_call_id")
                    result_content = msg.get("content", "")
                else:
                    tool_call_id = getattr(msg, "tool_call_id", None)
                    result_content = getattr(msg, "content", "")

                if tool_call_id is None:
                    continue

                call_info = pending_calls.pop(str(tool_call_id), None)
                if not call_info:
                    continue

                result_texts = self._extract_text_contents(result_content)

                self.append(
                    tool_name=call_info["tool_name"],
                    arguments=call_info["arguments"],
                    result=result_texts,
                )

        return self

    @staticmethod
    def _extract_text_contents(tool_content: Any) -> list[str]:
        if tool_content is None:
            return []

        if isinstance(tool_content, list):
            return [str(x) for x in tool_content]

        text = str(tool_content)

        # Expected: "[TextContent(type='text', text='...', annotations=None, meta=None)]"
        # Extract all occurrences of text='...'(or text="...") and return as a list.
        pattern = re.compile(
            r"text=(?:'((?:\\'|[^'])*)'|\"((?:\\\"|[^\"])*)\")"
        )
        matches = pattern.findall(text)
        if not matches:
            return [text] if text != "" else []

        extracted: list[str] = []
        for single_quoted, double_quoted in matches:
            raw = single_quoted or double_quoted
            # Normalize common escaped sequences that appear in repr strings.
            raw = raw.replace("\\n", "\n").replace("\\t", "\t")
            raw = raw.replace("\\'", "'").replace('\\"', '"')
            extracted.append(raw)

        return extracted

    def get_tool_usage_history(self) -> list:
        return self.tool_usages
        