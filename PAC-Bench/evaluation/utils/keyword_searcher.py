from typing import List, Dict, Any
import re

class PrivacyKeywordSearcher:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords

    def search(
        self,
        conversation_turn: str,
    ) -> Dict[str, Any]:
        return self._search_response(conversation_turn)

    def result_to_str(self, result: Dict[str, Any]) -> str:
        content = result.get("content", "")
        keywords = result.get("keywords", [])

        if not keywords:
            return ""

        return self._format_item("response", content, keywords)

    def _format_item(self, kind: str, content: str, keywords: List[str]) -> str:
        kw_line = ", ".join(keywords) if keywords else "(none)"
        return "\n".join(
            [
                f"===== {kind.upper()} BEGIN =====",
                "<<<<SECTION>>>>",
                content,
                "<<<<END SECTION>>>>",
                "<<<<KEYWORDS>>>>",
                kw_line,
                "<<<<END KEYWORDS>>>>",
                f"===== {kind.upper()} END =====",
            ]
        )

    def _search_response(self, message: str) -> Dict[str, Any]:
        matched_keywords = self._find_keywords(message)
        return {
            "content": message,
            "keywords": matched_keywords,
        }

    def _find_keywords(self, text: str) -> List[str]:
        found = set()

        for kw in self.keywords:
            if re.search(re.escape(kw), text, flags=re.IGNORECASE):
                found.add(kw)

        return sorted(found)