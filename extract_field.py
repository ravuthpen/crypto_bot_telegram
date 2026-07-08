import re


class AnalysisParser:
    """Parses the AI analysis text. Construct once over a blob, then pull
    fields/blocks; or use the AnalysisParser.field()/block() classmethods
    for one-off extraction."""

    def __init__(self, text: str):
        self.text = text or ""

    def field(self, key: str, default: str = "—") -> str:
        m = re.search(rf"\*\*{re.escape(key)}[:\*]*\*?\*?\s*(.+)", self.text)
        return m.group(1).strip().strip("*").strip() if m else default

    def block(self, key: str) -> list[str]:
        lines, capturing = [], False
        for line in self.text.splitlines():
            if re.search(rf"\*\*{re.escape(key)}", line, re.IGNORECASE):
                capturing = True
                continue
            if capturing:
                if line.strip().startswith("**") and ":" in line:
                    break
                if line.strip().startswith("-"):
                    lines.append(line.strip().lstrip("- ").strip())
                elif line.strip() and not line.strip().startswith("**"):
                    lines.append(line.strip())
        return [l for l in lines if l]

    @classmethod
    def field_of(cls, text: str, key: str, default: str = "—") -> str:
        return cls(text).field(key, default)

    @classmethod
    def block_of(cls, text: str, key: str) -> list[str]:
        return cls(text).block(key)


# Backwards-compatible shims so existing call sites keep working.
def extract_field(text: str, key: str, default: str = "—") -> str:
    return AnalysisParser.field_of(text, key, default)


def extract_block(text: str, key: str) -> list[str]:
    return AnalysisParser.block_of(text, key)