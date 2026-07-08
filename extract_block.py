import re


class BlockExtractor:
    """Pulls the bulleted/indented body under a **KEY** heading out of the
    analysis text. Construct once over a blob, then extract many blocks;
    or use BlockExtractor.of() for a one-off call."""

    def __init__(self, text: str):
        self.text = text or ""

    def __call__(self, key: str) -> list[str]:
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
    def of(cls, text: str, key: str) -> list[str]:
        return cls(text)(key)


# Backwards-compatible shim so existing call sites keep working.
def extract_block(text: str, key: str) -> list[str]:
    return BlockExtractor.of(text, key)