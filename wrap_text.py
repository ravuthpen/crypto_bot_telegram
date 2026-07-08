class TextWrapper:
    """Greedy word-wrap to a max line width. Construct once with a width,
    then wrap many strings; or use TextWrapper.wrap() for a one-off call."""

    def __init__(self, width: int = 46):
        self.width = width

    def __call__(self, text: str) -> list[str]:
        words, buf, out = text.split(), [], []
        for w in words:
            buf.append(w)
            if len(" ".join(buf)) > self.width:
                out.append(" ".join(buf[:-1]))
                buf = [w]
        if buf:
            out.append(" ".join(buf))
        return out

    @classmethod
    def wrap(cls, text: str, width: int = 46) -> list[str]:
        return cls(width)(text)


# Backwards-compatible shim so existing call sites keep working.
def wrap_text(text: str, width: int = 46) -> list[str]:
    return TextWrapper.wrap(text, width)