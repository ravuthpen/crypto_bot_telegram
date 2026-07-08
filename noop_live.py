class _NoopLive:
    """Stand-in for rich.Live when headless (Docker/piped); swallows every call."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, _name):
        return lambda *a, **k: None