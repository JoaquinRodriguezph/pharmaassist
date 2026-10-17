"""Environment-variable helper shared by backends (imports no provider SDK)."""

import os


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set (see .env.example).")
    return value