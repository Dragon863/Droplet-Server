import os
import dotenv

dotenv.load_dotenv()


def get_env(key: str, default: str = None):
    """Get environment variable or return default value."""
    value = os.getenv(key)
    if value is None and default is None:
        raise ValueError(f"Environment variable {key} not found.")
    return value if value is not None else default
