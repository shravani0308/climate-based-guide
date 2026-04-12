"""
FastAPI app for ClimateGuard AI
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

from openenv.core.env_server import create_fastapi_app

try:
    from .environment import ClimateGuardEnvironment
except ImportError:
    from server.environment import ClimateGuardEnvironment

app = create_fastapi_app(ClimateGuardEnvironment)


def main():
    """Entry point for [project.scripts] — runs the uvicorn server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
