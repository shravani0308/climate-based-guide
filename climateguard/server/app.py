"""
FastAPI app for ClimateGuard AI
FIX: create_fastapi_app takes only the Environment class (newer openenv-core style)
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
