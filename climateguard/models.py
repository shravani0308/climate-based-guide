"""
Pydantic models for ClimateGuard AI Environment
FIX: All models now inherit from OpenEnv base classes (Action, Observation, State)
     done and reward are INHERITED from Observation — do NOT redefine them
     episode_id and step_count are INHERITED from State — do NOT redefine them
"""

from typing import Dict, List, Optional, Any
from pydantic import Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    from pydantic import BaseModel
    class Action(BaseModel):
        pass
    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0


class ClimateAction(Action):
    """Action space for climate crisis response"""
    firefighting_allocation: Dict[str, int] = Field(
        default_factory=dict,
        description="Firefighting units per region e.g. {'0':3,'2':2}"
    )
    flood_rescue_allocation: Dict[str, int] = Field(
        default_factory=dict,
        description="Flood rescue teams per region"
    )
    medical_allocation: Dict[str, int] = Field(
        default_factory=dict,
        description="Medical teams per region"
    )
    evacuation_allocation: Dict[str, int] = Field(
        default_factory=dict,
        description="Evacuation buses per region"
    )
    power_restoration: List[int] = Field(
        default_factory=list,
        description="Region IDs for power restoration"
    )
    water_supply_restoration: List[int] = Field(
        default_factory=list,
        description="Region IDs for water restoration"
    )
    firebreaks: List[int] = Field(
        default_factory=list,
        description="Region IDs to create firebreaks"
    )
    flood_barriers: List[int] = Field(
        default_factory=list,
        description="Region IDs to build flood barriers"
    )
    evacuation_orders: List[int] = Field(
        default_factory=list,
        description="Region IDs to issue evacuation orders"
    )
    food_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Food packages per region"
    )
    water_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Water units per region"
    )
    satellite_coverage: List[int] = Field(
        default_factory=list,
        description="Region IDs for satellite monitoring"
    )
    priority_regions: List[int] = Field(
        default_factory=list,
        description="High priority region IDs"
    )


class ClimateObservation(Observation):
    """
    Observation from the climate crisis environment.
    IMPORTANT: done and reward are INHERITED from Observation — do NOT redefine.
    """
    day: int = Field(default=0, description="Current day of crisis response")
    total_casualties: int = Field(default=0, description="Total casualties so far")
    total_evacuated: int = Field(default=0, description="Total people evacuated")
    regions: List[dict] = Field(default_factory=list, description="List of region states")
    resources: dict = Field(default_factory=dict, description="Available resources")
    weather_forecast: str = Field(default="Clear", description="Weather conditions")
    task_description: str = Field(default="", description="Current task description")
    outcome_summary: str = Field(default="", description="Last step outcome")
    survival_rate: float = Field(default=1.0, description="Population survival rate 0-1")


class ClimateEpisodeState(State):
    """
    Episode metadata.
    IMPORTANT: episode_id and step_count are INHERITED from State — do NOT redefine.
    """
    task_id: str = Field(default="multi_crisis", description="Active task ID")
    num_regions: int = Field(default=10, description="Number of regions in scenario")
    difficulty: str = Field(default="MEDIUM", description="Task difficulty level")
    max_days: int = Field(default=15, description="Maximum days in episode")
    scenario_seed: int = Field(default=0, description="Random seed for reproducibility")
