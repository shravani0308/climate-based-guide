"""
Pydantic models for ClimateGuard AI Environment
Defines all typed models for actions, observations, and state
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid


class ClimateAction(BaseModel):
    """Action space for climate crisis response"""
    firefighting_allocation: Dict[str, int] = Field(default_factory=dict, description="Firefighting units per region")
    flood_rescue_allocation: Dict[str, int] = Field(default_factory=dict, description="Flood rescue teams per region")
    medical_allocation: Dict[str, int] = Field(default_factory=dict, description="Medical teams per region")
    evacuation_allocation: Dict[str, int] = Field(default_factory=dict, description="Evacuation buses per region")
    power_restoration: List[int] = Field(default_factory=list, description="Regions for power restoration")
    water_supply_restoration: List[int] = Field(default_factory=list, description="Regions for water restoration")
    firebreaks: List[int] = Field(default_factory=list, description="Regions to create firebreaks")
    flood_barriers: List[int] = Field(default_factory=list, description="Regions to build flood barriers")
    evacuation_orders: List[int] = Field(default_factory=list, description="Regions to issue evacuation orders")
    food_distribution: Dict[str, int] = Field(default_factory=dict, description="Food distribution per region")
    water_distribution: Dict[str, int] = Field(default_factory=dict, description="Water distribution per region")
    satellite_coverage: List[int] = Field(default_factory=list, description="Regions for satellite monitoring")
    priority_regions: List[int] = Field(default_factory=list, description="High priority regions")


class RegionState(BaseModel):
    """State of a single region"""
    region_id: int
    region_name: str
    crisis_type: str  # WILDFIRE, FLOOD, DROUGHT, STORM
    severity: float = Field(ge=0.0, le=1.0)
    population: int
    casualties: int = 0
    evacuated: int = 0
    power_grid: float = Field(ge=0.0, le=1.0, default=1.0)
    water_supply: float = Field(ge=0.0, le=1.0, default=1.0)
    road_access: float = Field(ge=0.0, le=1.0, default=1.0)
    communication: float = Field(ge=0.0, le=1.0, default=1.0)


class ResourceState(BaseModel):
    """Available resources for crisis response"""
    budget_millions: float
    firefighting_units: int
    medical_teams: int
    evacuation_buses: int
    engineering_crews: int
    helicopters: int
    satellites: int


class ClimateObservation(BaseModel):
    """Observation from the climate crisis environment"""
    day: int
    total_casualties: int
    total_evacuated: int
    regions: List[RegionState]
    resources: ResourceState
    weather_forecast: str
    done: bool = False
    reward: float = 0.0
    info: Dict[str, Any] = Field(default_factory=dict)


class ClimateEpisodeState(BaseModel):
    """Complete episode state"""
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_count: int = 0
    done: bool = False
    reward: float = 0.0