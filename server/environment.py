"""
ClimateGuard AI Environment
Complete implementation for facebook project
"""

import random
import uuid
import numpy as np
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ClimateAction, ClimateObservation, ClimateEpisodeState,
    RegionState, ResourceState
)

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass


class ClimateGuardEnvironment(Environment):
    """ClimateGuard AI: Multi-Crisis Climate Response"""
    
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    CRISIS_TYPES = ["WILDFIRE", "FLOOD", "DROUGHT", "HURRICANE", "HEATWAVE"]
    REGION_NAMES = [
        "Northern Valley", "Coastal Plains", "Mountain Ridge",
        "Central Basin", "Eastern Highlands", "Southern Delta",
        "Western Plateau", "River Junction", "Desert Edge",
        "Forest Belt", "Urban Center", "Agricultural Zone"
    ]
    
    TASK_CONFIGS = {
        "single_crisis": {
            "description": "TASK 1 (Easy): Single wildfire crisis",
            "num_regions": 5,
            "crisis_types": ["WILDFIRE"],
            "max_days": 10,
            "difficulty": "EASY"
        },
        "multi_crisis": {
            "description": "TASK 2 (Medium): Multiple crises",
            "num_regions": 10,
            "crisis_types": ["WILDFIRE", "FLOOD", "DROUGHT"],
            "max_days": 15,
            "difficulty": "MEDIUM"
        },
        "cascade_crisis": {
            "description": "TASK 3 (Hard): Cascading effects",
            "num_regions": 12,
            "crisis_types": ["WILDFIRE", "FLOOD", "HURRICANE", "HEATWAVE"],
            "max_days": 20,
            "difficulty": "HARD"
        }
    }
    
    def __init__(self):
        self._day = 0
        self._regions = []
        self._resources = GlobalResources()
        self._total_casualties = 0
        self._total_evacuated = 0
        self._task_id = "multi_crisis"
        self._seed = 0
    
    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs):
        """Reset environment"""
        self._seed = seed if seed is not None else random.randint(0, 99999)
        random.seed(self._seed)
        np.random.seed(self._seed % (2 ** 31))
        
        self._task_id = task_id or "multi_crisis"
        task_config = self.TASK_CONFIGS.get(self._task_id, self.TASK_CONFIGS["multi_crisis"])
        
        self._day = 0
        self._total_casualties = 0
        self._total_evacuated = 0
        
        # Generate regions
        self._regions = self._generate_regions(task_config)
        self._resources = self._generate_resources(task_config)
        
        return ClimateObservation(
            day=0,
            total_casualties=0,
            total_evacuated=0,
            regions=self._regions,
            resources=self._resources,
            weather_forecast="Clear skies expected",
            done=False,
            reward=0.0,
            info={"task_description": task_config["description"]}
        )
    
    def step(self, action: ClimateAction, timeout_s=None, **kwargs):
        """Execute one step"""
        self._day += 1
        
        # Execute actions
        outcome = self._execute_actions(action)
        self._total_casualties += outcome["casualties"]
        self._total_evacuated += outcome["evacuated"]
        
        # Consume resources
        self._consume_resources(action)
        
        # Calculate reward
        reward = self._compute_reward(outcome)
        
        # Check done
        task_config = self.TASK_CONFIGS.get(self._task_id, self.TASK_CONFIGS["multi_crisis"])
        done = self._day >= task_config["max_days"]
        
        return ClimateObservation(
            day=self._day,
            total_casualties=self._total_casualties,
            total_evacuated=self._total_evacuated,
            regions=self._regions,
            resources=self._resources,
            weather_forecast="Conditions deteriorating",
            done=done,
            reward=reward,
            info={"outcome_summary": outcome["summary"]}
        )
    
    @property
    def state(self):
        return ClimateEpisodeState(
            episode_id=str(uuid.uuid4()),
            step_count=self._day,
            task_id=self._task_id
        )
    
    def _generate_regions(self, task_config: Dict) -> List[RegionState]:
        """Generate regions with crises"""
        regions = []
        num_regions = task_config["num_regions"]
        crisis_types = task_config["crisis_types"]
        
        for i in range(num_regions):
            crisis_type = random.choice(crisis_types)
            severity = random.uniform(0.4, 0.9)
            population = random.randint(50000, 500000)
            
            region = RegionState(
                region_id=i,
                region_name=self.REGION_NAMES[i % len(self.REGION_NAMES)],
                crisis_type=crisis_type,
                severity=round(severity, 3),
                population=population,
                casualties=0,
                evacuated=0,
                power_grid=round(max(0.0, 1.0 - severity * 0.6), 2),
                water_supply=round(max(0.0, 1.0 - severity * 0.7), 2),
                road_access=round(max(0.0, 1.0 - severity * 0.5), 2),
                communication=round(max(0.0, 1.0 - severity * 0.4), 2)
            )
            regions.append(region)
        
        return regions
    
    def _generate_resources(self, task_config: Dict) -> ResourceState:
        """Generate resources"""
        num_regions = task_config["num_regions"]
        mult = {"EASY": 1.5, "MEDIUM": 1.0, "HARD": 0.7}.get(task_config["difficulty"], 1.0)
        
        return ResourceState(
            budget_millions=round(num_regions * 50 * mult, 1),
            firefighting_units=int(num_regions * 3 * mult),
            medical_teams=int(num_regions * 2 * mult),
            evacuation_buses=int(num_regions * 4 * mult),
            engineering_crews=int(num_regions * 2 * mult),
            helicopters=int(max(2, num_regions * 0.5 * mult)),
            satellites=int(max(1, num_regions * 0.3 * mult))
        )
    
    def _execute_actions(self, action: ClimateAction) -> Dict:
        """Execute actions"""
        casualties = 0
        evacuated = 0
        lives_saved = 0
        
        for region in self._regions:
            region_id_str = str(region.region_id)
            at_risk = region.population - region.evacuated
            
            # Firefighting
            if region.crisis_type == "WILDFIRE":
                units = action.firefighting_allocation.get(region_id_str, 0)
                if units > 0:
                    effectiveness = min(1.0, units / 5.0)
                    region.severity = max(0.0, region.severity - effectiveness * 0.15)
                    lives_saved += int(at_risk * effectiveness * 0.05)
            
            # Evacuation
            if region.region_id in action.evacuation_orders:
                buses = action.evacuation_allocation.get(region_id_str, 0)
                if buses > 0:
                    can_evacuate = min(at_risk, buses * 50)
                    region.evacuated += can_evacuate
                    evacuated += can_evacuate
                    lives_saved += int(can_evacuate * 0.95)
            
            # Casualties
            if region.severity > 0.5:
                casualty_rate = region.severity * 0.001
                casualties += int(at_risk * casualty_rate)
        
        summary = f"Day {self._day} | Saved: +{lives_saved} | Evacuated: +{evacuated} | Casualties: +{casualties}"
        
        return {
            "casualties": casualties,
            "evacuated": evacuated,
            "lives_saved": lives_saved,
            "summary": summary
        }
    
    def _compute_reward(self, outcome: Dict) -> float:
        """Calculate reward"""
        total_pop = sum(r.population for r in self._regions)
        lives_saved = outcome["lives_saved"]
        casualties = outcome["casualties"]
        
        lives_score = min(1.0, lives_saved / max(1, total_pop * 0.1))
        casualty_penalty = min(1.0, casualties / max(1, total_pop * 0.01))
        
        reward = 0.7 * lives_score + 0.3 * (1.0 - casualty_penalty)
        return float(np.clip(reward, 0.0, 1.0))
    
    def _consume_resources(self, action: ClimateAction):
        """Consume resources"""
        r = self._resources
        r.firefighting_units = max(0, r.firefighting_units - sum(action.firefighting_allocation.values()))
        r.medical_teams = max(0, r.medical_teams - sum(action.medical_allocation.values()))
        r.evacuation_buses = max(0, r.evacuation_buses - sum(action.evacuation_allocation.values()))
        r.budget_millions = max(0.0, r.budget_millions - 10.0)
