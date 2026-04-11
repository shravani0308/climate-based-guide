"""
ClimateGuard AI Environment
FIX: 3 separate task-specific graders instead of one shared reward function
FIX: Proper State with inherited episode_id/step_count
FIX: Proper Observation with inherited done/reward
"""

import random
import uuid
import math
import numpy as np
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ClimateAction, ClimateObservation, ClimateEpisodeState

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass


class ClimateGuardEnvironment(Environment):
    """
    ClimateGuard AI: Multi-Crisis Climate Response Coordination
    Trains AI agents to minimize casualties and coordinate resources
    across real-world climate disaster scenarios.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    REGION_NAMES = [
        "Northern Valley", "Coastal Plains", "Mountain Ridge",
        "Central Basin", "Eastern Highlands", "Southern Delta",
        "Western Plateau", "River Junction", "Desert Edge",
        "Forest Belt", "Urban Center", "Agricultural Zone"
    ]

    TASK_CONFIGS = {
        "single_crisis": {
            "description": "TASK 1 (Easy): Coordinate response to a single wildfire. "
                           "Allocate firefighting teams and evacuation resources optimally.",
            "num_regions": 5,
            "crisis_types": ["WILDFIRE"],
            "max_days": 10,
            "difficulty": "EASY"
        },
        "multi_crisis": {
            "description": "TASK 2 (Medium): Coordinate simultaneous wildfire, flood, and drought. "
                           "Balance limited resources across competing crises.",
            "num_regions": 10,
            "crisis_types": ["WILDFIRE", "FLOOD", "DROUGHT"],
            "max_days": 15,
            "difficulty": "MEDIUM"
        },
        "cascade_crisis": {
            "description": "TASK 3 (Hard): Cascading disasters — hurricane triggers flood, "
                           "which destroys roads, which blocks evacuations. Adapt in real-time.",
            "num_regions": 12,
            "crisis_types": ["WILDFIRE", "FLOOD", "HURRICANE", "HEATWAVE"],
            "max_days": 20,
            "difficulty": "HARD"
        },
    }

    def __init__(self):
        self._day = 0
        self._regions = []
        self._resources = {}
        self._total_casualties = 0
        self._total_evacuated = 0
        self._task_id = "multi_crisis"
        self._seed = 0
        self._step_count = 0
        self._episode_id = str(uuid.uuid4())
        # Task-specific tracking for graders
        self._wildfire_suppressed = 0
        self._flood_rescued = 0
        self._cascade_events = []
        self._power_restored_regions = set()

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs):
        """Reset to fresh disaster scenario."""
        self._seed = seed if seed is not None else random.randint(0, 99999)
        random.seed(self._seed)
        np.random.seed(self._seed % (2 ** 31))

        self._task_id = task_id or "multi_crisis"
        task_config = self.TASK_CONFIGS.get(self._task_id, self.TASK_CONFIGS["multi_crisis"])

        self._day = 0
        self._step_count = 0
        self._episode_id = episode_id or str(uuid.uuid4())
        self._total_casualties = 0
        self._total_evacuated = 0
        self._wildfire_suppressed = 0
        self._flood_rescued = 0
        self._cascade_events = []
        self._power_restored_regions = set()

        self._regions = self._generate_regions(task_config)
        self._resources = self._generate_resources(task_config)

        return ClimateObservation(
            done=False,
            reward=None,
            day=0,
            total_casualties=0,
            total_evacuated=0,
            regions=self._regions,
            resources=self._resources,
            weather_forecast="Clear — Crisis began. Respond immediately.",
            task_description=task_config["description"],
            outcome_summary="Crisis started. Deploy resources now.",
            survival_rate=1.0,
        )

    def step(self, action: ClimateAction, timeout_s=None, **kwargs):
        """Simulate one day of crisis response."""
        self._day += 1
        self._step_count += 1
        task_config = self.TASK_CONFIGS.get(self._task_id, self.TASK_CONFIGS["multi_crisis"])
        cascade = []

        # Dynamic weather progression
        weather = self._get_weather(self._day, task_config["difficulty"])

        # Cascade event: hurricane floods roads
        if self._task_id == "cascade_crisis" and self._day == 5:
            for region in self._regions:
                if region["crisis_type"] == "HURRICANE":
                    region["road_access"] = max(0.0, region["road_access"] - 0.4)
                    region["severity"] = min(1.0, region["severity"] + 0.1)
            cascade.append(f"Day{self._day}: Hurricane triggered road flooding — evacuation routes blocked")
            self._cascade_events.append(cascade[-1])

        # Execute actions
        outcome = self._execute_actions(action, cascade)
        self._total_casualties += outcome["casualties"]
        self._total_evacuated += outcome["evacuated"]
        self._wildfire_suppressed += outcome["wildfire_suppressed"]
        self._flood_rescued += outcome["flood_rescued"]

        # Track power restoration
        for rid in action.power_restoration:
            self._power_restored_regions.add(rid)

        # Consume resources
        self._consume_resources(action)

        # Compute task-specific reward
        reward = self._compute_task_reward(action, outcome)

        # Check done
        done = self._day >= task_config["max_days"]

        # Survival rate
        total_pop = sum(r["population"] for r in self._regions)
        survival_rate = max(0.0, 1.0 - (self._total_casualties / max(total_pop, 1)))

        summary = (
            f"Day {self._day}/{task_config['max_days']} | "
            f"Saved: +{outcome['lives_saved']} | "
            f"Evacuated: +{outcome['evacuated']} | "
            f"Casualties: +{outcome['casualties']} | "
            f"Weather: {weather}"
        )
        if cascade:
            summary += f" | CASCADE: {cascade[-1]}"

        return ClimateObservation(
            done=done,
            reward=reward,
            day=self._day,
            total_casualties=self._total_casualties,
            total_evacuated=self._total_evacuated,
            regions=self._regions,
            resources=self._resources,
            weather_forecast=weather,
            task_description=task_config["description"],
            outcome_summary=summary,
            survival_rate=round(survival_rate, 3),
        )

    @property
    def state(self) -> ClimateEpisodeState:
        task_config = self.TASK_CONFIGS.get(self._task_id, self.TASK_CONFIGS["multi_crisis"])
        return ClimateEpisodeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            num_regions=task_config["num_regions"],
            difficulty=task_config["difficulty"],
            max_days=task_config["max_days"],
            scenario_seed=self._seed,
        )

    # ── Task-Specific Graders ────────────────────────────────────────────────

    def _compute_task_reward(self, action: ClimateAction, outcome: Dict) -> float:
        """Route to correct task grader. All return float in [0.0, 1.0]."""
        if self._task_id == "single_crisis":
            return self._grade_single_crisis(action, outcome)
        elif self._task_id == "multi_crisis":
            return self._grade_multi_crisis(action, outcome)
        elif self._task_id == "cascade_crisis":
            return self._grade_cascade_crisis(action, outcome)
        else:
            return self._grade_multi_crisis(action, outcome)

    def _grade_single_crisis(self, action: ClimateAction, outcome: Dict) -> float:
        """
        TASK 1 Grader — Single Wildfire
        Focus: firefighting effectiveness + evacuation success
        """
        total_pop = sum(r["population"] for r in self._regions)

        # Component 1 (50%): Wildfire suppression
        wildfire_regions = [r for r in self._regions if r["crisis_type"] == "WILDFIRE"]
        avg_severity = sum(r["severity"] for r in wildfire_regions) / max(len(wildfire_regions), 1)
        suppression_score = 1.0 - avg_severity  # Lower severity = better

        # Component 2 (30%): Lives saved
        lives_score = min(1.0, outcome["lives_saved"] / max(total_pop * 0.05, 1))

        # Component 3 (20%): Evacuation completeness
        evacuation_score = min(1.0, outcome["evacuated"] / max(total_pop * 0.3, 1))

        reward = (
            0.50 * suppression_score +
            0.30 * lives_score +
            0.20 * evacuation_score
        )
        return float(np.clip(reward, 0.0, 1.0))

    def _grade_multi_crisis(self, action: ClimateAction, outcome: Dict) -> float:
        """
        TASK 2 Grader — Multi-Crisis Coordination
        Focus: equity across crisis types + resource efficiency + lives saved
        """
        total_pop = sum(r["population"] for r in self._regions)

        # Component 1 (40%): Lives saved
        lives_score = min(1.0, outcome["lives_saved"] / max(total_pop * 0.08, 1))

        # Component 2 (25%): Equity — responded to multiple crisis types
        crisis_types_responded = set()
        for rid, teams in action.firefighting_allocation.items():
            if teams > 0 and int(rid) < len(self._regions):
                crisis_types_responded.add(self._regions[int(rid)]["crisis_type"])
        for rid, teams in action.flood_rescue_allocation.items():
            if teams > 0 and int(rid) < len(self._regions):
                crisis_types_responded.add(self._regions[int(rid)]["crisis_type"])
        total_crisis_types = len(set(r["crisis_type"] for r in self._regions))
        equity_score = len(crisis_types_responded) / max(total_crisis_types, 1)

        # Component 3 (20%): Casualty prevention
        casualty_rate = self._total_casualties / max(total_pop, 1)
        prevention_score = max(0.0, 1.0 - (casualty_rate * 10))

        # Component 4 (15%): Resource efficiency
        total_dispatched = (
            sum(action.firefighting_allocation.values()) +
            sum(action.flood_rescue_allocation.values()) +
            sum(action.medical_allocation.values())
        )
        efficiency = min(1.0, total_dispatched / max(len(self._regions) * 2, 1))

        reward = (
            0.40 * lives_score +
            0.25 * equity_score +
            0.20 * prevention_score +
            0.15 * efficiency
        )
        return float(np.clip(reward, 0.0, 1.0))

    def _grade_cascade_crisis(self, action: ClimateAction, outcome: Dict) -> float:
        """
        TASK 3 Grader — Cascade Crisis Management
        Focus: adaptation to cascade events + infrastructure + lives under cascading failures
        """
        total_pop = sum(r["population"] for r in self._regions)

        # Component 1 (35%): Lives saved under cascade conditions
        lives_score = min(1.0, outcome["lives_saved"] / max(total_pop * 0.06, 1))

        # Component 2 (25%): Infrastructure resilience — did agent restore power?
        infrastructure_score = len(self._power_restored_regions) / max(len(self._regions), 1)

        # Component 3 (25%): Adaptation — used varied response types
        response_types_used = 0
        if sum(action.firefighting_allocation.values()) > 0:
            response_types_used += 1
        if sum(action.flood_rescue_allocation.values()) > 0:
            response_types_used += 1
        if action.evacuation_orders:
            response_types_used += 1
        if action.power_restoration:
            response_types_used += 1
        if action.flood_barriers:
            response_types_used += 1
        adaptation_score = min(1.0, response_types_used / 4.0)

        # Component 4 (15%): Survival rate
        survival_rate = max(0.0, 1.0 - (self._total_casualties / max(total_pop, 1)))
        survival_score = min(1.0, survival_rate * 2)  # Amplify since cascades are deadly

        reward = (
            0.35 * lives_score +
            0.25 * infrastructure_score +
            0.25 * adaptation_score +
            0.15 * survival_score
        )
        return float(np.clip(reward, 0.0, 1.0))

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _generate_regions(self, task_config: Dict) -> List[dict]:
        regions = []
        num_regions = task_config["num_regions"]
        crisis_types = task_config["crisis_types"]

        for i in range(num_regions):
            crisis_type = random.choice(crisis_types)
            severity = round(random.uniform(0.4, 0.9), 3)
            population = random.randint(50000, 500000)
            regions.append({
                "region_id": i,
                "region_name": self.REGION_NAMES[i % len(self.REGION_NAMES)],
                "crisis_type": crisis_type,
                "severity": severity,
                "population": population,
                "casualties": 0,
                "evacuated": 0,
                "power_grid": round(max(0.0, 1.0 - severity * 0.6), 2),
                "water_supply": round(max(0.0, 1.0 - severity * 0.7), 2),
                "road_access": round(max(0.1, 1.0 - severity * 0.5), 2),
                "communication": round(max(0.0, 1.0 - severity * 0.4), 2),
            })
        return regions

    def _generate_resources(self, task_config: Dict) -> dict:
        n = task_config["num_regions"]
        mult = {"EASY": 1.5, "MEDIUM": 1.0, "HARD": 0.7}.get(task_config["difficulty"], 1.0)
        return {
            "budget_millions": round(n * 50 * mult, 1),
            "firefighting_units": int(n * 3 * mult),
            "medical_teams": int(n * 2 * mult),
            "evacuation_buses": int(n * 4 * mult),
            "engineering_crews": int(n * 2 * mult),
            "helicopters": int(max(2, n * 0.5 * mult)),
            "satellites": int(max(1, n * 0.3 * mult)),
        }

    def _execute_actions(self, action: ClimateAction, cascade: list) -> Dict:
        lives_saved = 0
        casualties = 0
        evacuated = 0
        wildfire_suppressed = 0
        flood_rescued = 0

        for region in self._regions:
            rid = str(region["region_id"])
            at_risk = region["population"] - region["evacuated"]

            # Firefighting (wildfire)
            if region["crisis_type"] == "WILDFIRE":
                units = action.firefighting_allocation.get(rid, 0)
                if units > 0:
                    eff = min(1.0, units / 5.0)
                    reduction = eff * 0.12
                    region["severity"] = max(0.0, region["severity"] - reduction)
                    saved = int(at_risk * eff * 0.04)
                    lives_saved += saved
                    wildfire_suppressed += int(reduction * 100)

            # Flood rescue
            if region["crisis_type"] in ["FLOOD", "HURRICANE"]:
                units = action.flood_rescue_allocation.get(rid, 0)
                if units > 0:
                    eff = min(1.0, units / 4.0)
                    saved = int(at_risk * eff * 0.05)
                    lives_saved += saved
                    flood_rescued += saved

            # Medical teams
            medical = action.medical_allocation.get(rid, 0)
            if medical > 0:
                lives_saved += int(at_risk * 0.01 * min(1.0, medical / 3.0))

            # Evacuation
            if region["region_id"] in action.evacuation_orders:
                road_ok = region["road_access"] > 0.2
                buses = action.evacuation_allocation.get(rid, 0) if road_ok else 0
                if buses > 0:
                    can_evac = min(at_risk, buses * 50)
                    region["evacuated"] += can_evac
                    evacuated += can_evac
                    lives_saved += int(can_evac * 0.9)
                elif not road_ok:
                    cascade.append(f"Region{region['region_id']}: Road blocked — evacuation failed")

            # Power restoration
            if region["region_id"] in action.power_restoration:
                region["power_grid"] = min(1.0, region["power_grid"] + 0.3)
                region["communication"] = min(1.0, region["communication"] + 0.2)

            # Passive casualties if ignored
            if region["severity"] > 0.5:
                casualty_rate = region["severity"] * 0.0008
                loss = int(at_risk * casualty_rate)
                casualties += loss
                region["casualties"] += loss

        return {
            "lives_saved": lives_saved,
            "casualties": casualties,
            "evacuated": evacuated,
            "wildfire_suppressed": wildfire_suppressed,
            "flood_rescued": flood_rescued,
            "summary": f"Day {self._day}: Saved {lives_saved}, Evacuated {evacuated}, Casualties {casualties}",
        }

    def _consume_resources(self, action: ClimateAction):
        r = self._resources
        r["firefighting_units"] = max(0, r["firefighting_units"] - sum(action.firefighting_allocation.values()) // 3)
        r["medical_teams"] = max(0, r["medical_teams"] - sum(action.medical_allocation.values()) // 4)
        r["evacuation_buses"] = max(0, r["evacuation_buses"] - sum(action.evacuation_allocation.values()) // 3)
        r["budget_millions"] = max(0.0, r["budget_millions"] - 8.0)

    def _get_weather(self, day: int, difficulty: str) -> str:
        if day < 3:
            return "Clear — optimal response conditions"
        elif day < 7:
            return random.choice(["Clear", "Partly cloudy", "Windy — fire risk elevated"])
        elif day < 12:
            return random.choice(["Rain — flood risk rising", "Storm warning", "High winds"])
        else:
            return random.choice(["Severe storm", "Flood alert", "Extreme heat"])
