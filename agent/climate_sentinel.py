"""
ClimateGuard SENTINEL Agent
Strategic Emergency Network for Triage & Intelligent Navigation of Losses
Adapted for Climate Crisis Response
"""

from openai import OpenAI
import json
from typing import List, Dict


class ClimateSentinelAgent:
    """
    SENTINEL: Strategic Emergency Network for
    Triage & Intelligent Navigation of Losses
    
    6-Layer Architecture:
    1. PERCEIVER - Analyzes climate crisis situation
    2. REASONER - Makes strategic decisions
    3. CRITIC - Self-evaluates decisions
    4. MEMORY - Tracks episode decisions
    5. REFLECTOR - Learns from outcomes
    6. EXECUTOR - Formats final actions
    """
    
    def __init__(self, api_base_url: str, model_name: str, api_key: str):
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model = model_name
        self.episode_memory: List[Dict] = []
        self.learned_strategies: List[str] = []
        self.episode_count = 0
    
    # ─────────────────────────────────────────────
    # LAYER 1: PERCEIVER
    # Analyzes climate crisis situation with expertise
    # ─────────────────────────────────────────────
    def perceive(self, state: dict) -> str:
        """Analyze climate crisis situation with expert knowledge"""
        
        # Build system prompt with climate expertise
        system_prompt = f"""
You are SENTINEL — an expert AI climate crisis coordinator.

You have deep knowledge of:
- Climate crisis patterns (wildfires, floods, droughts, hurricanes)
- Resource allocation under extreme constraints
- Evacuation logistics and timing
- Infrastructure vulnerability assessment
- Multi-crisis cascade effects

LEARNED STRATEGIES from past episodes:
{chr(10).join(self.learned_strategies[-5:]) if self.learned_strategies else "None yet"}

Your job: Analyze the current situation and identify:
1. Most critical threats (severity × population)
2. Resource availability vs needs
3. Infrastructure vulnerabilities
4. Cascade risk factors
5. Time-sensitive decisions needed

Be precise and actionable.
"""

        # Format state for analysis
        regions = state.get('regions', [])
        resources = state.get('resources', {})
        day = state.get('day', 0)
        
        situation = f"""
DAY {day} CLIMATE CRISIS STATUS:

REGIONS AT RISK:
"""
        for region in regions[:5]:  # Top 5 regions
            situation += f"""
- {region['region_name']}: {region['crisis_type']} 
  Severity: {region['severity']:.2f} | Population: {region['population']:,}
  Power: {region.get('power_grid', 1.0):.1f} | Water: {region.get('water_supply', 1.0):.1f}
"""

        situation += f"""
AVAILABLE RESOURCES:
- Firefighting Units: {resources.get('firefighting_units', 0)}
- Medical Teams: {resources.get('medical_teams', 0)}
- Evacuation Buses: {resources.get('evacuation_buses', 0)}
- Budget: ${resources.get('budget_millions', 0)}M

CASUALTIES SO FAR: {state.get('total_casualties', 0)}
EVACUATED SO FAR: {state.get('total_evacuated', 0)}

Analyze this situation and provide strategic assessment.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": situation}
            ],
            temperature=0.2,
            max_tokens=800,
            timeout=30
        )
        return response.choices[0].message.content

    # ─────────────────────────────────────────────
    # LAYER 2: REASONER
    # Makes strategic climate crisis decisions
    # ─────────────────────────────────────────────
    def reason(self, perception: str, task_id: str) -> str:
        """Make strategic climate crisis decisions"""
        
        system_prompt = f"""
You are SENTINEL's decision-making layer for climate crisis response.

Based on the situation analysis, make SPECIFIC resource allocation decisions.

DECISION FRAMEWORK:
STEP 1: What is the highest priority threat right now?
STEP 2: What is the biggest risk if I do nothing this turn?
STEP 3: What infrastructure action is most urgent?
STEP 4: How should I allocate limited resources optimally?
STEP 5: What should I reserve for cascade effects?

OUTPUT FORMAT - Return valid JSON:
{{
  "firefighting_allocation": {{"region_id": units}},
  "flood_rescue_allocation": {{"region_id": units}},
  "medical_allocation": {{"region_id": units}},
  "evacuation_allocation": {{"region_id": units}},
  "power_restoration": [region_ids],
  "water_supply_restoration": [region_ids],
  "firebreaks": [region_ids],
  "flood_barriers": [region_ids],
  "evacuation_orders": [region_ids],
  "food_distribution": {{"region_id": tons}},
  "water_distribution": {{"region_id": tons}},
  "satellite_coverage": [region_ids],
  "priority_regions": [region_ids]
}}

Think through each step, then output your final action as JSON.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SITUATION ANALYSIS:\n{perception}\n\nTask: {task_id}\n\nMake your decision:"}
            ],
            temperature=0.1,  # Low temp for consistent decisions
            max_tokens=1000,
            timeout=30
        )
        return response.choices[0].message.content

    # ─────────────────────────────────────────────
    # LAYER 3: CRITIC
    # Self-evaluates decisions before acting
    # ─────────────────────────────────────────────
    def critique(self, reasoning: str, perception: str) -> str:
        """Self-evaluate climate crisis decision before acting"""

        critique_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """
You are SENTINEL's self-critic for climate crisis decisions.
Your job is to find flaws in decisions before they cost lives.

Check for:
- Sending resources to regions without infrastructure
- Ignoring high-population areas in immediate danger
- Deploying all resources with no reserves for cascades
- Missing critical infrastructure protection
- Ignoring weather/climate factors

If decision is good, say "APPROVED".
If flawed, say "REJECTED: [specific reason]" and suggest fix.
"""},
                {"role": "user", "content": f"""
SITUATION:
{perception}

PROPOSED DECISION:
{reasoning}

Is this decision correct for climate crisis response? Be harsh. Lives depend on it.
"""}
            ],
            temperature=0.0,
            timeout=30
        )
        return critique_response.choices[0].message.content

    # ─────────────────────────────────────────────
    # LAYER 4: MEMORY
    # Tracks all decisions within this episode
    # ─────────────────────────────────────────────
    def remember(self, day: int, action: dict, reward: float, outcome: str):
        """Remember decision and outcome"""
        self.episode_memory.append({
            "day": day,
            "action": action,
            "reward": reward,
            "outcome": outcome
        })

    def _format_memory(self) -> str:
        """Format recent memory for context"""
        if not self.episode_memory:
            return "No decisions made yet this episode."
        lines = []
        for m in self.episode_memory[-4:]:  # Last 4 decisions
            lines.append(
                f"Day {m['day']}: reward={m['reward']:.2f} "
                f"— {m['outcome']}"
            )
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # LAYER 5: REFLECTOR
    # After each episode — learns what worked
    # ─────────────────────────────────────────────
    def reflect(self, total_score: float, lives_saved: int):
        """Learn from episode outcomes"""

        if not self.episode_memory:
            return

        reflection = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """
You are SENTINEL's learning system for climate crisis response.
After each episode, extract 1-2 specific strategic lessons.
Format: "LESSON: [specific actionable insight]"
Be concrete. E.g.:
"LESSON: Always protect power grid in regions with hospitals first
         — power loss cascades to medical capacity failure"
"""},
                {"role": "user", "content": f"""
CLIMATE CRISIS EPISODE SUMMARY:
Total Score: {total_score:.3f}
Lives Saved: {lives_saved:,}
Episode: #{self.episode_count}

DECISION HISTORY:
{self._format_memory()}

What 2 specific lessons should I remember for next climate crisis?
"""}
            ],
            temperature=0.2,
            timeout=30
        )

        # Extract and store lessons
        lessons = reflection.choices[0].message.content
        new_lessons = [
            line.strip() for line in lessons.split('\n')
            if line.strip().startswith('LESSON:')
        ]
        self.learned_strategies.extend(new_lessons)

        # Keep only top 10 lessons (most recent)
        self.learned_strategies = self.learned_strategies[-10:]

        # Reset episode memory
        self.episode_memory = []
        self.episode_count += 1
        print(f"\n🧠 CLIMATE SENTINEL learned: {new_lessons}")

    # ─────────────────────────────────────────────
    # LAYER 6: EXECUTOR
    # Formats final action perfectly for OpenEnv
    # ─────────────────────────────────────────────
    def execute(self, reasoning: str) -> dict:
        """Extract and format final action for ClimateGuard"""
        try:
            # Find JSON block in response
            start = reasoning.find('{')
            end = reasoning.rfind('}') + 1
            json_str = reasoning[start:end]
            action = json.loads(json_str)
            return action
        except Exception:
            # Fallback — safe default action for climate crisis
            return self._safe_climate_fallback()

    def _safe_climate_fallback(self) -> dict:
        """Safe fallback action for climate crisis"""
        return {
            "firefighting_allocation": {"0": 2, "1": 2},
            "flood_rescue_allocation": {"0": 1},
            "medical_allocation": {"0": 1, "1": 1},
            "evacuation_allocation": {"0": 3, "1": 2},
            "power_restoration": [0],
            "water_supply_restoration": [1],
            "firebreaks": [0],
            "flood_barriers": [1],
            "evacuation_orders": [0, 1],
            "food_distribution": {"0": 10, "1": 5},
            "water_distribution": {"0": 8, "1": 6},
            "satellite_coverage": [0, 1, 2],
            "priority_regions": [0, 1]
        }

    # ─────────────────────────────────────────────
    # MAIN DECISION LOOP
    # All 6 layers working together
    # ─────────────────────────────────────────────
    def decide(self, state: dict, task_id: str, day: int) -> dict:
        """Make climate crisis decision using all 6 layers"""

        print(f"\n🧠 CLIMATE SENTINEL thinking... (Day {day})")

        # Layer 1: Perceive
        perception = self.perceive(state)

        # Layer 2: Reason
        reasoning = self.reason(perception, task_id)
        print(f"💭 Reasoning: {reasoning[:200]}...")

        # Layer 3: Critique (self-check)
        critique = self.critique(reasoning, perception)
        print(f"🔍 Critique: {critique[:100]}")

        # If rejected — re-reason with critique feedback
        if "REJECTED" in critique:
            reasoning = self.reason(
                perception + f"\n\nCRITIQUE FEEDBACK: {critique}",
                task_id
            )
            print("♻️  Re-reasoning after critique...")

        # Layer 6: Execute
        action = self.execute(reasoning)
        print(f"⚡ Action: {list(action.keys())}")

        return action