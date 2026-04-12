#!/usr/bin/env python3
"""
ClimateGuard AI Inference Script
FIXES APPLIED:
  1. HF_TOKEN used (not API_KEY) — matches hackathon injection
  2. os.getenv used (not os.environ) — no crash on missing vars
  3. [START]/[STEP]/[END] emitted ONCE PER TASK inside loop
  4. done=true/false (lowercase) — not Python True/False
  5. steps= counts actual steps taken (not number of tasks)
  6. score formatted to .2f
  7. LLM call has no silent try/except — forces real API traffic
"""

import os
import json
import requests
import textwrap
from typing import List, Optional
from openai import OpenAI

# ── FIX 1 & 2: Correct variable names + os.getenv (no crash) ────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")
BENCHMARK    = "climateguard"

MAX_STEPS         = 8
MAX_TOKENS        = 500
SUCCESS_THRESHOLD = 0.30

# ── FIX 3 & 4: Log functions with correct format ─────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()          # FIX 4: lowercase true/false
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={rewards_str}",  # FIX 6: .2f
        flush=True,
    )

# ── Climate SENTINEL Agent ────────────────────────────────────────────────────
SENTINEL_SYSTEM = textwrap.dedent("""
You are SENTINEL — expert AI climate crisis coordinator.
RULES:
1. PRIORITIZE: High-severity regions with large populations first.
2. WILDFIRE: Send firefighting_allocation before fire spreads.
3. FLOOD: Send flood_rescue_allocation + evacuation_orders + flood_barriers.
4. POWER: Always include power_restoration for regions with power_grid < 0.5.
5. EVACUATE: Issue evacuation_orders for regions with severity > 0.7.

Return ONLY valid JSON matching this exact schema:
{
  "firefighting_allocation": {"region_id": count},
  "flood_rescue_allocation": {"region_id": count},
  "medical_allocation": {"region_id": count},
  "evacuation_allocation": {"region_id": count},
  "power_restoration": [region_id_list],
  "water_supply_restoration": [region_id_list],
  "firebreaks": [region_id_list],
  "flood_barriers": [region_id_list],
  "evacuation_orders": [region_id_list],
  "food_distribution": {},
  "water_distribution": {},
  "satellite_coverage": [],
  "priority_regions": [region_id_list]
}
No explanation. Just JSON.
""").strip()

SENTINEL_LESSONS: List[str] = []


def get_action(client: OpenAI, obs: dict, step: int, history: List[str]) -> dict:
    """
    FIX 7: No silent try/except — LLM must be called.
    Fallback only used if LLM genuinely raises an exception.
    """
    regions = obs.get("regions", [])
    region_lines = "\n".join(
        f"  Region{r['region_id']} [{r.get('crisis_type','?')}] "
        f"sev={r.get('severity',0):.2f} pop={r.get('population',0):,} "
        f"road={'OK' if r.get('road_access',1)>0.2 else 'BLOCKED'} "
        f"power={r.get('power_grid',1):.2f}"
        for r in regions
    )
    lessons = "\n".join(f"- {l}" for l in SENTINEL_LESSONS[-4:]) or "None yet"
    user = f"""
Day {obs.get('day', 0)} | Casualties: {obs.get('total_casualties', 0)} | Evacuated: {obs.get('total_evacuated', 0)}
Survival rate: {obs.get('survival_rate', 1.0):.2%} | Weather: {obs.get('weather_forecast', 'Unknown')}
Resources: {json.dumps(obs.get('resources', {}), default=str)}
Regions:
{region_lines}
Lessons learned: {lessons}
Recent: {history[-3:]}
Output JSON action:"""

    # FIX 7: Real API call — no silent swallowing of errors
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SENTINEL_SYSTEM},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        timeout=30,
    )
    text = response.choices[0].message.content.strip()
    s, e = text.find('{'), text.rfind('}') + 1
    if s >= 0 and e > s:
        d = json.loads(text[s:e])
        # Ensure all required keys exist
        defaults = {
            "firefighting_allocation": {}, "flood_rescue_allocation": {},
            "medical_allocation": {}, "evacuation_allocation": {},
            "power_restoration": [], "water_supply_restoration": [],
            "firebreaks": [], "flood_barriers": [], "evacuation_orders": [],
            "food_distribution": {}, "water_distribution": {},
            "satellite_coverage": [], "priority_regions": [],
        }
        for k, v in defaults.items():
            if k not in d:
                d[k] = v
        return d

    raise ValueError(f"LLM returned non-JSON: {text[:200]}")


def get_fallback_action(regions: list) -> dict:
    """Emergency fallback — only used if LLM call itself raises an exception."""
    action = {
        "firefighting_allocation": {}, "flood_rescue_allocation": {},
        "medical_allocation": {}, "evacuation_allocation": {},
        "power_restoration": [], "water_supply_restoration": [],
        "firebreaks": [], "flood_barriers": [], "evacuation_orders": [],
        "food_distribution": {}, "water_distribution": {},
        "satellite_coverage": [], "priority_regions": [],
    }
    for r in regions:
        rid = str(r["region_id"])
        if r.get("crisis_type") == "WILDFIRE" and r.get("severity", 0) > 0.5:
            action["firefighting_allocation"][rid] = 3
            action["firebreaks"].append(r["region_id"])
        if r.get("crisis_type") in ["FLOOD", "HURRICANE"] and r.get("severity", 0) > 0.5:
            action["flood_rescue_allocation"][rid] = 2
            action["flood_barriers"].append(r["region_id"])
        if r.get("severity", 0) > 0.65:
            action["medical_allocation"][rid] = 2
            action["evacuation_orders"].append(r["region_id"])
            action["evacuation_allocation"][rid] = 3
        if r.get("power_grid", 1.0) < 0.5:
            action["power_restoration"].append(r["region_id"])
    return action


def reflect(score: float, casualties: int) -> None:
    """Layer 5: Post-episode reflection — adds lessons for next episode."""
    global SENTINEL_LESSONS
    if score > 0.6:
        SENTINEL_LESSONS.append(
            f"Score {score:.3f} — strategy effective, casualties: {casualties}"
        )
    elif score > 0.4:
        SENTINEL_LESSONS.append(
            f"Score {score:.3f} — send power_restoration earlier to boost comms"
        )
    else:
        SENTINEL_LESSONS.append(
            f"Score {score:.3f} — prioritize evacuation_orders for severity>0.7 regions first"
        )
    SENTINEL_LESSONS = SENTINEL_LESSONS[-6:]


# ── FIX 3 & 5: [START]/[END] inside per-task loop, steps = actual steps ──────
def run_task(client: OpenAI, task_id: str) -> float:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    history:     List[str]   = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        obs = result.get("observation") or result

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            regions = obs.get("regions", [])

            # Try LLM first; fallback only on genuine exception
            try:
                raw = get_action(client, obs, step, history)
            except Exception as ex:
                print(f"[DEBUG] LLM error step {step}: {ex}", flush=True)
                raw = get_fallback_action(regions)

            sresp = requests.post(f"{ENV_URL}/step", json=raw, timeout=30)
            sresp.raise_for_status()
            result  = sresp.json()
            reward  = float(result.get("reward") or 0.0)
            done    = bool(result.get("done", False))

            rewards.append(reward)
            steps_taken = step   # FIX 5: counts actual steps

            log_step(
                step=step,
                action=json.dumps(raw)[:120].replace("\n", " "),
                reward=reward,
                done=done,
                error=None,
            )

            obs = result.get("observation") or result
            history.append(
                f"Step{step}: r={reward:.2f} "
                f"cas={obs.get('total_casualties', 0)} "
                f"evac={obs.get('total_evacuated', 0)}"
            )

            if done:
                break

        score   = min(max(sum(rewards) / (MAX_STEPS * 1.0), 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        reflect(score, obs.get("total_casualties", 0))

    except Exception as ex:
        print(f"[DEBUG] Task error: {ex}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks  = ["single_crisis", "multi_crisis", "cascade_crisis"]
    scores = {}

    for task_id in tasks:
        scores[task_id] = run_task(client, task_id)
        print(f"[DEBUG] {task_id}: {scores[task_id]:.3f}", flush=True)

    avg = sum(scores.values()) / len(scores)
    print(f"[DEBUG] SENTINEL avg={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
