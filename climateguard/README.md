---
title: ClimateGuard AI
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# ClimateGuard AI 🌍
## Multi-Crisis Climate Response Coordination Environment

> An RL environment for training AI agents to coordinate multi-agency climate
> crisis response — wildfires, floods, hurricanes, and cascading disasters.

## Motivation

Climate disasters are intensifying. In 2023 alone, wildfires burned 10M+ hectares and
floods displaced millions. Effective AI-assisted coordination can dramatically reduce
casualties by optimizing resource allocation under time pressure and incomplete information.

## Tasks

| Task | Difficulty | Description |
|---|---|---|
| single_crisis | Easy | Coordinate wildfire response across 5 regions |
| multi_crisis | Medium | Simultaneous wildfire + flood + drought across 10 regions |
| cascade_crisis | Hard | Hurricane triggers flood which blocks roads — adapt in real-time |

## Action Space

| Field | Type | Description |
|---|---|---|
| firefighting_allocation | Dict[str,int] | Firefighting units per region e.g. `{"0":3}` |
| flood_rescue_allocation | Dict[str,int] | Flood rescue teams per region |
| medical_allocation | Dict[str,int] | Medical teams per region |
| evacuation_allocation | Dict[str,int] | Evacuation buses per region |
| evacuation_orders | List[int] | Region IDs to issue evacuation orders |
| power_restoration | List[int] | Region IDs to restore power grid |
| flood_barriers | List[int] | Region IDs to build flood barriers |
| firebreaks | List[int] | Region IDs to create firebreaks |
| priority_regions | List[int] | High-priority region IDs |

## Observation Space

| Field | Type | Description |
|---|---|---|
| day | int | Current response day |
| regions | List[dict] | Region states (severity, population, crisis_type, road_access) |
| resources | dict | Available units (firefighting, medical, buses, helicopters) |
| total_casualties | int | Cumulative casualties this episode |
| total_evacuated | int | Cumulative evacuations this episode |
| survival_rate | float | Population survival rate 0.0–1.0 |
| weather_forecast | str | Current weather conditions |
| done | bool | Episode complete |
| reward | float | Step reward 0.0–1.0 |

## Reward Functions (task-specific graders)

### Task 1 — single_crisis
| Component | Weight | Description |
|---|---|---|
| Wildfire suppression | 50% | Reduction in average fire severity |
| Lives saved | 30% | People saved vs total at risk |
| Evacuation | 20% | Completeness of evacuation |

### Task 2 — multi_crisis
| Component | Weight | Description |
|---|---|---|
| Lives saved | 40% | Overall lives saved |
| Equity across crisis types | 25% | Responded to all crisis types, not just one |
| Casualty prevention | 20% | Lower total casualties = higher score |
| Resource efficiency | 15% | Units dispatched vs regions covered |

### Task 3 — cascade_crisis
| Component | Weight | Description |
|---|---|---|
| Lives saved under cascade | 35% | Lives saved despite cascading failures |
| Infrastructure resilience | 25% | Power restored, comms maintained |
| Adaptation | 25% | Used varied response types (fire+flood+evac+power) |
| Survival rate | 15% | Final population survival rate |

## Baseline Scores

| Agent | single_crisis | multi_crisis | cascade_crisis |
|---|---|---|---|
| Random | 0.18 | 0.15 | 0.11 |
| Greedy heuristic | 0.49 | 0.42 | 0.34 |
| Human expert | 0.81 | 0.76 | 0.68 |
| SENTINEL (our agent) | ~0.55 | ~0.48 | ~0.41 |

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Run Agent

```bash
export HF_TOKEN=hf_your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=https://your-space.hf.space
python inference.py
```
