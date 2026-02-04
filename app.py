from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

DEFAULT_EVENT_PROFILE = {
    "demand": {"health": 1.0, "mobility": 1.0, "energy": 1.0},
    "capacity_penalty": {"health": 0.0, "mobility": 0.0, "energy": 0.0},
    "route_penalty": 0.0,
    "stock_use": 1.0,
    "co2": 1.0,
}

EVENT_PROFILES: dict[str, dict[str, Any]] = {
    "Greve transport": {
        "demand": {"health": 1.05, "mobility": 1.42, "energy": 1.02},
        "capacity_penalty": {"health": 0.02, "mobility": 0.22, "energy": 0.03},
        "route_penalty": 0.26,
        "stock_use": 1.08,
        "co2": 1.12,
    },
    "Canicule intense": {
        "demand": {"health": 1.24, "mobility": 1.08, "energy": 1.33},
        "capacity_penalty": {"health": 0.05, "mobility": 0.03, "energy": 0.09},
        "route_penalty": 0.08,
        "stock_use": 1.17,
        "co2": 1.18,
    },
    "Cyberattaque municipale": {
        "demand": {"health": 1.12, "mobility": 1.16, "energy": 1.11},
        "capacity_penalty": {"health": 0.08, "mobility": 0.12, "energy": 0.14},
        "route_penalty": 0.17,
        "stock_use": 1.05,
        "co2": 1.06,
    },
    "Afflux aux urgences": {
        "demand": {"health": 1.45, "mobility": 1.04, "energy": 1.02},
        "capacity_penalty": {"health": 0.14, "mobility": 0.01, "energy": 0.02},
        "route_penalty": 0.11,
        "stock_use": 1.24,
        "co2": 1.03,
    },
    "Inondation locale": {
        "demand": {"health": 1.12, "mobility": 1.22, "energy": 1.14},
        "capacity_penalty": {"health": 0.06, "mobility": 0.17, "energy": 0.08},
        "route_penalty": 0.32,
        "stock_use": 1.1,
        "co2": 1.09,
    },
    "Panne reseau electrique": {
        "demand": {"health": 1.08, "mobility": 1.0, "energy": 1.38},
        "capacity_penalty": {"health": 0.03, "mobility": 0.04, "energy": 0.26},
        "route_penalty": 0.06,
        "stock_use": 1.09,
        "co2": 1.15,
    },
    "Subvention exceptionnelle": {
        "demand": {"health": 0.98, "mobility": 0.97, "energy": 0.96},
        "capacity_penalty": {"health": -0.04, "mobility": -0.04, "energy": -0.04},
        "route_penalty": -0.04,
        "stock_use": 0.95,
        "co2": 0.92,
    },
    "Innovation locale": {
        "demand": {"health": 0.99, "mobility": 0.98, "energy": 0.99},
        "capacity_penalty": {"health": -0.05, "mobility": -0.08, "energy": -0.05},
        "route_penalty": -0.08,
        "stock_use": 0.94,
        "co2": 0.93,
    },
    "Mobilisation citoyenne": {
        "demand": {"health": 1.0, "mobility": 0.97, "energy": 0.99},
        "capacity_penalty": {"health": -0.03, "mobility": -0.05, "energy": -0.02},
        "route_penalty": -0.06,
        "stock_use": 0.93,
        "co2": 0.9,
    },
    "Rupture fournisseur": {
        "demand": {"health": 1.09, "mobility": 1.08, "energy": 1.02},
        "capacity_penalty": {"health": 0.06, "mobility": 0.04, "energy": 0.03},
        "route_penalty": 0.05,
        "stock_use": 1.38,
        "co2": 1.05,
    },
    "Alerte pollution": {
        "demand": {"health": 1.19, "mobility": 1.15, "energy": 1.07},
        "capacity_penalty": {"health": 0.06, "mobility": 0.12, "energy": 0.04},
        "route_penalty": 0.14,
        "stock_use": 1.11,
        "co2": 1.26,
    },
    "Semaine calme": {
        "demand": {"health": 0.93, "mobility": 0.93, "energy": 0.95},
        "capacity_penalty": {"health": -0.02, "mobility": -0.02, "energy": -0.02},
        "route_penalty": -0.02,
        "stock_use": 0.9,
        "co2": 0.88,
    },
}

STRATEGY_MODIFIERS = {
    "Conservateur": {"capacity": 0.94, "cost": 0.9, "co2": 0.91, "risk": 0.94},
    "Equilibre": {"capacity": 1.0, "cost": 1.0, "co2": 1.0, "risk": 1.0},
    "Agressif": {"capacity": 1.12, "cost": 1.17, "co2": 1.14, "risk": 1.09},
}

PRIORITY_MODIFIERS = {
    "Cout": {"capacity": 0.95, "cost": 0.88, "equity_boost": 0.0},
    "Service": {"capacity": 1.08, "cost": 1.06, "equity_boost": 0.05},
    "Equite": {"capacity": 0.98, "cost": 1.02, "equity_boost": 0.28},
    "Resilience": {"capacity": 1.03, "cost": 1.08, "equity_boost": 0.12},
}

ROUTING_MODIFIERS = {
    "Plus court chemin": {"time": 0.9, "reliability": -0.04, "equity": -0.04},
    "Equite prioritaire": {"time": 1.03, "reliability": -0.01, "equity": 0.22},
    "Robuste multi-routes": {"time": 0.97, "reliability": 0.08, "equity": 0.08},
}

KEYWORD_BONUS = {
    "variable": 6,
    "variables": 6,
    "objectif": 8,
    "contrainte": 6,
    "contraintes": 6,
    "hypothese": 5,
    "hypotheses": 5,
    "robuste": 6,
    "scenario": 4,
    "incertitude": 4,
}

EMPTY_EVENT = {
    "name": "Pas de choc",
    "description": "Aucun evenement majeur sur ce tour.",
    "impact": {"performance": 0, "robustness": 0, "social": 0},
    "severity_label": "Neutre",
    "severity_factor": 1.0,
    "target_districts": [],
    "profile": DEFAULT_EVENT_PROFILE,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp_score(value: float) -> int:
    return int(round(clamp_float(value, 0, 100)))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = clamp_float(q, 0.0, 1.0) * (len(ordered) - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return ordered[low]
    share = idx - low
    return ordered[low] + (ordered[high] - ordered[low]) * share


def normalize_allocations(raw: dict[str, int]) -> tuple[dict[str, float], int]:
    total = sum(raw.values())
    if total <= 0:
        return {key: 0.0 for key in raw}, 0
    return {key: raw[key] / total for key in raw}, total


def role_weights(selected_role: str, roles: list[dict[str, Any]]) -> dict[str, int]:
    for role in roles:
        if role["name"] == selected_role:
            return role["weights"]
    return {"performance": 1, "robustness": 1, "clarity": 1, "social": 1}


def pick_mission(modules: list[dict[str, Any]], module_name: str) -> tuple[str, str]:
    module = next((m for m in modules if m["name"] == module_name), modules[0])
    return module["name"], random.choice(module["missions"])


def build_event_card(raw_event: dict[str, Any], district_ids: list[str]) -> dict[str, Any]:
    severity_draw = random.choices(
        [("Mineur", 0.85), ("Modere", 1.0), ("Majeur", 1.3)],
        weights=[0.25, 0.55, 0.2],
        k=1,
    )[0]
    severity_label, severity_factor = severity_draw
    target_count = 1 if severity_label == "Mineur" else 2 if severity_label == "Modere" else 3
    targets = random.sample(district_ids, k=min(target_count, len(district_ids)))

    profile = EVENT_PROFILES.get(raw_event["name"], DEFAULT_EVENT_PROFILE)
    impact = {
        key: int(round(raw_event["impact"][key] * severity_factor))
        for key in ("performance", "robustness", "social")
    }

    return {
        "name": raw_event["name"],
        "description": raw_event["description"],
        "impact": impact,
        "severity_label": severity_label,
        "severity_factor": severity_factor,
        "target_districts": targets,
        "profile": profile,
    }


def build_adjacency(
    edges: list[dict[str, Any]],
    impacted_nodes: set[str],
    severity_factor: float,
    route_penalty: float,
    routing_mode: str,
    dispatch_ratio: float,
) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = {}
    routing_mod = ROUTING_MODIFIERS[routing_mode]
    for edge in edges:
        src = edge["src"]
        dst = edge["dst"]
        base_time = float(edge["time"])

        local_penalty = 1.0
        if src in impacted_nodes or dst in impacted_nodes:
            local_penalty += route_penalty * severity_factor
        local_penalty *= routing_mod["time"]
        local_penalty *= max(0.78, 1.0 - dispatch_ratio * 0.18)

        adjusted_time = max(1.0, base_time * local_penalty)
        adjacency.setdefault(src, []).append((dst, adjusted_time))
        adjacency.setdefault(dst, []).append((src, adjusted_time))
    return adjacency


def shortest_path(adjacency: dict[str, list[tuple[str, float]]], start: str, end: str) -> float:
    if start == end:
        return 0.0
    visited: set[str] = set()
    distances = {start: 0.0}

    while True:
        current = None
        current_distance = float("inf")
        for node, dist in distances.items():
            if node in visited:
                continue
            if dist < current_distance:
                current_distance = dist
                current = node
        if current is None:
            break
        if current == end:
            return current_distance
        visited.add(current)
        for neighbor, weight in adjacency.get(current, []):
            candidate = current_distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
    return 99.0


def compute_network_map(
    city: dict[str, Any],
    event: dict[str, Any],
    routing_mode: str,
    dispatch_ratio: float,
) -> dict[str, Any]:
    hub = city["hub"]
    districts = city["districts"]
    impacted = set(event["target_districts"])
    profile = event["profile"]

    adjacency = build_adjacency(
        edges=city["edges"],
        impacted_nodes=impacted,
        severity_factor=event["severity_factor"],
        route_penalty=profile["route_penalty"],
        routing_mode=routing_mode,
        dispatch_ratio=dispatch_ratio,
    )

    route_times: dict[str, float] = {}
    for district in districts:
        district_id = district["id"]
        route_times[district_id] = shortest_path(adjacency, hub, district_id)

    routing_mod = ROUTING_MODIFIERS[routing_mode]
    reliability = 0.84
    reliability -= profile["route_penalty"] * 0.22 * event["severity_factor"]
    reliability += routing_mod["reliability"]
    reliability += dispatch_ratio * 0.1
    reliability = clamp_float(reliability, 0.4, 0.98)

    average_time = mean(
        [time for district, time in route_times.items() if district != hub]
    )
    return {
        "times": route_times,
        "avg_time": average_time,
        "reliability": reliability,
    }


def evaluate_clarity(model_text: str) -> tuple[int, list[str]]:
    stripped = model_text.strip()
    if not stripped:
        return 0, []

    text = stripped.lower()
    words = stripped.split()
    score = 22
    score += min(20, len(words) // 5)
    if any(symbol in text for symbol in ("<=", ">=", "=", "min", "max")):
        score += 8
    if "\n" in stripped:
        score += 4

    matched: list[str] = []
    for keyword, pts in KEYWORD_BONUS.items():
        if keyword in text:
            matched.append(keyword)
            score += pts
    return clamp_score(score), sorted(set(matched))


def run_simulation(
    city: dict[str, Any],
    system_state: dict[str, float],
    event: dict[str, Any],
    decision: dict[str, Any],
    scenarios: int,
) -> dict[str, Any]:
    districts = city["districts"]
    profile = event["profile"]
    allocations = decision["allocations"]
    strategy_mod = STRATEGY_MODIFIERS[decision["strategy"]]
    priority_mod = PRIORITY_MODIFIERS[decision["priority"]]
    routing_mod = ROUTING_MODIFIERS[decision["routing_mode"]]
    dispatch_ratio = decision["dispatch"] / 100.0

    network = compute_network_map(
        city=city,
        event=event,
        routing_mode=decision["routing_mode"],
        dispatch_ratio=dispatch_ratio,
    )

    total_population = sum(d["population"] for d in districts)
    population_share = {
        district["id"]: district["population"] / total_population for district in districts
    }

    demand_weights = {}
    service_weights = {}
    for district in districts:
        district_id = district["id"]
        vuln = district["vulnerability"]
        pop_share = population_share[district_id]
        demand_weight = pop_share * (1.0 + vuln * 0.44)
        if district_id in event["target_districts"]:
            demand_weight *= 1.0 + 0.12 * event["severity_factor"]
        demand_weights[district_id] = demand_weight

        access = max(0.35, 1.0 - network["times"][district_id] / 42.0)
        service_weight = pop_share * access
        service_weight *= 1.0 + priority_mod["equity_boost"] * vuln
        service_weight *= 1.0 + routing_mod["equity"] * vuln
        service_weights[district_id] = max(0.01, service_weight)

    demand_norm, _ = normalize_allocations(
        {key: int(value * 10000) for key, value in demand_weights.items()}
    )
    service_norm, _ = normalize_allocations(
        {key: int(value * 10000) for key, value in service_weights.items()}
    )

    scenario_rows: list[dict[str, float]] = []
    district_accumulator: dict[str, dict[str, float]] = {
        district["id"]: {"demand": 0.0, "service": 0.0, "risk": 0.0, "wait": 0.0}
        for district in districts
    }

    unmet_rates: list[float] = []
    waits: list[float] = []
    costs: list[float] = []
    co2_values: list[float] = []
    risks: list[float] = []
    stock_uses: list[float] = []
    health_unmet_values: list[float] = []
    mobility_unmet_values: list[float] = []
    energy_unmet_values: list[float] = []
    vulnerable_unmet: list[float] = []
    non_vulnerable_unmet: list[float] = []

    for run_id in range(1, scenarios + 1):
        global_shock = random.uniform(0.88, 1.18) * event["severity_factor"]
        health_demand = (
            210 + system_state["health_backlog"] * 1.45
        ) * profile["demand"]["health"] * global_shock
        mobility_demand = (
            185 + system_state["mobility_backlog"] * 1.35
        ) * profile["demand"]["mobility"] * random.uniform(0.9, 1.15) * global_shock
        energy_demand = (
            195 + system_state["energy_backlog"] * 1.3
        ) * profile["demand"]["energy"] * random.uniform(0.92, 1.12) * global_shock

        health_capacity = (
            165
            + allocations["health"] * 220
            + decision["temp_clinics"] * 32
            + decision["overtime"] * 2.2
        )
        mobility_capacity = (
            150
            + allocations["mobility"] * 210
            + dispatch_ratio * 70
            + decision["overtime"] * 1.4
        )
        energy_capacity = (
            160 + allocations["energy"] * 240 + allocations["climate"] * 60
        )

        health_capacity *= (
            strategy_mod["capacity"]
            * priority_mod["capacity"]
            * (1.0 - profile["capacity_penalty"]["health"] * event["severity_factor"])
        )
        mobility_capacity *= (
            strategy_mod["capacity"]
            * priority_mod["capacity"]
            * (1.0 - profile["capacity_penalty"]["mobility"] * event["severity_factor"])
        )
        energy_capacity *= (
            strategy_mod["capacity"]
            * priority_mod["capacity"]
            * (1.0 - profile["capacity_penalty"]["energy"] * event["severity_factor"])
        )

        if decision["priority"] == "Resilience":
            health_capacity += 8
            mobility_capacity += 6
            energy_capacity += 6

        health_capacity = max(25.0, health_capacity)
        mobility_capacity = max(20.0, mobility_capacity)
        energy_capacity = max(20.0, energy_capacity)

        health_unmet = max(0.0, health_demand - health_capacity)
        mobility_unmet = max(0.0, mobility_demand - mobility_capacity)
        energy_unmet = max(0.0, energy_demand - energy_capacity)

        total_demand = health_demand + mobility_demand + energy_demand
        total_unmet = health_unmet + mobility_unmet + energy_unmet
        unmet_rate = total_unmet / max(1.0, total_demand)
        service_rate = 1.0 - unmet_rate

        wait_time = (
            5.0
            + (health_unmet / max(1.0, health_capacity)) * 58.0
            + network["avg_time"] * 0.63
            + (1.0 - network["reliability"]) * 19.0
        )
        wait_time *= random.uniform(0.92, 1.1)

        cost = (
            95.0
            + health_capacity * 0.22
            + mobility_capacity * 0.18
            + energy_capacity * 0.16
            + decision["temp_clinics"] * 15.0
            + decision["overtime"] * 1.35
            + decision["stock_order"] * 0.2
            + total_unmet * 0.43
        )
        cost *= strategy_mod["cost"] * priority_mod["cost"]

        co2 = (
            18.0
            + mobility_demand * 0.085
            + energy_demand * 0.07
            + decision["overtime"] * 0.3
        )
        co2 *= strategy_mod["co2"] * profile["co2"]
        co2 *= 1.0 - allocations["climate"] * 0.16
        co2 = max(5.0, co2)

        stock_use = (
            48.0
            + health_demand * 0.11
            + decision["temp_clinics"] * 2.5
            + decision["dispatch"] * 0.04
        )
        stock_use *= profile["stock_use"] * random.uniform(0.92, 1.08)

        district_risks: list[float] = []
        for district in districts:
            district_id = district["id"]
            vulnerability = district["vulnerability"]
            district_demand = (
                health_demand
                * demand_norm[district_id]
                * random.uniform(0.93, 1.07)
            )
            district_capacity = (
                health_capacity
                * service_norm[district_id]
                * random.uniform(0.94, 1.06)
            )
            district_unmet = max(0.0, district_demand - district_capacity)
            district_unmet_ratio = district_unmet / max(1.0, district_demand)

            district_wait = (
                4.0
                + network["times"][district_id] * 0.95
                + district_unmet_ratio * 52.0
            )
            if district_id in event["target_districts"]:
                district_wait += 3.5

            district_service_score = clamp_float(
                100.0
                - district_unmet_ratio * 100.0
                - network["times"][district_id] * 0.85
                + dispatch_ratio * 18.0,
                0.0,
                100.0,
            )
            district_risk_score = clamp_float(
                15.0
                + district_unmet_ratio * 72.0
                + vulnerability * 24.0
                + (1.0 - network["reliability"]) * 22.0
                + (6.0 if district_id in event["target_districts"] else 0.0),
                0.0,
                100.0,
            )

            district_accumulator[district_id]["demand"] += district_demand
            district_accumulator[district_id]["service"] += district_service_score
            district_accumulator[district_id]["risk"] += district_risk_score
            district_accumulator[district_id]["wait"] += district_wait
            district_risks.append(district_risk_score)

            if vulnerability >= 0.65:
                vulnerable_unmet.append(district_unmet_ratio)
            else:
                non_vulnerable_unmet.append(district_unmet_ratio)

        scenario_rows.append(
            {
                "scenario": float(run_id),
                "service_rate": service_rate * 100.0,
                "cost": cost,
                "wait_time": wait_time,
                "co2": co2,
                "risk": mean(district_risks),
            }
        )
        unmet_rates.append(unmet_rate)
        waits.append(wait_time)
        costs.append(cost)
        co2_values.append(co2)
        risks.append(mean(district_risks))
        stock_uses.append(stock_use)
        health_unmet_values.append(health_unmet)
        mobility_unmet_values.append(mobility_unmet)
        energy_unmet_values.append(energy_unmet)

    service_mean = mean(row["service_rate"] for row in scenario_rows)
    service_p10 = percentile([row["service_rate"] for row in scenario_rows], 0.1)
    cost_mean = mean(costs)
    wait_mean = mean(waits)
    wait_p90 = percentile(waits, 0.9)
    co2_mean = mean(co2_values)
    risk_mean = mean(risks)
    unmet_p90 = percentile(unmet_rates, 0.9)
    if vulnerable_unmet and non_vulnerable_unmet:
        equity_gap = abs(mean(vulnerable_unmet) - mean(non_vulnerable_unmet))
    else:
        equity_gap = 0.0
    stock_pressure = max(
        0.0,
        (mean(stock_uses) - (system_state["stock"] + decision["stock_order"])) / 120.0,
    )

    performance_score = clamp_score(
        service_mean
        - max(0.0, cost_mean - 220.0) * 0.22
        - max(0.0, wait_mean - 18.0) * 0.95
        + dispatch_ratio * 6.0
    )
    robustness_score = clamp_score(
        95.0
        - unmet_p90 * 130.0
        - max(0.0, wait_p90 - wait_mean) * 0.8
        - (1.0 - network["reliability"]) * 32.0
        - stock_pressure * 34.0
        + (4.0 if decision["priority"] == "Resilience" else 0.0)
    )
    social_score = clamp_score(
        100.0
        - equity_gap * 155.0
        - max(0.0, co2_mean - 40.0) * 1.1
        - risk_mean * 0.35
        + allocations["climate"] * 18.0
    )

    district_rows: list[dict[str, Any]] = []
    for district in districts:
        district_id = district["id"]
        district_values = district_accumulator[district_id]
        district_rows.append(
            {
                "quartier": district_id,
                "population": district["population"],
                "vulnerabilite": round(district["vulnerability"], 2),
                "temps_route_min": round(network["times"][district_id], 1),
                "demande_sante": round(district_values["demand"] / scenarios, 1),
                "service_score": round(district_values["service"] / scenarios, 1),
                "risque": round(district_values["risk"] / scenarios, 1),
                "attente_min": round(district_values["wait"] / scenarios, 1),
            }
        )

    recommendations: list[str] = []
    if service_p10 < 55:
        recommendations.append(
            "Le plan est fragile sur les mauvais scenarios. Renforcer capacites de reserve."
        )
    if cost_mean > 300:
        recommendations.append(
            "Le cout moyen est tres haut. Prioriser une strategie Cout ou reduire overtime."
        )
    if equity_gap > 0.1:
        recommendations.append(
            "Ecart d'equite eleve. Basculer vers priorite Equite ou routage prioritaire."
        )
    if co2_mean > 55:
        recommendations.append(
            "CO2 en tension. Augmenter budget climat ou limiter strategie agressive."
        )
    if not recommendations:
        recommendations.append(
            "Plan globalement stable. Chercher un gain marginal via stress-tests supplementaires."
        )

    return {
        "scores": {
            "performance": performance_score,
            "robustness": robustness_score,
            "social": social_score,
        },
        "kpis": {
            "service_moyen": service_mean,
            "service_p10": service_p10,
            "cout_moyen": cost_mean,
            "attente_moyenne": wait_mean,
            "attente_p90": wait_p90,
            "co2_moyen": co2_mean,
            "equity_gap": equity_gap,
            "fiabilite_reseau": network["reliability"],
            "stock_use_moyen": mean(stock_uses),
            "health_unmet": mean(health_unmet_values),
            "mobility_unmet": mean(mobility_unmet_values),
            "energy_unmet": mean(energy_unmet_values),
            "risk_moyen": risk_mean,
        },
        "scenario_rows": scenario_rows,
        "district_rows": sorted(district_rows, key=lambda row: row["risque"], reverse=True),
        "network_rows": [
            {"quartier": district["id"], "temps_route_min": round(network["times"][district["id"]], 1)}
            for district in districts
        ],
        "recommendations": recommendations,
    }


def update_system_state(
    system_state: dict[str, float],
    simulation: dict[str, Any],
    decision: dict[str, Any],
    event: dict[str, Any],
    current_stability: int,
) -> tuple[dict[str, float], int, int]:
    scores = simulation["scores"]
    kpis = simulation["kpis"]
    new_state = dict(system_state)

    budget_delta = 10.0 - kpis["cout_moyen"] / 28.0 + event["impact"]["performance"] * 0.2
    if scores["performance"] > 75:
        budget_delta += 3.0
    new_state["budget"] = clamp_float(system_state["budget"] + budget_delta, 0.0, 450.0)

    stock_delta = decision["stock_order"] - kpis["stock_use_moyen"]
    new_state["stock"] = clamp_float(system_state["stock"] + stock_delta, 0.0, 700.0)

    trust_delta = (
        (scores["social"] - 50.0) / 10.0
        + (scores["performance"] - 50.0) / 20.0
        + event["impact"]["social"] * 0.25
    )
    new_state["trust"] = clamp_float(system_state["trust"] + trust_delta, 0.0, 100.0)

    new_state["co2"] = clamp_float(
        system_state["co2"] * 0.65 + kpis["co2_moyen"] * 0.35, 5.0, 140.0
    )
    new_state["health_backlog"] = clamp_float(
        system_state["health_backlog"] * 0.55 + kpis["health_unmet"] * 0.58, 0.0, 280.0
    )
    new_state["mobility_backlog"] = clamp_float(
        system_state["mobility_backlog"] * 0.57 + kpis["mobility_unmet"] * 0.56,
        0.0,
        280.0,
    )
    new_state["energy_backlog"] = clamp_float(
        system_state["energy_backlog"] * 0.6 + kpis["energy_unmet"] * 0.52, 0.0, 280.0
    )

    backlog_penalty = (
        new_state["health_backlog"] + new_state["mobility_backlog"] + new_state["energy_backlog"]
    ) / 18.0

    new_stability = clamp_score(
        0.24 * scores["performance"]
        + 0.28 * scores["robustness"]
        + 0.2 * scores["social"]
        + 0.18 * new_state["trust"]
        + 0.1 * (100.0 - new_state["co2"] * 0.6)
        - backlog_penalty
    )
    stability_delta = new_stability - current_stability
    return new_state, new_stability, stability_delta


def bootstrap_district_snapshot(
    city: dict[str, Any], system_state: dict[str, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for district in city["districts"]:
        vulnerability = district["vulnerability"]
        base_service = clamp_float(67.0 - vulnerability * 22.0 + system_state["trust"] * 0.15, 20.0, 95.0)
        base_risk = clamp_float(28.0 + vulnerability * 34.0, 8.0, 90.0)
        rows.append(
            {
                "quartier": district["id"],
                "population": district["population"],
                "vulnerabilite": round(vulnerability, 2),
                "temps_route_min": 0.0,
                "demande_sante": round(85.0 + vulnerability * 26.0, 1),
                "service_score": round(base_service, 1),
                "risque": round(base_risk, 1),
                "attente_min": round(8.0 + vulnerability * 10.0, 1),
            }
        )
    return rows


def init_state(modules: list[dict[str, Any]], city: dict[str, Any]) -> None:
    defaults: dict[str, Any] = {
        "turn": 1,
        "stability": 62,
        "current_module": modules[0]["name"],
        "current_mission": modules[0]["missions"][0],
        "current_event": EMPTY_EVENT,
        "last_result": None,
        "history": [],
        "system_state": dict(city["base_system"]),
        "district_snapshot": bootstrap_district_snapshot(city, city["base_system"]),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_campaign(modules: list[dict[str, Any]], city: dict[str, Any]) -> None:
    st.session_state.turn = 1
    st.session_state.stability = 62
    st.session_state.current_module = modules[0]["name"]
    st.session_state.current_mission = modules[0]["missions"][0]
    st.session_state.current_event = EMPTY_EVENT
    st.session_state.last_result = None
    st.session_state.history = []
    st.session_state.system_state = dict(city["base_system"])
    st.session_state.district_snapshot = bootstrap_district_snapshot(city, city["base_system"])


def main() -> None:
    st.set_page_config(page_title="OPTIMAX Simulator", page_icon=":city_sunrise:", layout="wide")

    modules = load_json(DATA_DIR / "modules.json")
    events = load_json(DATA_DIR / "events.json")
    roles = load_json(DATA_DIR / "roles.json")
    city = load_json(DATA_DIR / "city.json")
    init_state(modules, city)

    role_names = [role["name"] for role in roles]
    module_names = [module["name"] for module in modules]
    district_ids = [district["id"] for district in city["districts"]]

    st.title("OPTIMAX // Super Simulateur OR")
    st.caption(
        "Digital twin + Monte Carlo + arbitrages politiques. "
        "Formuler, decider, stresser, expliquer."
    )

    with st.sidebar:
        st.header("Controle de mission")
        team_name = st.text_input("Equipe", value="Equipe Alpha")
        role_index = 0
        selected_role = st.selectbox("Mandat", options=role_names, index=role_index)

        current_module = st.session_state.current_module
        module_index = module_names.index(current_module) if current_module in module_names else 0
        selected_module = st.selectbox("Module OR", options=module_names, index=module_index)
        scenario_count = st.slider("Scenarios Monte Carlo", 30, 300, 90, 10)

        if st.button("Nouvelle mission", use_container_width=True, type="primary"):
            module_name, mission = pick_mission(modules, selected_module)
            st.session_state.current_module = module_name
            st.session_state.current_mission = mission
            st.session_state.last_result = None

        if st.button("Tirer evenement", use_container_width=True):
            st.session_state.current_event = build_event_card(random.choice(events), district_ids)

        if st.button("Reset campagne", use_container_width=True):
            reset_campaign(modules, city)

    system_state = st.session_state.system_state
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Tour", st.session_state.turn)
    m2.metric("Stabilite", f"{st.session_state.stability}/100")
    m3.metric("Budget", f"{system_state['budget']:.1f}")
    m4.metric("Stock critique", f"{system_state['stock']:.1f}")
    m5.metric("Confiance", f"{system_state['trust']:.1f}/100")
    m6.metric("CO2 indice", f"{system_state['co2']:.1f}")

    tab_sim, tab_twin, tab_history, tab_rules = st.tabs(
        ["Simulation", "Digital Twin", "Historique", "Regles"]
    )

    with tab_sim:
        st.markdown("### 1) Brief de mission")
        st.info(
            f"Module: **{st.session_state.current_module}**\n\n"
            f"Mission: {st.session_state.current_mission}"
        )

        st.markdown("### 2) Choc de la ville")
        event = st.session_state.current_event
        st.warning(
            f"Carte evenement: **{event['name']}** ({event['severity_label']})\n\n"
            f"{event['description']}\n\n"
            f"Quartiers cibles: {', '.join(event['target_districts']) if event['target_districts'] else 'Aucun'}\n\n"
            "Effets macro: "
            f"performance {event['impact']['performance']:+d}, "
            f"robustesse {event['impact']['robustness']:+d}, "
            f"social {event['impact']['social']:+d}"
        )

        st.markdown("### 3) Decision OR")
        with st.form("simulation_form", clear_on_submit=False):
            a1, a2 = st.columns(2)
            budget_health = a1.slider("Budget Sante (%)", 0, 100, 30, 5)
            budget_mobility = a1.slider("Budget Mobilite (%)", 0, 100, 25, 5)
            budget_energy = a2.slider("Budget Energie (%)", 0, 100, 25, 5)
            budget_climate = a2.slider("Budget Climat (%)", 0, 100, 20, 5)

            raw_alloc = {
                "health": budget_health,
                "mobility": budget_mobility,
                "energy": budget_energy,
                "climate": budget_climate,
            }
            allocations, alloc_total = normalize_allocations(raw_alloc)
            st.caption(
                "Repartition normalisee: "
                f"Sante {allocations['health']:.0%}, "
                f"Mobilite {allocations['mobility']:.0%}, "
                f"Energie {allocations['energy']:.0%}, "
                f"Climat {allocations['climate']:.0%}"
            )

            b1, b2, b3 = st.columns(3)
            dispatch = b1.slider("Intensite dispatch ambulances", 0, 100, 55, 5)
            temp_clinics = b2.slider("Centres temporaires ouverts", 0, 4, 1, 1)
            overtime = b3.slider("Heures sup equipes (unites)", 0, 40, 12, 2)

            c1, c2, c3 = st.columns(3)
            stock_order = c1.slider("Commande stock critique", 0, 320, 90, 10)
            strategy = c2.selectbox("Style de pilotage", options=list(STRATEGY_MODIFIERS))
            routing_mode = c3.selectbox("Strategie de routage", options=list(ROUTING_MODIFIERS))
            priority = st.selectbox("Priorite politique", options=list(PRIORITY_MODIFIERS))

            model_text = st.text_area(
                "Modele propose (variables, objectif, contraintes, hypotheses)",
                height=130,
                placeholder=(
                    "Ex: variables x_ij, y_j. "
                    "Objectif minimiser cout + penalite attente. "
                    "Contraintes capacite, budget, equite."
                ),
            )
            submitted = st.form_submit_button("Simuler le tour")

        if submitted:
            if alloc_total == 0:
                st.error("La somme des budgets doit etre > 0.")
            else:
                decision = {
                    "allocations": allocations,
                    "dispatch": dispatch,
                    "dispatch_ratio": dispatch / 100.0,
                    "temp_clinics": temp_clinics,
                    "overtime": overtime,
                    "stock_order": stock_order,
                    "strategy": strategy,
                    "routing_mode": routing_mode,
                    "priority": priority,
                }

                simulation = run_simulation(
                    city=city,
                    system_state=st.session_state.system_state,
                    event=event,
                    decision=decision,
                    scenarios=scenario_count,
                )

                clarity_score, clarity_keywords = evaluate_clarity(model_text)
                simulation["scores"]["performance"] = clamp_score(
                    simulation["scores"]["performance"] + event["impact"]["performance"] * 0.4
                )
                simulation["scores"]["robustness"] = clamp_score(
                    simulation["scores"]["robustness"] + event["impact"]["robustness"] * 0.4
                )
                simulation["scores"]["social"] = clamp_score(
                    simulation["scores"]["social"] + event["impact"]["social"] * 0.4
                )
                simulation["scores"]["clarity"] = clarity_score
                simulation["clarity_keywords"] = clarity_keywords

                weights = role_weights(selected_role, roles)
                total_score = round(
                    (
                        simulation["scores"]["performance"] * weights["performance"]
                        + simulation["scores"]["robustness"] * weights["robustness"]
                        + simulation["scores"]["clarity"] * weights["clarity"]
                        + simulation["scores"]["social"] * weights["social"]
                    )
                    / sum(weights.values()),
                    1,
                )
                simulation["total_score"] = total_score

                new_system, new_stability, stability_delta = update_system_state(
                    system_state=st.session_state.system_state,
                    simulation=simulation,
                    decision=decision,
                    event=event,
                    current_stability=st.session_state.stability,
                )

                st.session_state.system_state = new_system
                st.session_state.stability = new_stability
                st.session_state.last_result = simulation
                st.session_state.district_snapshot = simulation["district_rows"]

                st.session_state.history.append(
                    {
                        "tour": st.session_state.turn,
                        "equipe": team_name,
                        "role": selected_role,
                        "module": st.session_state.current_module,
                        "mission": st.session_state.current_mission,
                        "evenement": event["name"],
                        "severite": event["severity_label"],
                        "score_total": total_score,
                        "performance": simulation["scores"]["performance"],
                        "robustesse": simulation["scores"]["robustness"],
                        "clarte": simulation["scores"]["clarity"],
                        "social": simulation["scores"]["social"],
                        "service_moyen": round(simulation["kpis"]["service_moyen"], 1),
                        "cout_moyen": round(simulation["kpis"]["cout_moyen"], 1),
                        "attente_moyenne": round(simulation["kpis"]["attente_moyenne"], 1),
                        "equity_gap": round(simulation["kpis"]["equity_gap"], 3),
                        "stabilite": st.session_state.stability,
                    }
                )

                st.session_state.turn += 1

                st.success(
                    f"Score tour: {total_score}/100 | "
                    f"Stabilite: {st.session_state.stability}/100 ({stability_delta:+d})"
                )

        result = st.session_state.last_result
        if result:
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Performance", result["scores"]["performance"])
            s2.metric("Robustesse", result["scores"]["robustness"])
            s3.metric("Clarte", result["scores"]["clarity"])
            s4.metric("Impact social", result["scores"]["social"])
            s5.metric("Total pondere", result["total_score"])

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Service moyen", f"{result['kpis']['service_moyen']:.1f}%")
            k2.metric("Cout moyen", f"{result['kpis']['cout_moyen']:.1f}")
            k3.metric("Attente moyenne", f"{result['kpis']['attente_moyenne']:.1f} min")
            k4.metric("CO2 moyen", f"{result['kpis']['co2_moyen']:.1f}")
            k5.metric("Fiabilite reseau", f"{result['kpis']['fiabilite_reseau']:.2f}")

            if result["clarity_keywords"]:
                st.caption(
                    "Mots detectes pour la clarte du modele: "
                    + ", ".join(result["clarity_keywords"])
                )

            scenario_df = pd.DataFrame(result["scenario_rows"])
            c1, c2 = st.columns(2)
            c1.subheader("Frontiere Scenario (cout vs service)")
            c1.scatter_chart(scenario_df, x="cost", y="service_rate")

            c2.subheader("Stress test sur scenarios")
            c2.line_chart(
                scenario_df.set_index("scenario")[["service_rate", "risk"]]
            )

            st.subheader("Lecture rapide du moteur")
            for line in result["recommendations"]:
                st.write(f"- {line}")

    with tab_twin:
        st.markdown("### Digital Twin: quartiers")
        live_system = st.session_state.system_state
        district_df = pd.DataFrame(st.session_state.district_snapshot)
        if not district_df.empty:
            st.dataframe(district_df, use_container_width=True, hide_index=True)
            left, right = st.columns(2)
            left.bar_chart(district_df.set_index("quartier")[["service_score", "risque"]])
            right.bar_chart(district_df.set_index("quartier")[["attente_min"]])
        else:
            st.write("Pas de donnees district pour le moment.")

        if st.session_state.last_result:
            st.markdown("### Reseau logistique")
            network_df = pd.DataFrame(st.session_state.last_result["network_rows"])
            st.dataframe(network_df, use_container_width=True, hide_index=True)

        st.markdown("### Sante systeme")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Backlog sante", f"{live_system['health_backlog']:.1f}")
        d2.metric("Backlog mobilite", f"{live_system['mobility_backlog']:.1f}")
        d3.metric("Backlog energie", f"{live_system['energy_backlog']:.1f}")
        d4.metric("Stock restant", f"{live_system['stock']:.1f}")

    with tab_history:
        history = st.session_state.history
        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            st.line_chart(
                history_df.set_index("tour")[
                    ["score_total", "stabilite", "performance", "robustesse", "social"]
                ]
            )
            st.download_button(
                "Exporter historique CSV",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="optimax_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.write("Lance un premier tour pour remplir l'historique.")

    with tab_rules:
        st.markdown("### Regles express")
        st.write(
            "1) Choisir mission + evenement. "
            "2) Definir une politique (budget, capacites, priorite). "
            "3) Le simulateur lance un Monte Carlo et score sur 4 axes."
        )
        st.write(
            "Le score final est pondere selon le mandat (Maire, CFO, Sante, Mobilite, Climat)."
        )
        st.write(
            "L'objectif n'est pas de maximiser un seul KPI: "
            "il faut un modele lisible, performant, robuste et socialement defensible."
        )
        st.code(
            "Formulation type: min cout + penalite_attente; "
            "s.c. capacite, budget, equite, robustesse."
        )


if __name__ == "__main__":
    main()
