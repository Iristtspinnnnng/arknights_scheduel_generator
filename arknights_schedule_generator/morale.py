from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import Layout, ShiftPlan
from .skill_rules import fatigue_delta_from_text


MAX_MORALE = 24.0
MAX_CYCLE_HOURS = 24.0 * 7.0

#: Morale values closer than this are treated as identical.
MORALE_STATE_TOLERANCE = 0.0001
#: A cycle is sustainable when no operator loses more morale than it recovers.
MORALE_BALANCE_TOLERANCE = 0.0001
#: Bound for the (diagnostic only) steady-state replay.
MAX_STEADY_STATE_ITERATIONS = 16

AUDIT_VERSION = 2


def active_operator_names(shift: ShiftPlan) -> set[str]:
    return {
        skill.operator_name
        for room in shift.rooms
        for skill in room.operators
    }


def dormitory_operator_names(shift: ShiftPlan) -> set[str]:
    return {
        skill.operator_name
        for room in shift.dormitories
        for skill in room.operators
    }


def audit_morale_cycle(layout: Layout, shifts: list[ShiftPlan]) -> dict[str, Any]:
    """Audit whether repeating ``shifts`` keeps every operator's morale sustainable.

    The verdict is a **per-operator morale balance sheet**: over one full cycle,
    accumulate the morale an operator spends (time on duty plus parsed consumption
    skills) and the morale it can recover (dormitory base rate plus parsed recovery
    bonuses), and require ``recovery - consumption >= 0`` for everybody.

    That accounting is what makes the audit robust:

    * The ``MAX_MORALE`` cap throws recovery away whenever an operator is already
      full, so "morale at the end of the cycle vs. morale at the start" is *not* a
      sustainability criterion. Replaying the cycle until a fixed point does not
      rescue it either: the cap makes the replay non-contractive, and a plan that
      merely wastes recovery can look like a plan that loses morale.
    * Accounting is closed-form and deterministic: (almost) no iteration, no
      dependence on the state a replay happens to end in, and no unbounded drain.
      A drain shows up as a negative balance instead of ``minimumMorale = -180``.

    Recovery is booked as *potential* recovery (rate x hours), regardless of the
    cap, so an operator that rests while full is not punished for the wasted
    recovery. The reported ``minimumMoraleByOperator`` / ``endingMoraleByOperator``
    come from a single deterministic cycle that starts with everybody at
    ``MAX_MORALE``; for a sustainable cycle that replay starts from the highest
    possible state, so it is the most favourable (bounded) reading.

    ``steadyState`` is a bounded fixed-point replay kept for diagnostics only. It
    never gates the verdict: coupled mechanics (for example dormitory skills that
    target the lowest-morale neighbour) can make the replay orbit forever without
    making the cycle unsustainable.
    """
    if not shifts:
        return _failed_audit("empty_schedule", [], 0.0, [])

    cycle_hours = sum(float(shift.duration_hours) for shift in shifts)
    active_by_shift = [active_operator_names(shift) for shift in shifts]
    dorm_by_shift = [dormitory_operator_names(shift) for shift in shifts]
    all_names = set().union(*active_by_shift, *dorm_by_shift)

    start_morale = {name: MAX_MORALE for name in all_names}
    trace = _simulate_cycle(layout, shifts, start_morale)
    balance = {
        name: trace.recovery.get(name, 0.0) - trace.consumption.get(name, 0.0)
        for name in all_names
    }

    transitions: list[dict[str, Any]] = []
    unrested: set[str] = set()
    conflicts: set[str] = set()

    for index, shift in enumerate(shifts):
        active = active_by_shift[index]
        dorm = dorm_by_shift[index]
        outgoing = active_by_shift[index - 1] - active
        missing_rest = outgoing - dorm
        duplicate = active & dorm
        unrested.update(missing_rest)
        conflicts.update(duplicate)

        transitions.append(
            {
                "shift": shift.name,
                "start": shift.start,
                "durationHours": round(float(shift.duration_hours), 6),
                "activeCount": len(active),
                "outgoingCount": len(outgoing),
                "dormitoryOccupancy": len(dorm),
                "requiredRestCount": len(outgoing),
                "missingRestOperators": sorted(missing_rest),
                "activeDormitoryConflicts": sorted(duplicate),
            }
        )

    depleted = {
        name
        for name, value in trace.minimum.items()
        if value < -MORALE_STATE_TOLERANCE
    }
    balance_deficit = {
        name
        for name, value in balance.items()
        if value < -MORALE_BALANCE_TOLERANCE
    }
    unsupported = unrested | depleted | balance_deficit
    hard_gate = (
        cycle_hours <= MAX_CYCLE_HOURS + 0.0001
        and not unsupported
        and not conflicts
    )

    failure_reasons: list[str] = []
    if cycle_hours > MAX_CYCLE_HOURS + 0.0001:
        failure_reasons.append("cycle_exceeds_seven_days")
    if any(item["missingRestOperators"] for item in transitions):
        failure_reasons.append("required_rest_exceeds_dormitory_assignment")
    if conflicts:
        failure_reasons.append("operator_active_and_resting_in_same_shift")
    if depleted:
        failure_reasons.append("operator_morale_depleted")
    if balance_deficit:
        failure_reasons.append("morale_balance_deficit")

    return {
        "version": AUDIT_VERSION,
        "hardGatePassed": hard_gate,
        "criterion": "per_operator_morale_balance",
        "cycleHours": round(cycle_hours, 6),
        "cycleDays": round(cycle_hours / 24.0, 6),
        "planCount": len(shifts),
        "maxMorale": MAX_MORALE,
        "minimumMorale": round(min(trace.minimum.values(), default=MAX_MORALE), 6),
        "minimumMoraleByOperator": {
            name: round(value, 6) for name, value in sorted(trace.minimum.items())
        },
        "endingMoraleByOperator": {
            name: round(value, 6) for name, value in sorted(trace.morale.items())
        },
        "moraleLedgerByOperator": {
            name: {
                "consumption": round(trace.consumption.get(name, 0.0), 6),
                "recovery": round(trace.recovery.get(name, 0.0), 6),
                "appliedRecovery": round(trace.applied_recovery.get(name, 0.0), 6),
                "balance": round(balance[name], 6),
            }
            for name in sorted(all_names)
        },
        "moraleBalanceDeficitOperators": sorted(balance_deficit),
        "balanceTolerance": MORALE_BALANCE_TOLERANCE,
        "unrestedOperators": sorted(unsupported),
        "failureReasons": failure_reasons,
        "transitions": transitions,
        "steadyState": _steady_state(layout, shifts, start_morale),
        "assumptions": [
            "Working operators consume one morale per hour plus parsed positive skill consumption.",
            "Dormitory recovery uses the game-data level base rate (1.6 through 2.0 morale per hour).",
            "Parsed self, single-target, and room-wide recovery bonuses are counted; other bonuses are excluded.",
            "A cycle is sustainable when every operator's potential recovery covers its consumption.",
            "Recovery booked while an operator is already at 24 morale is counted as potential recovery only.",
            "Reported minimum and ending morale come from one cycle that starts at 24 morale for everyone.",
        ],
    }


@dataclass
class _CycleTrace:
    """Result of replaying one full cycle of ``shifts``."""

    morale: dict[str, float]
    minimum: dict[str, float]
    consumption: dict[str, float]
    #: Potential recovery: rate x hours, ignoring the 24 morale cap.
    recovery: dict[str, float]
    #: Recovery that actually landed on the morale value (the cap removes the rest).
    applied_recovery: dict[str, float]


def _simulate_cycle(
    layout: Layout,
    shifts: list[ShiftPlan],
    start_morale: dict[str, float],
) -> _CycleTrace:
    morale = dict(start_morale)
    minimum = dict(start_morale)
    trace = _CycleTrace(
        morale=morale,
        minimum=minimum,
        consumption={name: 0.0 for name in start_morale},
        recovery={name: 0.0 for name in start_morale},
        applied_recovery={name: 0.0 for name in start_morale},
    )
    for shift in shifts:
        hours = float(shift.duration_hours)
        for room in shift.rooms:
            for skill in room.operators:
                name = skill.operator_name
                extra_consumption = max(0.0, fatigue_delta_from_text(skill.description))
                consumed = (1.0 + extra_consumption) * hours
                morale[name] = morale.get(name, MAX_MORALE) - consumed
                trace.consumption[name] = trace.consumption.get(name, 0.0) + consumed
        for room_index, room in enumerate(shift.dormitories):
            room_level = (
                layout.dormitory_levels[room_index]
                if room_index < len(layout.dormitory_levels)
                else 5
            )
            all_bonus, self_bonuses, single_bonus = dormitory_recovery_bonuses(room)
            single_target = dormitory_single_recovery_target(room, morale, single_bonus)
            for skill in room.operators:
                name = skill.operator_name
                recovery_rate = (
                    dormitory_base_recovery_rate(room_level)
                    + all_bonus
                    + self_bonuses.get(name, 0.0)
                    + (single_bonus if name == single_target else 0.0)
                )
                potential = recovery_rate * hours
                current = morale.get(name, MAX_MORALE)
                updated = min(MAX_MORALE, current + potential)
                morale[name] = updated
                trace.recovery[name] = trace.recovery.get(name, 0.0) + potential
                trace.applied_recovery[name] = (
                    trace.applied_recovery.get(name, 0.0) + (updated - current)
                )
        for name, value in morale.items():
            minimum[name] = min(minimum.get(name, value), value)
    return trace


def _steady_state(
    layout: Layout,
    shifts: list[ShiftPlan],
    start_morale: dict[str, float],
) -> dict[str, Any]:
    """Bounded fixed-point replay, reported as a diagnostic and never used as a gate."""
    state = dict(start_morale)
    residual = 0.0
    iterations = 0
    converged = False
    for iterations in range(1, MAX_STEADY_STATE_ITERATIONS + 1):
        previous = dict(state)
        state = _simulate_cycle(layout, shifts, state).morale
        residual = max(
            (abs(state[name] - previous[name]) for name in state),
            default=0.0,
        )
        if residual <= MORALE_STATE_TOLERANCE:
            converged = True
            break
    return {
        "converged": converged,
        "iterations": iterations,
        "residual": round(residual, 6),
        "maxIterations": MAX_STEADY_STATE_ITERATIONS,
        "tolerance": MORALE_STATE_TOLERANCE,
    }


def dormitory_base_recovery_rate(room_level: int) -> float:
    return 1.5 + 0.1 * min(5, max(1, int(room_level)))


def dormitory_recovery_bonuses(room: Any) -> tuple[float, dict[str, float], float]:
    all_bonus = 0.0
    single_bonus = 0.0
    self_bonuses: dict[str, float] = {}
    for skill in room.operators:
        description = re.sub(r"<[^>]+>", "", skill.description)
        all_matches = re.findall(
            r"所有干员(?:的)?心情每小时恢复(?:速度)?\+([0-9]+(?:\.[0-9]+)?)",
            description,
        )
        if all_matches:
            all_bonus = max(all_bonus, *(float(value) for value in all_matches))
        self_matches = re.findall(
            r"自身心情每小时恢复(?:速度)?\+([0-9]+(?:\.[0-9]+)?)",
            description,
        )
        if self_matches:
            self_bonuses[skill.operator_name] = max(float(value) for value in self_matches)
        single_matches = re.findall(
            r"(?:某一名|前一位).*?心情每小时恢复(?:速度)?\+([0-9]+(?:\.[0-9]+)?)",
            description,
        )
        if single_matches:
            single_bonus = max(single_bonus, *(float(value) for value in single_matches))
    return all_bonus, self_bonuses, single_bonus


def dormitory_single_recovery_target(
    room: Any,
    morale: dict[str, float],
    single_bonus: float,
) -> str | None:
    if single_bonus <= 0.0:
        return None
    candidates = [skill.operator_name for skill in room.operators]
    return min(candidates, key=lambda name: morale.get(name, MAX_MORALE), default=None)


def _failed_audit(
    reason: str,
    unrested: list[str],
    cycle_hours: float,
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": AUDIT_VERSION,
        "hardGatePassed": False,
        "criterion": "per_operator_morale_balance",
        "cycleHours": round(cycle_hours, 6),
        "cycleDays": round(cycle_hours / 24.0, 6),
        "planCount": len(transitions),
        "maxMorale": MAX_MORALE,
        "minimumMorale": None,
        "minimumMoraleByOperator": {},
        "endingMoraleByOperator": {},
        "moraleLedgerByOperator": {},
        "moraleBalanceDeficitOperators": [],
        "balanceTolerance": MORALE_BALANCE_TOLERANCE,
        "unrestedOperators": sorted(unrested),
        "failureReasons": [reason],
        "transitions": transitions,
        "steadyState": {
            "converged": False,
            "iterations": 0,
            "residual": None,
            "maxIterations": MAX_STEADY_STATE_ITERATIONS,
            "tolerance": MORALE_STATE_TOLERANCE,
        },
        "assumptions": [],
    }
