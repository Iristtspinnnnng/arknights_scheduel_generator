from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import Layout, ShiftPlan
from .skill_rules import self_contained_recovery_rate


MAX_MORALE = 24.0
MAX_CYCLE_HOURS = 24.0 * 7.0

#: Morale values closer than this are treated as identical.
MORALE_STATE_TOLERANCE = 0.0001
#: A cycle is sustainable when no operator loses more morale than it recovers.
MORALE_BALANCE_TOLERANCE = 0.0001
#: Bound for the (diagnostic only) steady-state replay.
MAX_STEADY_STATE_ITERATIONS = 16

#: building_data.comfortLimit.
DORMITORY_COMFORT_LIMIT = 5000.0
#: building_data.dormData.phases[].decorationLimit (1000 per level up to 5000).
DORMITORY_COMFORT_PER_LEVEL = 1000.0
#: Green text: building_data.comfortManpowerRecoverFactor is 25, i.e. comfort / (25 * 100).
#: Full comfort at level 5 is 5000 points, worth 2.0 morale per hour (0 -> 24 in 6h).
COMFORT_RECOVERY_PER_POINT = 0.0004
#: building_data.controlData.basicCostBuff is -5 (1/100 units) per stationed operator.
CONTROL_MORALE_RELIEF_PER_OPERATOR = 0.05

AUDIT_VERSION = 3


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
        "unmodelledMoraleEffects": _collect_unmodelled_morale_effects(shifts),
        "assumptions": [
            "On-duty consumption is 1 morale per hour, adjusted by self, room-wide, "
            "control-centre and base-wide morale skills, and floored at 0.",
            "Dormitory recovery uses the game-data level base rate (1.6 through 2.0 morale per hour).",
            "Dormitory recovery adds 0.0004 morale per comfort point.",
            "Full dormitory comfort is assumed: 1000 points per level, capped at 5000.",
            "Each operator stationed in the control centre lowers every on-duty operator's "
            "morale consumption by 0.05 per hour (control-centre staff included).",
            "Parsed self, single-target, and room-wide recovery bonuses are counted; other bonuses are excluded.",
            "A single-target recovery skill follows the lowest-morale roommate until it is full, "
            "then moves on; its owner is never the target.",
            "An operator whose recovery cannot come from other sources (自律) only recovers "
            "its own rate in a dormitory and never receives neighbour bonuses.",
            "A cycle is sustainable when every operator's potential recovery covers its consumption.",
            "Recovery booked while an operator is already at 24 morale is counted as potential recovery only.",
            "Reported minimum and ending morale come from one cycle that starts at 24 morale for everyone.",
            "Morale clauses the model cannot attribute are listed in unmodelledMoraleEffects and not credited.",
        ],
    }


@dataclass(frozen=True)
class MoraleEffects:
    """Scope-attributed morale effects parsed out of one on-duty skill description.

    Every value is a **consumption delta**: recovery is booked as negative
    consumption, so ``-0.25`` means "this skill saves 0.25 morale per hour".
    """

    #: Applies to the skill owner only.
    self_delta: float = 0.0
    #: Applies to everybody stationed in the skill owner's room (same-effect max).
    room_delta: float = 0.0
    #: Applies to every operator in the control centre (control-scoped skills stack).
    control_delta: float = 0.0
    #: Applies to the other control-centre operators, never to the owner.
    control_others_delta: float = 0.0
    #: Applies to every on-duty operator outside the control centre.
    other_facilities_delta: float = 0.0
    #: Applies to the whole base.
    global_delta: float = 0.0
    #: Extra recovery for every dormitory occupant (recovery, not a consumption cut).
    all_dormitories_recovery: float = 0.0
    #: Clauses the model cannot attribute; reported instead of being silently dropped.
    unmodelled: tuple[str, ...] = ()


_CLAUSE_SPLIT = re.compile(r"[，；。]")
_SENTENCE_SPLIT = re.compile(r"[；。]")
_MORALE_CLAUSE = re.compile(r"心情每小时(消耗|恢复)(?:速度)?([+-])(\d+(?:\.\d+)?)")
_FACILITY_WORDS = ("贸易站", "制造站", "发电站", "会客室", "办公室", "训练室", "加工站")
_CONDITIONAL_PARTNER = re.compile(r"当与([^，；。]{1,12}?)(?:一起|进驻)")
#: Clauses that are scoped by a counted/targeted group instead of by facility.
_COUNTED_GROUP_MARKERS = ("每个", "每名", "每有", "每人")


def parse_morale_effects(
    description: str,
    room_names: frozenset[str] | set[str] | tuple[str, ...] = (),
) -> MoraleEffects:
    """Attribute the ``心情每小时消耗/恢复`` clauses of an on-duty skill.

    Scopes are read from the clause itself (``自身`` / ``当前制造站内`` /
    ``控制中枢内`` / ``基建内`` / ``其他设施内`` / ``所有宿舍内``), because the same
    number means different things depending on the words around it. Dormitory-room
    clauses are left to :func:`dormitory_recovery_bonuses`. Anything the parser
    cannot attribute safely -- counted faction bonuses, class-restricted dormitory
    recovery, named operators, ``部分设施``, an unmet ``当与`` partner -- is returned
    in ``unmodelled`` instead of being credited to the wrong scope.
    """
    text = re.sub(r"<[^>]+>", "", description)
    if "心情每小时" not in text:
        return MoraleEffects()

    names = set(room_names)
    values: dict[str, float] = {}
    unmodelled: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        pending_condition = False
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            is_condition = "当与" in clause
            if is_condition:
                conditional = _CONDITIONAL_PARTNER.search(clause)
                unmet = bool(conditional) and conditional.group(1).strip() not in names
            else:
                unmet = pending_condition
            # A "当与 X" partner condition binds the very next clause only.
            pending_condition = is_condition and unmet
            if "心情每小时" not in clause:
                continue
            if unmet:
                unmodelled.append(clause)
                continue
            matches = list(_MORALE_CLAUSE.finditer(clause))
            if not matches:
                unmodelled.append(clause)
                continue
            for match in matches:
                action, sign, amount = match.groups()
                value = float(amount) if sign == "+" else -float(amount)
                consumption_delta = value if action == "消耗" else -value
                channel = _morale_clause_scope(clause, match.start())
                if channel is None:
                    unmodelled.append(clause)
                    continue
                if channel == "dormitory":
                    # Modelled by ``dormitory_recovery_bonuses`` for dormitory rooms.
                    continue
                if channel == "all_dormitories" and consumption_delta > 0:
                    # A dormitory-wide consumption *penalty* has no game equivalent we model.
                    unmodelled.append(clause)
                    continue
                values[channel] = values.get(channel, 0.0) + (
                    -consumption_delta if channel == "all_dormitories" else consumption_delta
                )
    return MoraleEffects(
        self_delta=values.get("self", 0.0),
        room_delta=values.get("room", 0.0),
        control_delta=values.get("control", 0.0),
        control_others_delta=values.get("control_others", 0.0),
        other_facilities_delta=values.get("other_facilities", 0.0),
        global_delta=values.get("global", 0.0),
        all_dormitories_recovery=values.get("all_dormitories", 0.0),
        unmodelled=tuple(unmodelled),
    )


def _morale_clause_scope(clause: str, clause_offset: int) -> str | None:
    """Return the scope channel of one morale clause, or ``None`` when unattributable."""
    if any(marker in clause for marker in _COUNTED_GROUP_MARKERS):
        return None
    if "部分设施" in clause:
        return None
    if "所有宿舍" in clause:
        if "所有宿舍内所有干员" in clause or "所有宿舍内干员" in clause:
            return "all_dormitories"
        return None
    if "宿舍" in clause:
        if "该宿舍内" in clause or "本宿舍" in clause or "当前宿舍" in clause:
            return "dormitory"
        return None
    if "控制中枢内" in clause:
        if "除自身以外" in clause:
            return "control_others"
        if "所有干员" in clause or "其余干员" in clause or "全体干员" in clause:
            return "control"
        return None
    if "基建内" in clause:
        return "global"
    if "其他设施内" in clause or "该设施内" in clause:
        return "other_facilities"
    if any(f"{word}内" in clause for word in _FACILITY_WORDS):
        return "room"
    if "全体" in clause:
        # Bare "全体" right after a station clause refers to that station (see 愿者上钩).
        return "room"
    leading = clause[:clause_offset]
    if "自身" in clause:
        return "self"
    if "的" in leading:
        # "丰川祥子的心情每小时恢复+0.1": a named operator, not the skill owner.
        return None
    return "self"


def _same_effect_max(values: Any) -> float:
    """Combine "同种效果取最高" values: the strongest relief, or the strongest penalty."""
    numbers = [float(value) for value in values if value]
    if not numbers:
        return 0.0
    return min(numbers) if all(value < 0.0 for value in numbers) else max(numbers)


def _collect_unmodelled_morale_effects(shifts: list[ShiftPlan]) -> list[str]:
    seen: set[str] = set()
    for shift in shifts:
        room_names = {
            skill.operator_name
            for room in shift.rooms
            for skill in room.operators
        }
        for room in list(shift.rooms) + list(shift.dormitories):
            for skill in room.operators:
                effects = parse_morale_effects(skill.description, room_names)
                for clause in effects.unmodelled:
                    seen.add(f"{skill.operator_name}：{clause}")
    return sorted(seen)


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
        room_names = {
            skill.operator_name
            for room in shift.rooms
            for skill in room.operators
        }
        effects = {
            room.room_id: [
                parse_morale_effects(skill.description, room_names)
                for skill in room.operators
            ]
            for room in shift.rooms
        }
        control_rooms = [room for room in shift.rooms if room.room_type == "CONTROL"]
        control_effects = [
            effect for room in control_rooms for effect in effects[room.room_id]
        ]
        control_staff = sum(len(room.operators) for room in control_rooms)
        control_relief = CONTROL_MORALE_RELIEF_PER_OPERATOR * control_staff
        control_delta = sum(effect.control_delta for effect in control_effects)
        control_others_total = sum(effect.control_others_delta for effect in control_effects)
        global_delta = sum(
            effect.global_delta for room in shift.rooms for effect in effects[room.room_id]
        )
        other_facilities_delta = sum(
            effect.other_facilities_delta
            for room in shift.rooms
            for effect in effects[room.room_id]
        )
        dormitory_boost = _dormitory_boost_from_effects(control_effects)

        for room in shift.rooms:
            room_effects = effects[room.room_id]
            room_delta = _same_effect_max(effect.room_delta for effect in room_effects)
            in_control = room.room_type == "CONTROL"
            for skill, effect in zip(room.operators, room_effects):
                name = skill.operator_name
                rate = 1.0 + effect.self_delta + room_delta + global_delta
                if in_control:
                    rate += control_delta + (control_others_total - effect.control_others_delta)
                else:
                    rate += other_facilities_delta
                rate = max(0.0, rate - control_relief)
                consumed = rate * hours
                morale[name] = morale.get(name, MAX_MORALE) - consumed
                trace.consumption[name] = trace.consumption.get(name, 0.0) + consumed

        for room_index, room in enumerate(shift.dormitories):
            room_level = (
                layout.dormitory_levels[room_index]
                if room_index < len(layout.dormitory_levels)
                else 5
            )
            passive = dormitory_passive_recovery_rate(room_level) + dormitory_boost
            _recover_dormitory_room(room, passive, morale, hours, trace)
        for name, value in morale.items():
            minimum[name] = min(minimum.get(name, value), value)
    return trace


def _recover_dormitory_room(
    room: Any,
    passive: float,
    morale: dict[str, float],
    hours: float,
    trace: _CycleTrace,
) -> None:
    """Replay one dormitory stay, including the sticky single-target recovery skill.

    ``除自身以外心情未满的某个干员每小时恢复+N`` picks the lowest-morale roommate
    as it is and holds it until that operator is full (then the next lowest takes
    over); the skill owner can never be the target. Everything is simulated as a
    sequence of constant-rate segments so the "until full" rule is exact.

    Operators whose recovery cannot come from other sources (``自律``) are handled
    as **self-contained nodes**: their rate is their own parsed value, the white text,
    comfort, room-wide skills and control-centre bonus are not added, and they are
    never a single-target recipient.
    """
    single = dormitory_single_target(room)
    occupants = [skill.operator_name for skill in room.operators]
    self_contained = _self_contained_recovery_rates(room)
    base_rates = dormitory_recovery_rates(room, passive)
    single_bonus, owner = single if single is not None else (0.0, None)
    candidates = [
        name for name in occupants if name != owner and name not in self_contained
    ]

    remaining = float(hours)
    while remaining > 1e-9 and single_bonus > 0.0 and candidates:
        target = _lowest_unfilled(candidates, morale)
        if target is None:
            break
        rate = base_rates[target] + single_bonus
        headroom = MAX_MORALE - morale[target]
        segment = remaining if rate <= 0.0 else min(remaining, headroom / rate)
        _advance_dormitory_occupants(
            occupants,
            base_rates,
            morale,
            trace,
            segment,
            target=target,
            bonus=single_bonus,
        )
        remaining -= segment
        candidates.remove(target)
    if remaining > 1e-9:
        _advance_dormitory_occupants(
            occupants,
            base_rates,
            morale,
            trace,
            remaining,
            target=None,
            bonus=0.0,
        )


def _self_contained_recovery_rates(room: Any) -> dict[str, float]:
    """Operators whose dormitory recovery is self-provided only (``自律``)."""
    rates: dict[str, float] = {}
    for skill in room.operators:
        rate = self_contained_recovery_rate(skill.description)
        if rate is not None:
            rates[skill.operator_name] = rate
    return rates


def _lowest_unfilled(candidates: list[str], morale: dict[str, float]) -> str | None:
    unfilled = [
        name
        for name in candidates
        if morale.get(name, MAX_MORALE) < MAX_MORALE - MORALE_STATE_TOLERANCE
    ]
    if not unfilled:
        return None
    return min(
        unfilled,
        key=lambda name: (morale.get(name, MAX_MORALE), candidates.index(name)),
    )


def _advance_dormitory_occupants(
    occupants: list[str],
    base_rates: dict[str, float],
    morale: dict[str, float],
    trace: _CycleTrace,
    hours: float,
    *,
    target: str | None,
    bonus: float,
) -> None:
    if hours <= 0.0:
        return
    for name in occupants:
        rate = base_rates[name] + (bonus if name == target else 0.0)
        potential = rate * hours
        current = morale.get(name, MAX_MORALE)
        updated = min(MAX_MORALE, current + potential)
        morale[name] = updated
        trace.recovery[name] = trace.recovery.get(name, 0.0) + potential
        trace.applied_recovery[name] = (
            trace.applied_recovery.get(name, 0.0) + (updated - current)
        )


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
    """White text: ``manpowerRecover`` / 100, i.e. 1.6 at level 1 through 2.0 at level 5."""
    return 1.5 + 0.1 * min(5, max(1, int(room_level)))


def dormitory_comfort(room_level: int) -> float:
    """Assumed comfort: players keep dormitories fully decorated."""
    return min(
        DORMITORY_COMFORT_LIMIT,
        DORMITORY_COMFORT_PER_LEVEL * min(5, max(1, int(room_level))),
    )


def dormitory_comfort_recovery_rate(room_level: int) -> float:
    """Green text: ``0.0004`` morale per comfort point (0.4/h at level 1, 2.0/h at level 5)."""
    return COMFORT_RECOVERY_PER_POINT * dormitory_comfort(room_level)


def dormitory_passive_recovery_rate(room_level: int) -> float:
    """White plus green text, before skill bonuses."""
    return dormitory_base_recovery_rate(room_level) + dormitory_comfort_recovery_rate(
        room_level
    )


def dormitory_operator_bonuses(skill: Any) -> tuple[float, float, float | None]:
    """Parse one dormitory skill into ``(room_wide, own, single_target)`` bonuses.

    This is the single regex layer behind :func:`dormitory_recovery_bonuses`; the
    optimizer's dormitory value model parses skill-by-skill through it so that both
    sides read exactly the same numbers.
    """
    description = re.sub(r"<[^>]+>", "", skill.description)
    all_bonus = 0.0
    all_matches = re.findall(
        r"所有干员(?:的)?心情每小时恢复(?:速度)?\+([0-9]+(?:\.[0-9]+)?)",
        description,
    )
    if all_matches:
        all_bonus = max(float(value) for value in all_matches)
    own_bonus = 0.0
    self_matches = re.findall(
        r"自身心情每小时恢复(?:速度)?\+([0-9]+(?:\.[0-9]+)?)",
        description,
    )
    if self_matches:
        own_bonus = max(float(value) for value in self_matches)
    single_bonus: float | None = None
    single_matches = re.findall(
        r"(?:某一名|某个|前一位).{0,12}?每小时恢复(?:速度)?\+([0-9]+(?:\.[0-9]+)?)",
        description,
    )
    if single_matches:
        single_bonus = max(float(value) for value in single_matches)
    return all_bonus, own_bonus, single_bonus


def dormitory_recovery_bonuses(
    room: Any,
) -> tuple[float, dict[str, float], tuple[float, str] | None]:
    """Parse one dormitory room's recovery skills.

    Returns ``(all_bonus, self_bonuses, single)`` where ``single`` is the strongest
    ``除自身以外心情未满的某个干员每小时恢复+N`` effect as ``(bonus, owner_name)``.
    """
    all_bonus = 0.0
    single: tuple[float, str] | None = None
    self_bonuses: dict[str, float] = {}
    for skill in room.operators:
        skill_all, skill_own, skill_single = dormitory_operator_bonuses(skill)
        all_bonus = max(all_bonus, skill_all)
        if skill_own > 0.0:
            self_bonuses[skill.operator_name] = max(
                self_bonuses.get(skill.operator_name, 0.0),
                skill_own,
            )
        if skill_single is not None and (single is None or skill_single > single[0]):
            single = (skill_single, skill.operator_name)
    return all_bonus, self_bonuses, single


def dormitory_single_target(room: Any) -> tuple[float, str] | None:
    """The sticky ``某个干员`` recovery effect of one dormitory, if any."""
    _, _, single = dormitory_recovery_bonuses(room)
    return single


def dormitory_recovery_rates(room: Any, passive: float) -> dict[str, float]:
    """Per-operator dormitory recovery rate: 白字 + 绿字 + 房间/自身技能.

    ``passive`` is :func:`dormitory_passive_recovery_rate` for the room level plus
    the control-centre ``所有宿舍内`` boost. The room-wide bonus is the **maximum**
    over occupants, exactly as in :func:`dormitory_recovery_bonuses`. Self-contained
    nodes (``自律``) ignore the passive and every bonus and only recover their own
    rate. The sticky single-target bonus is *not* included here: it lands on one
    roommate at a time (see :func:`dormitory_single_target`).

    The morale audit and the optimizer's dormitory value model both go through this
    function, so "the beam thinks it is worth it" and "the audit books it" cannot
    drift apart.
    """
    all_bonus, self_bonuses, _ = dormitory_recovery_bonuses(room)
    self_contained = _self_contained_recovery_rates(room)
    rates: dict[str, float] = {}
    for skill in room.operators:
        name = skill.operator_name
        if name in self_contained:
            rates[name] = self_contained[name]
        else:
            rates[name] = passive + all_bonus + self_bonuses.get(name, 0.0)
    return rates


def _dormitory_boost_from_effects(effects: Iterable[MoraleEffects]) -> float:
    """Largest ``所有宿舍内`` recovery bonus among (control-centre) parsed effects."""
    return max((effect.all_dormitories_recovery for effect in effects), default=0.0)


def control_dormitory_boost(shift: ShiftPlan) -> float:
    """Room-wide dormitory recovery the control centre grants (``所有宿舍内``)."""
    control_rooms = [room for room in shift.rooms if room.room_type == "CONTROL"]
    if not control_rooms:
        return 0.0
    room_names = {
        skill.operator_name
        for room in shift.rooms
        for skill in room.operators
    }
    return _dormitory_boost_from_effects(
        parse_morale_effects(skill.description, room_names)
        for room in control_rooms
        for skill in room.operators
    )


def morale_consumption_by_operator(
    layout: Layout,
    shifts: list[ShiftPlan],
) -> dict[str, float]:
    """Per-operator morale consumption over one cycle, booked by the audit itself.

    One deterministic replay (not the fixed-point diagnostic), so the optimizer can
    weigh dormitory beds by how much morale each operator actually spends.
    """
    names = {
        skill.operator_name
        for shift in shifts
        for room in [*shift.rooms, *shift.dormitories]
        for skill in room.operators
    }
    if not names:
        return {}
    trace = _simulate_cycle(layout, shifts, {name: MAX_MORALE for name in names})
    return {name: trace.consumption.get(name, 0.0) for name in names}


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
        "unmodelledMoraleEffects": [],
        "assumptions": [],
    }
