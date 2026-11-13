"""Contract tests for the morale-cycle audit (the hard gate on sustainable rotation).

The audit is the only thing standing between a plan and ``optimize()`` raising
``ValueError``, so its verdict has to be stable, physically meaningful and free of
the clamping artefacts of an iterated replay.  These tests pin down:

* the per-operator morale balance sheet (consumption vs. recovery) as the
  sustainability criterion, so the ``MAX_MORALE`` cap cannot produce a false
  "the cycle does not restore the initial morale" verdict;
* the structural checks (every outgoing operator rests, nobody works while resting);
* bounded, physically meaningful reported numbers (a cycle may not report
  ``minimumMorale = -180`` just because a replay was iterated 16 times);
* the diagnostic steady-state replay, which must never gate the verdict;
* the recovery/consumption model itself: dormitory comfort (green text),
  consumption-reduction skills, room-wide relief and the control-centre
  ``-0.05/h per stationed operator`` rule, plus explicit reporting of the
  morale clauses the model cannot attribute.
"""

from __future__ import annotations

import unittest

from arknights_schedule_generator.models import (
    BaseSkill,
    Layout,
    RoomAssignment,
    ShiftPlan,
)
from arknights_schedule_generator.morale import MAX_MORALE, audit_morale_cycle


class MoraleAuditTest(unittest.TestCase):
    def test_balanced_two_shift_cycle_passes_hard_gate(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                shift("A", "08:00", 12.0, working=["值班甲"], resting=["值班乙"]),
                shift("B", "20:00", 12.0, working=["值班乙"], resting=["值班甲"]),
            ],
        )

        self.assertTrue(audit["hardGatePassed"], audit["failureReasons"])
        self.assertEqual(audit["failureReasons"], [])
        self.assertEqual(audit["cycleHours"], 24.0)
        self.assertEqual(audit["unrestedOperators"], [])
        # 12h of work costs 12; 12h in a level-5 dorm recovers 12 * 4.0 = 48
        # (1.5 + 0.1 * 5 white text = 2.0/h, plus 0.0004 * 5000 comfort = 2.0/h).
        self.assertAlmostEqual(audit["moraleLedgerByOperator"]["值班甲"]["consumption"], 12.0)
        self.assertAlmostEqual(audit["moraleLedgerByOperator"]["值班甲"]["recovery"], 48.0)
        self.assertAlmostEqual(audit["moraleLedgerByOperator"]["值班甲"]["balance"], 36.0)

    def test_cycle_ending_below_full_morale_still_passes(self) -> None:
        """Recovery that lands while an operator is already at 24 morale is wasted.

        Resting during the first shift of the cycle means the 24 morale cap throws
        recovery away, so the cycle ends *below* the 24 it started from even though
        it is perfectly sustainable.  The old audit compared end vs. start morale and
        could reject exactly this shape; the balance sheet must accept it.
        """
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                # 值班甲 rests first, then works: the first dorm slot is wasted at the cap.
                shift("A", "08:00", 12.0, working=["值班乙"], resting=["值班甲"]),
                shift("B", "20:00", 12.0, working=["值班甲"], resting=["值班乙"]),
            ],
        )

        ledger = audit["moraleLedgerByOperator"]["值班甲"]
        self.assertTrue(audit["hardGatePassed"], audit["failureReasons"])
        self.assertGreater(ledger["balance"], 0.0)
        self.assertLess(ledger["appliedRecovery"], ledger["recovery"])
        self.assertLess(audit["endingMoraleByOperator"]["值班甲"], MAX_MORALE)

    def test_operator_without_rest_fails_with_structural_reason(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                shift("A", "08:00", 12.0, working=["值班甲"], resting=["值班乙"]),
                # 值班甲 leaves the production room but is not sent to a dormitory.
                shift("B", "20:00", 12.0, working=["值班乙"], resting=[]),
            ],
        )

        self.assertFalse(audit["hardGatePassed"])
        self.assertIn("required_rest_exceeds_dormitory_assignment", audit["failureReasons"])
        self.assertIn("值班甲", audit["unrestedOperators"])
        self.assertLess(audit["moraleLedgerByOperator"]["值班甲"]["balance"], 0.0)

    def test_active_and_resting_in_same_shift_fails(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                shift("A", "08:00", 12.0, working=["值班甲"], resting=["值班甲"]),
                shift("B", "20:00", 12.0, working=["值班乙"], resting=["值班乙"]),
            ],
        )

        self.assertFalse(audit["hardGatePassed"])
        self.assertIn("operator_active_and_resting_in_same_shift", audit["failureReasons"])

    def test_morale_balance_deficit_fails_with_bounded_numbers(self) -> None:
        """A cycle that recovers less than it spends must fail on the balance sheet.

        The numbers in the report have to stay inside one cycle's worth of morale:
        replaying the drain until an iteration limit turns 12h of work into
        ``minimumMorale = -102`` and hides which operator is short.
        """
        audit = audit_morale_cycle(
            entry_level_dormitory_layout(),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    working=["高耗干员"],
                    resting=["值班乙"],
                    descriptions={"高耗干员": "心情每小时消耗+1.5"},
                ),
                shift("B", "20:00", 12.0, working=["值班乙"], resting=["高耗干员"]),
            ],
        )

        ledger = audit["moraleLedgerByOperator"]["高耗干员"]
        self.assertFalse(audit["hardGatePassed"])
        self.assertIn("morale_balance_deficit", audit["failureReasons"])
        self.assertIn("高耗干员", audit["moraleBalanceDeficitOperators"])
        # 2.5/h for 12h = 30 consumed vs. 12h * 2.0 = 24 recovered (level-1 dorm).
        self.assertAlmostEqual(ledger["consumption"], 30.0)
        self.assertAlmostEqual(ledger["recovery"], 24.0)
        self.assertAlmostEqual(ledger["balance"], -6.0)
        # Bounded by a single cycle instead of 16 replays: 24 - 30 = -6.
        self.assertAlmostEqual(audit["minimumMorale"], -6.0)
        self.assertAlmostEqual(audit["endingMoraleByOperator"]["高耗干员"], 18.0)

    def test_exact_break_even_cycle_passes(self) -> None:
        audit = audit_morale_cycle(
            entry_level_dormitory_layout(),
            [
                shift("A", "08:00", 8.0, working=["值班甲"], resting=["值班乙"]),
                shift("B", "16:00", 8.0, working=["值班甲"], resting=["值班乙"]),
                shift("C", "00:00", 8.0, working=["值班乙"], resting=["值班甲"]),
            ],
        )

        ledger = audit["moraleLedgerByOperator"]["值班甲"]
        # 16h of work = 16 consumed, one 8h level-1 dormitory shift = 16 recovered.
        self.assertAlmostEqual(ledger["balance"], 0.0, places=6)
        self.assertTrue(audit["hardGatePassed"], audit["failureReasons"])

    def test_steady_state_replay_converges_for_feasible_cycle(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                shift("A", "08:00", 12.0, working=["值班甲"], resting=["值班乙"]),
                shift("B", "20:00", 12.0, working=["值班乙"], resting=["值班甲"]),
            ],
        )

        steady = audit["steadyState"]
        self.assertTrue(steady["converged"])
        self.assertGreaterEqual(steady["iterations"], 1)
        self.assertLessEqual(steady["residual"], 0.0001)

    def test_steady_state_replay_never_gates_the_verdict(self) -> None:
        """The replay is diagnostics only; the balance sheet decides."""
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                shift("A", "08:00", 12.0, working=["值班甲"], resting=["值班乙"]),
                shift("B", "20:00", 12.0, working=["值班甲"], resting=["值班乙"]),
            ],
        )

        self.assertFalse(audit["steadyState"]["converged"])
        self.assertNotIn("morale_cycle_did_not_converge", audit["failureReasons"])
        self.assertFalse(audit["hardGatePassed"])
        # 值班甲 works both shifts and never recovers: the balance sheet says why.
        self.assertIn("morale_balance_deficit", audit["failureReasons"])
        self.assertIn("值班甲", audit["moraleBalanceDeficitOperators"])

    def test_cycle_over_seven_days_fails(self) -> None:
        shifts = [
            shift(f"P{index}", "08:00", 24.0, working=["值班甲"], resting=["值班乙"])
            for index in range(8)
        ]

        audit = audit_morale_cycle(dormitory_layout(), shifts)

        self.assertFalse(audit["hardGatePassed"])
        self.assertIn("cycle_exceeds_seven_days", audit["failureReasons"])

    def test_empty_schedule_fails_with_full_schema(self) -> None:
        audit = audit_morale_cycle(dormitory_layout(), [])

        self.assertFalse(audit["hardGatePassed"])
        self.assertEqual(audit["failureReasons"], ["empty_schedule"])
        self.assertEqual(audit["moraleLedgerByOperator"], {})
        self.assertEqual(audit["moraleBalanceDeficitOperators"], [])
        self.assertEqual(audit["unmodelledMoraleEffects"], [])
        self.assertFalse(audit["steadyState"]["converged"])

    def test_audit_is_deterministic_and_does_not_mutate_shifts(self) -> None:
        shifts = [
            shift("A", "08:00", 12.0, working=["值班甲"], resting=["值班乙"]),
            shift("B", "20:00", 12.0, working=["值班乙"], resting=["值班甲"]),
        ]
        before = [(item.name, item.start, item.duration_hours) for item in shifts]

        first = audit_morale_cycle(dormitory_layout(), shifts)
        second = audit_morale_cycle(dormitory_layout(), shifts)

        self.assertEqual(first, second)
        self.assertEqual(
            before,
            [(item.name, item.start, item.duration_hours) for item in shifts],
        )


class MoraleModelEffectsTest(unittest.TestCase):
    """The recovery/consumption model: comfort, consumption reduction, control centre."""

    def test_dormitory_comfort_scales_recovery_with_room_level(self) -> None:
        audits = {
            level: audit_morale_cycle(
                dormitory_layout((level, level, level, level)),
                [
                    shift("A", "08:00", 12.0, working=["值班甲"], resting=["值班乙"]),
                    shift("B", "20:00", 12.0, working=["值班乙"], resting=["值班甲"]),
                ],
            )
            for level in (1, 5)
        }

        # Level 1: 1.5 + 0.1 * 1 = 1.6 white text, 0.0004 * 1000 = 0.4 green text.
        self.assertAlmostEqual(
            audits[1]["moraleLedgerByOperator"]["值班甲"]["recovery"],
            12.0 * 2.0,
        )
        # Level 5: 2.0 white text, 0.0004 * 5000 = 2.0 green text.
        self.assertAlmostEqual(
            audits[5]["moraleLedgerByOperator"]["值班甲"]["recovery"],
            12.0 * 4.0,
        )
        self.assertIn(
            "Dormitory recovery adds 0.0004 morale per comfort point.",
            audits[5]["assumptions"],
        )

    def test_consumption_reduction_skill_lowers_own_consumption(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    working=["节能干员"],
                    resting=["值班乙"],
                    descriptions={
                        "节能干员": "进驻办公室时，人脉资源的联络速度+35%，心情每小时消耗-0.25",
                    },
                ),
                shift("B", "20:00", 12.0, working=["值班乙"], resting=["节能干员"]),
            ],
        )

        # 1.0 - 0.25 = 0.75 morale per hour instead of 1.0.
        self.assertAlmostEqual(
            audit["moraleLedgerByOperator"]["节能干员"]["consumption"],
            9.0,
        )

    def test_room_wide_consumption_reduction_applies_to_workmates(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(levels=(1, 1, 1, 1)),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    working=["工头", "同事甲", "同事乙"],
                    resting=["值班丁"],
                    descriptions={"工头": "进驻制造站时，当前制造站内所有干员心情每小时消耗-0.1"},
                ),
                shift(
                    "B",
                    "20:00",
                    12.0,
                    working=["值班丁"],
                    resting=["工头", "同事甲", "同事乙"],
                ),
            ],
        )

        for name in ("工头", "同事甲", "同事乙"):
            self.assertAlmostEqual(
                audit["moraleLedgerByOperator"][name]["consumption"],
                12.0 * 0.9,
                msg=name,
            )

    def test_control_centre_relief_applies_to_every_working_operator(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(levels=(1, 1, 1, 1)),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    control=["中枢甲", "中枢乙", "中枢丙"],
                    working=["产线甲", "产线乙"],
                    resting=["值班丁"],
                ),
                shift(
                    "B",
                    "20:00",
                    12.0,
                    control=["中枢甲", "中枢乙", "中枢丙"],
                    working=["值班丁"],
                    resting=["产线甲", "产线乙"],
                ),
            ],
        )

        # Three operators in the control centre: -0.05 * 3 = -0.15 morale per hour
        # for everybody who is on duty, control-centre staff included. The control
        # staff works both shifts, so their cycle consumption is 24h * 0.85.
        for name in ("中枢甲", "中枢丙"):
            self.assertAlmostEqual(
                audit["moraleLedgerByOperator"][name]["consumption"],
                24.0 * 0.85,
                msg=name,
            )
        self.assertAlmostEqual(
            audit["moraleLedgerByOperator"]["产线甲"]["consumption"],
            12.0 * 0.85,
        )
        # Resting operators keep the full dormitory recovery: relief is for people on duty.
        self.assertAlmostEqual(
            audit["moraleLedgerByOperator"]["产线甲"]["recovery"],
            12.0 * 2.0,
        )

    def test_control_centre_skill_recovery_stacks_on_the_relief(self) -> None:
        description = "进驻控制中枢时，控制中枢内所有干员的心情每小时恢复+0.05"
        audit = audit_morale_cycle(
            dormitory_layout(levels=(1, 1, 1, 1)),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    control=["中枢甲", "中枢乙", "中枢丙"],
                    working=["产线甲"],
                    resting=["值班丁"],
                    descriptions={"中枢甲": description},
                ),
                shift(
                    "B",
                    "20:00",
                    12.0,
                    control=["中枢甲", "中枢乙", "中枢丙"],
                    working=["值班丁"],
                    resting=["产线甲"],
                    descriptions={"中枢甲": description},
                ),
            ],
        )

        # 1.0 - 0.05 * 3 relief - 0.05 parsed recovery = 0.80 morale per hour, 24h.
        for name in ("中枢甲", "中枢乙"):
            self.assertAlmostEqual(
                audit["moraleLedgerByOperator"][name]["consumption"],
                24.0 * 0.80,
                msg=name,
            )

    def test_control_centre_consumption_penalty_hits_control_operators(self) -> None:
        description = (
            "进驻控制中枢时，提升会客室内干员所属派系的线索倾向效果，"
            "但控制中枢内所有干员的心情每小时消耗+1.5"
        )
        audit = audit_morale_cycle(
            dormitory_layout(levels=(1, 1, 1, 1)),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    control=["神经质", "中枢乙"],
                    working=["产线甲"],
                    resting=["值班丁"],
                    descriptions={"神经质": description},
                ),
                shift(
                    "B",
                    "20:00",
                    12.0,
                    control=["神经质", "中枢乙"],
                    working=["值班丁"],
                    resting=["产线甲"],
                    descriptions={"神经质": description},
                ),
            ],
        )

        # 1.0 + 1.5 parsed penalty - 0.05 * 2 relief = 2.40 morale per hour, 24h.
        self.assertAlmostEqual(
            audit["moraleLedgerByOperator"]["中枢乙"]["consumption"],
            24.0 * 2.4,
        )
        # A production worker is not in the control centre: relief only.
        self.assertAlmostEqual(
            audit["moraleLedgerByOperator"]["产线甲"]["consumption"],
            12.0 * 0.90,
        )

    def test_control_centre_dormitory_recovery_bonus_reaches_dormitories(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(levels=(1, 1, 1, 1)),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    control=["宿管"],
                    working=["值班甲"],
                    resting=["值班乙"],
                    descriptions={
                        "宿管": "进驻控制中枢时，所有宿舍内所有干员的心情每小时恢复+0.05",
                    },
                ),
                shift(
                    "B",
                    "20:00",
                    12.0,
                    control=["宿管"],
                    working=["值班乙"],
                    resting=["值班甲"],
                ),
            ],
        )

        # Level-1 dorm: 2.0/h base + comfort plus 0.05/h from the control centre.
        self.assertAlmostEqual(
            audit["moraleLedgerByOperator"]["值班乙"]["recovery"],
            12.0 * 2.05,
        )

    def test_unattributable_morale_clauses_are_reported_not_dropped(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(levels=(1, 1, 1, 1)),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    control=["阵营干员"],
                    working=["值班甲"],
                    resting=["值班乙"],
                    descriptions={
                        "阵营干员": "进驻控制中枢时，控制中枢内每个龙门近卫局干员"
                        "可使控制中枢内所有干员的心情每小时恢复+0.05",
                    },
                ),
                shift("B", "20:00", 12.0, working=["值班乙"], resting=["值班甲"]),
            ],
        )

        unmodelled = audit["unmodelledMoraleEffects"]
        self.assertTrue(
            any("龙门近卫局" in item for item in unmodelled),
            unmodelled,
        )
        # Unattributable clauses are not silently credited: only the relief applies.
        self.assertAlmostEqual(
            audit["moraleLedgerByOperator"]["阵营干员"]["consumption"],
            12.0 * 0.95,
        )


def dormitory_layout(levels: tuple[int, ...] = (5, 5, 5, 5)) -> Layout:
    return Layout(
        "243",
        2,
        4,
        3,
        dormitory=len(levels),
        dormitory_levels=levels,
    )


def entry_level_dormitory_layout() -> Layout:
    """Level-1 dormitories: 1.6 white text + 0.4 comfort = 2.0 morale per hour."""
    return dormitory_layout((1, 1, 1, 1))


def morale_skill(name: str, room_type: str, description: str = "") -> BaseSkill:
    return BaseSkill(
        char_id=f"char_{name}",
        operator_name=name,
        room_type=room_type,
        buff_id="fixture",
        buff_name="fixture",
        description=description,
        efficiency=10.0,
        targets=(),
        cond_elite=0,
        cond_level=1,
        unlocked=True,
        complex_condition=False,
        parsed_score=10.0,
    )


def shift(
    name: str,
    start: str,
    duration_hours: float,
    *,
    working: list[str] | None = None,
    resting: list[str] | None = None,
    control: list[str] | None = None,
    descriptions: dict[str, str] | None = None,
) -> ShiftPlan:
    working = working or []
    resting = resting or []
    control = control or []
    descriptions = descriptions or {}
    rooms = []
    if working:
        rooms.append(
            RoomAssignment(
                "manufacture_1",
                "MANUFACTURE",
                "制造站",
                "F_EXP",
                [
                    morale_skill(item, "MANUFACTURE", descriptions.get(item, ""))
                    for item in working
                ],
                0.0,
                room_level=3,
                slots=3,
            )
        )
    if control:
        rooms.append(
            RoomAssignment(
                "control_1",
                "CONTROL",
                "控制中枢",
                None,
                [
                    morale_skill(item, "CONTROL", descriptions.get(item, ""))
                    for item in control
                ],
                0.0,
                room_level=5,
                slots=5,
            )
        )
    dormitories = []
    if resting:
        dormitories.append(
            RoomAssignment(
                "dormitory_1",
                "DORMITORY",
                "宿舍",
                None,
                [morale_skill(item, "DORMITORY", descriptions.get(item, "")) for item in resting],
                0.0,
                room_level=5,
                slots=5,
            )
        )
    return ShiftPlan(name, start, duration_hours, rooms, dormitories)


if __name__ == "__main__":
    unittest.main()
