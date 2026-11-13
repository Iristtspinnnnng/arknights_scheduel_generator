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
* the diagnostic steady-state replay, which must never gate the verdict.
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
        # 12h of work costs 12; 12h in a level-5 dorm recovers 12 * 2.0 = 24.
        self.assertAlmostEqual(audit["moraleLedgerByOperator"]["值班甲"]["consumption"], 12.0)
        self.assertAlmostEqual(audit["moraleLedgerByOperator"]["值班甲"]["recovery"], 24.0)
        self.assertAlmostEqual(audit["moraleLedgerByOperator"]["值班甲"]["balance"], 12.0)

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
            dormitory_layout(),
            [
                shift(
                    "A",
                    "08:00",
                    12.0,
                    working=["高耗干员"],
                    resting=["值班乙"],
                    work_description="心情每小时消耗+1.5",
                ),
                shift("B", "20:00", 12.0, working=["值班乙"], resting=["高耗干员"]),
            ],
        )

        ledger = audit["moraleLedgerByOperator"]["高耗干员"]
        self.assertFalse(audit["hardGatePassed"])
        self.assertIn("morale_balance_deficit", audit["failureReasons"])
        self.assertIn("高耗干员", audit["moraleBalanceDeficitOperators"])
        # 2.5/h for 12h = 30 consumed vs. 12h * 2.0 = 24 recovered.
        self.assertAlmostEqual(ledger["consumption"], 30.0)
        self.assertAlmostEqual(ledger["recovery"], 24.0)
        self.assertAlmostEqual(ledger["balance"], -6.0)
        # Bounded by a single cycle instead of 16 replays: 24 - 30 = -6.
        self.assertAlmostEqual(audit["minimumMorale"], -6.0)
        self.assertAlmostEqual(audit["endingMoraleByOperator"]["高耗干员"], 18.0)

    def test_exact_break_even_cycle_passes(self) -> None:
        audit = audit_morale_cycle(
            dormitory_layout(),
            [
                shift("A", "08:00", 8.0, working=["值班甲"], resting=["值班乙"]),
                shift("B", "16:00", 8.0, working=["值班甲"], resting=["值班乙"]),
                shift("C", "00:00", 8.0, working=["值班乙"], resting=["值班甲"]),
            ],
        )

        ledger = audit["moraleLedgerByOperator"]["值班甲"]
        # 16h of work = 16 consumed, one 8h dormitory shift = 16 recovered.
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


def dormitory_layout(levels: tuple[int, ...] = (5, 5, 5, 5)) -> Layout:
    return Layout(
        "243",
        2,
        4,
        3,
        dormitory=len(levels),
        dormitory_levels=levels,
    )


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
    working: list[str],
    resting: list[str],
    work_description: str = "",
) -> ShiftPlan:
    rooms = []
    if working:
        rooms.append(
            RoomAssignment(
                "manufacture_1",
                "MANUFACTURE",
                "制造站",
                "F_EXP",
                [morale_skill(item, "MANUFACTURE", work_description) for item in working],
                0.0,
                room_level=3,
                slots=3,
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
                [morale_skill(item, "DORMITORY") for item in resting],
                0.0,
                room_level=5,
                slots=5,
            )
        )
    return ShiftPlan(name, start, duration_hours, rooms, dormitories)


if __name__ == "__main__":
    unittest.main()
