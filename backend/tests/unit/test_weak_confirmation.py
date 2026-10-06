"""
test_weak_confirmation.py — gate-refused detections may RESCUE a delivery, never BECOME one.

The defect these tests exist for
-------------------------------
The single pass produced deliveries that the evidence plainly described but the
confirmation bar refused: the ball was on screen, YOLO boxed it every frame, and
the association gate rejected the boxes, so the event closed with fewer than
MIN_DELIVERY_FRAMES confirmed detections and was thrown away. The fix is a
SECONDARY confirmation path, and a second path is exactly where a system starts
accepting things it should not.

So the risk here is not that the rescue never fires. It is that it fires on the
wrong evidence. Every test below attacks that, by breaking one agreement at a
time and asserting the promotion is refused:

    temporal_continuity    one continuous sighting, not three glimpses
    path_coherence         a physically possible journey, not two balls at once
    pitch_corridor_residency   where a played ball is, not a boot or a hat
    physically_possible_speed   a speed the event's own detections support
    single_object_not_two  one bounded region, not two balls in the same passage
    cricket_context_present     something was happening in the cricket area

The first two are MANDATORY. Without continuity or without a possible path the
remaining four prove nothing, so failing either one is fatal on its own.

WHAT MUST NEVER CHANGE
----------------------
The refused measurements stay EVIDENCE. They are not detections, not trajectory
points, and never Kalman input. Tests below assert that on the promoted event
itself, not merely on the refusal path: a promoted event's `confirmed_frames`
still contains exactly the frames the tracker accepted.

And the standard path is untouched. An event that clears MIN_DELIVERY_FRAMES on
its own is never even offered to the rescue, so its confirmation method, its
confidence and its trajectory are exactly what they were before this feature
existed.
"""

from __future__ import annotations

import unittest
from typing import Optional

from app.core.config import SegmentationConfig
from app.segmentation.candidate_generation import (
    ConfirmationMethod,
    EventSegmenter,
    FrameEvidence,
    WeakSignal,
)

# Production geometry, so the frame-fraction corridor means what it means on a
# real broadcast frame rather than being rescued by a toy resolution.
W, H, FPS = 1280, 720, 25.0
CORRIDOR_X = (0.30 * W, 0.70 * W)   # 384 .. 896
CORRIDOR_Y = (0.20 * H, 0.80 * H)   # 144 .. 576

# A ball walking up the pitch at 10 px/frame, entirely inside the corridor.
BALL_X0, BALL_Y0, BALL_DX, BALL_DY = 600.0, 500.0, 8.0, -6.0


def ball_at(t: int, scale: float = 1.0, origin_x: float = BALL_X0,
            origin_y: float = BALL_Y0) -> tuple[float, float]:
    """Position `t` steps along the ball's path, optionally rescaled in speed."""
    return (origin_x + BALL_DX * scale * t, origin_y + BALL_DY * scale * t)


def in_corridor(pt: tuple[float, float]) -> bool:
    return (CORRIDOR_X[0] <= pt[0] <= CORRIDOR_X[1]
            and CORRIDOR_Y[0] <= pt[1] <= CORRIDOR_Y[1])


class WeakCase(unittest.TestCase):
    """Builds an event out of explicit evidence, so nothing is left to chance."""

    def segmenter(self) -> EventSegmenter:
        return EventSegmenter(
            fps=FPS, width=W, height=H, config=SegmentationConfig()
        )

    def feed_confirmed(self, seg, frame: int, pt, **kw) -> None:
        seg.observe(FrameEvidence(
            frame=frame, confirmed=True, candidate=False, position=pt,
            confidence=0.31,
            in_pitch_corridor=kw.get("in_corridor", in_corridor(pt)),
            motion_energy=kw.get("motion", 6.0),
            pitch_motion=kw.get("pitch_motion", 4.0),
        ))

    def feed_refused(self, seg, frame: int, pt, **kw) -> None:
        seg.observe(FrameEvidence(
            frame=frame, confirmed=False, candidate=True, position=pt,
            confidence=0.28,
            in_pitch_corridor=kw.get("in_corridor", in_corridor(pt)),
            motion_energy=kw.get("motion", 6.0),
            pitch_motion=kw.get("pitch_motion", 4.0),
        ))

    def expire(self, seg, after: int = 73) -> None:
        """Close the open event the way the pass does: by coasting past the ceiling.

        Deliberately NOT `close_open_event`, whose default reason is END_OF_VIDEO
        and which caps confidence at 0.9 and marks the event ambiguous. A coast
        ceiling is the ordinary, corroborated end of a delivery.
        """
        seg.observe(FrameEvidence(frame=after))

    def promote(self, *, confirmed: int = 2, refused: int = 11,
                refused_positions=None, confirmed_positions=None,
                confirmed_scale: float = 1.0, refused_scale: float = 1.0,
                refused_in_corridor=None, confirmed_in_corridor=None,
                motion: float = 6.0, pitch_motion: float = 4.0,
                expire_after: Optional[int] = None):
        """Run one event of `confirmed` accepted detections then N gate refusals."""
        seg = self.segmenter()
        first = 40
        for i in range(confirmed):
            pt = (
                ball_at(i, scale=confirmed_scale)
                if confirmed_positions is None else confirmed_positions[i]
            )
            self.feed_confirmed(
                seg, first + i, pt,
                motion=motion, pitch_motion=pitch_motion,
                **({} if confirmed_in_corridor is None
                   else {"in_corridor": confirmed_in_corridor}),
            )
        base = first + confirmed
        for i in range(refused):
            pt = (
                ball_at(base - first + i, scale=refused_scale)
                if refused_positions is None else refused_positions[i]
            )
            self.feed_refused(
                seg, base + i, pt, motion=motion, pitch_motion=pitch_motion,
                **({} if refused_in_corridor is None
                   else {"in_corridor": refused_in_corridor}),
            )
        self.expire(seg, after=base + refused + 20)
        return seg


# ═══════════════════════════════════════════════════════════════════════════
# 1. The rescue fires, and it is a real delivery
# ═══════════════════════════════════════════════════════════════════════════
class TestCoherentWeakEvidenceIsPromoted(WeakCase):
    def setUp(self):
        self.seg = self.promote()
        self.event = self.seg.events[0]

    def test_two_confirmed_detections_alone_would_be_discarded(self):
        """The premise: below the standard bar, so the rescue is what admits it."""
        self.assertEqual(
            len(self.event.confirmed_frames), 2,
            "this test is meaningless unless the event is under the standard bar",
        )
        self.assertLess(
            len(self.event.confirmed_frames),
            self.seg.resolved.min_event_confirmed_frames,
        )

    def test_it_is_promoted_on_the_coherence_path(self):
        self.assertTrue(
            self.event.weak_confirmation["promoted"],
            self.event.weak_confirmation["reasons"],
        )
        self.assertEqual(
            self.event.confirmation_method, ConfirmationMethod.COHERENT_WEAK
        )
        self.assertTrue(self.event.is_weak_confirmation)

    def test_all_six_signals_agreed_and_the_numbers_are_recorded(self):
        report = self.event.weak_confirmation
        self.assertEqual(report["signals_agreed"], len(WeakSignal.ALL))
        self.assertEqual(report["signals_agreed"], 6)
        for name in WeakSignal.ALL:
            self.assertTrue(
                report["signals"][name]["pass"], f"{name} did not agree"
            )
        # The numbers behind the verdict travel with it, so "the evidence was
        # thin" and "the evidence was examined and did not agree" stay
        # distinguishable to whoever reads this later.
        self.assertIn("contiguous_fraction", report["signals"][WeakSignal.TEMPORAL_CONTINUITY])
        self.assertIn("worst_reach_ratio", report["signals"][WeakSignal.PATH_COHERENCE])
        self.assertIn("ratio", report["signals"][WeakSignal.CORRIDOR_RESIDENCY])
        self.assertIn("speed_ratio", report["signals"][WeakSignal.PHYSICAL_MOVEMENT])
        self.assertIn("spread_px", report["signals"][WeakSignal.RIVAL_ABSENCE])
        self.assertIn("peak_pitch_energy", report["signals"][WeakSignal.PITCH_CONTEXT])

    def test_it_gets_its_own_confidence_not_the_standard_one(self):
        self.assertAlmostEqual(
            self.event.confidence,
            self.seg.resolved.weak_confirmation_confidence, places=6,
        )
        self.assertLess(self.event.confidence, 1.0)

    def test_a_corroborated_ending_is_not_flagged_ambiguous(self):
        self.assertFalse(
            self.event.ambiguous,
            f"ambiguous for reasons nobody asked for: "
            f"{self.event.ambiguous_reasons}",
        )

    def test_the_event_is_reported_as_a_primary_event(self):
        self.assertEqual(
            [e.index for e in self.seg.primary_events()], [self.event.index]
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. THE REFUSED MEASUREMENTS STAY EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════
class TestRefusedMeasurementsRemainEvidenceOnly(WeakCase):
    def setUp(self):
        self.seg = self.promote()
        self.event = self.seg.events[0]

    def test_promotion_does_not_promote_the_measurements_to_detections(self):
        self.assertEqual(self.event.confirmed_frames, [40, 41])
        self.assertEqual(self.event.weak_evidence_frames, list(range(42, 53)))
        self.assertEqual(
            set(self.event.confirmed_frames) & set(self.event.weak_evidence_frames),
            set(),
            "a gate-refused frame leaked into confirmed_frames — the refusal "
            "would no longer be a refusal",
        )

    def test_total_measurements_counts_both_channels_separately(self):
        self.assertEqual(self.event.total_measurement_count, 13)
        self.assertEqual(self.event.weak_evidence_count, 11)
        self.assertEqual(len(self.event.confirmed_frames), 2)

    def test_the_promoted_event_still_bounds_itself_by_what_was_accepted(self):
        """A rescue widens the evidence, never the set of accepted detections."""
        self.assertEqual(self.event.start_frame, 40)
        self.assertEqual(self.event.confirmed_frames, [40, 41])
        # The event may keep running over frames it declined to confirm, but it
        # may not reach past the evidence that was actually collected.
        self.assertLessEqual(self.event.end_frame, 52)
        self.assertGreaterEqual(self.event.end_frame, 41)

    def test_the_refusals_are_auditable_with_their_frames(self):
        # Frame RANGES, not the full list, are what the evidence block persists.
        self.assertEqual(self.event.evidence["weak_evidence_frame_ranges"], [[42, 52]])
        d = self.event.as_dict()
        self.assertEqual(d["weak_evidence_count"], 11)
        self.assertEqual(d["confirmation_method"], "coherent_weak_detection")
        self.assertEqual(d["weak_confirmation"]["promoted"], True)

    def test_the_report_states_the_evidence_only_rule_in_words(self):
        note = self.event.weak_confirmation["evidence_only_note"]
        for word in ("refused", "evidence", "Kalman"):
            self.assertIn(word, note)


# ═══════════════════════════════════════════════════════════════════════════
# 3. The standard path is untouched
# ═══════════════════════════════════════════════════════════════════════════
class TestStandardConfirmationIsUnchanged(WeakCase):
    def test_eight_confirmed_detections_need_no_rescue(self):
        seg = self.segmenter()
        for i in range(8):
            self.feed_confirmed(seg, 40 + i, ball_at(i))
        self.expire(seg, after=100)
        ev = seg.events[0]
        self.assertEqual(ev.confirmation_method, ConfirmationMethod.STANDARD)
        self.assertFalse(ev.is_weak_confirmation)
        # Never even offered to the rescue, so there is nothing to audit and
        # nothing that can be read as "this delivery was a near miss".
        self.assertIsNone(ev.weak_confirmation)
        self.assertEqual(ev.confidence, 1.0)
        self.assertEqual(ev.ambiguous, False)

    def test_the_delivery_floor_itself_is_not_relaxed_by_the_rescue(self):
        seg = self.segmenter()
        for i in range(8):
            self.feed_confirmed(seg, 40 + i, ball_at(i))
        self.expire(seg, after=100)
        self.assertEqual(
            len(seg.events[0].confirmed_frames),
            seg.resolved.min_event_confirmed_frames,
        )
        # The rescue is a SECOND path for thin events, never a lowered bar for
        # the first one: the resolved floor is still the documented 8.
        self.assertEqual(seg.resolved.min_event_confirmed_frames, 8)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Each agreement is load-bearing
# ═══════════════════════════════════════════════════════════════════════════
class TestTemporalContinuityIsMandatory(WeakCase):
    def test_three_glimpses_are_not_one_sighting(self):
        """Two long silences split 11 refusals into islands. Continuity fails."""
        seg = self.segmenter()
        self.feed_confirmed(seg, 40, ball_at(0))
        self.feed_confirmed(seg, 41, ball_at(1))
        for i, f in enumerate((42, 43, 44, 60, 61, 62, 63, 78, 79, 80, 81)):
            self.feed_refused(seg, f, ball_at(2 + i))
        self.expire(seg, after=120)
        ev = seg.events[0]
        report = ev.weak_confirmation
        self.assertFalse(report["signals"][WeakSignal.TEMPORAL_CONTINUITY]["pass"])
        self.assertFalse(report["promoted"])
        self.assertEqual(ev.confirmation_method, ConfirmationMethod.STANDARD)
        self.assertTrue(ev.ambiguous)
        self.assertTrue(
            any(WeakSignal.TEMPORAL_CONTINUITY in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_measured_silence_is_reported_not_just_the_verdict(self):
        seg = self.segmenter()
        self.feed_confirmed(seg, 40, ball_at(0))
        self.feed_confirmed(seg, 41, ball_at(1))
        for i, f in enumerate((42, 43, 44, 60, 61, 62, 63, 78, 79, 80, 81)):
            self.feed_refused(seg, f, ball_at(2 + i))
        self.expire(seg, after=120)
        signal = seg.events[0].weak_confirmation["signals"][WeakSignal.TEMPORAL_CONTINUITY]
        self.assertGreater(signal["silence_fraction"], signal["silence_fraction_max"])
        self.assertLess(
            signal["contiguous_fraction"], signal["contiguous_fraction_min"]
        )
        # 40..44, 60..63, 78..81 — the longest run is five frames, not eleven.
        self.assertEqual(signal["longest_run_frames"], 5)
        self.assertEqual(signal["measurements"], 13)
        self.assertEqual(signal["span_frames"], 42)


class TestPathCoherenceIsMandatory(WeakCase):
    def test_a_teleport_is_refused_even_when_everything_else_agrees(self):
        positions = [ball_at(2 + i) for i in range(11)]
        positions[5] = (1100.0, 160.0)      # a second ball, elsewhere in the frame
        seg = self.promote(refused_positions=positions)
        ev = seg.events[0]
        report = ev.weak_confirmation
        self.assertFalse(report["signals"][WeakSignal.PATH_COHERENCE]["pass"])
        self.assertGreaterEqual(
            report["signals"][WeakSignal.PATH_COHERENCE]["unreachable_pairs"], 1
        )
        self.assertFalse(report["promoted"])
        self.assertTrue(
            any("path_coherence" in r for r in report["reasons"]), report["reasons"]
        )

    def test_path_verdict_is_measured_against_the_event_s_own_speed(self):
        seg = self.promote()
        path = seg.events[0].weak_confirmation["signals"][WeakSignal.PATH_COHERENCE]
        self.assertAlmostEqual(path["reference_speed_px_per_frame"], 10.0, places=1)
        self.assertLess(path["worst_reach_ratio"], 1.0)
        self.assertEqual(path["unreachable_pairs"], 0)


class TestRivalAbsenceIsMeasuredAndCounted(WeakCase):
    """
    TWO OBJECTS IN ONE PASSAGE OF PLAY
    ----------------------------------
    The protection against a second ball is primarily PATH_COHERENCE, which is
    mandatory: two balls in different parts of the frame cannot be joined by any
    physically possible travel, so the pair is refused outright (see
    `TestPathCoherenceIsMandatory`).

    RIVAL_ABSENCE is the second, independent reading of the same evidence — the
    refused positions must fit in ONE bounded region — and per
    `core/constants.py` it is a COUNTED signal, not a mandatory one: two of the
    six are mandatory and four must agree in total. So a lone RIVAL_ABSENCE
    failure does not by itself veto a promotion. That is deliberate: a ball off
    the bat genuinely travels further than 35% of the frame diagonal, and making
    this signal mandatory would throw away real hard-hit deliveries.

    These tests pin both halves of that contract so it cannot drift silently in
    either direction.
    """

    SPREAD_POSITIONS = [ball_at(i, scale=7.5) for i in range(13)]

    def test_a_path_leaving_the_allowed_region_is_measured_and_flagged(self):
        seg = self.promote(refused=13, refused_positions=self.SPREAD_POSITIONS,
                           confirmed_scale=7.5)
        rival = seg.events[0].weak_confirmation["signals"][WeakSignal.RIVAL_ABSENCE]
        self.assertFalse(rival["pass"])
        self.assertGreater(rival["spread_px"], rival["spread_limit_px"])
        self.assertEqual(rival["spread_fraction_of_diagonal"], 0.35)

    def test_a_genuine_two_ball_passage_is_refused_outright(self):
        """Two independent sightings in one passage: the path cannot join them."""
        seg = self.promote(
            refused=12,
            refused_positions=[(420.0 + 8 * i, 480.0) for i in range(6)]
                              + [(1150.0 - 6 * i, 180.0) for i in range(6)],
        )
        report = seg.events[0].weak_confirmation
        self.assertFalse(report["signals"][WeakSignal.PATH_COHERENCE]["pass"])
        self.assertFalse(report["signals"][WeakSignal.RIVAL_ABSENCE]["pass"])
        self.assertFalse(report["promoted"], report["reasons"])
        self.assertEqual(
            seg.events[0].confirmation_method, ConfirmationMethod.STANDARD
        )

    def test_a_lone_counted_disagreement_is_tolerated_by_design(self):
        """The tolerance, pinned. If this ever inverts, the contract changed."""
        seg = self.promote(refused=13, refused_positions=self.SPREAD_POSITIONS,
                           confirmed_scale=7.5)
        report = seg.events[0].weak_confirmation
        failed = [k for k in WeakSignal.ALL if not report["signals"][k]["pass"]]
        self.assertIn(WeakSignal.RIVAL_ABSENCE, failed)
        self.assertGreaterEqual(
            report["signals_agreed"], seg.resolved.weak_min_signal_agreement
        )

    def test_two_disagreements_drop_below_the_floor_and_are_refused(self):
        """RIVAL_ABSENCE failing alongside CORRIDOR_RESIDENCY is enough."""
        off_pitch_spread = [(200.0 + 75.0 * i, 640.0) for i in range(9)]
        seg = self.promote(
            refused=9, refused_positions=off_pitch_spread,
            refused_in_corridor=False,
            confirmed_positions=[(300.0 + 8 * i, 600.0) for i in range(2)],
            confirmed_in_corridor=False,
        )
        report = seg.events[0].weak_confirmation
        failed = {k for k in WeakSignal.ALL if not report["signals"][k]["pass"]}
        self.assertIn(WeakSignal.RIVAL_ABSENCE, failed)
        self.assertIn(WeakSignal.CORRIDOR_RESIDENCY, failed)
        self.assertLess(
            report["signals_agreed"],
            seg.resolved.weak_min_signal_agreement,
        )
        self.assertFalse(report["promoted"], report["reasons"])


class TestCorridorResidencyRefusesOffPitchEvidence(WeakCase):
    def test_refusals_off_the_pitch_are_not_a_played_ball(self):
        off_pitch = [(1150.0, 40.0 + 4 * i) for i in range(11)]
        seg = self.promote(refused_positions=off_pitch, refused_in_corridor=False)
        report = seg.events[0].weak_confirmation
        self.assertFalse(
            report["signals"][WeakSignal.CORRIDOR_RESIDENCY]["pass"]
        )
        self.assertEqual(report["signals"][WeakSignal.CORRIDOR_RESIDENCY]["ratio"], 0.0)
        self.assertFalse(report["promoted"])


class TestPhysicalMovementIsMeasured(WeakCase):
    """
    A ball off the bat accelerates; it does not multiply its speed twentyfold
    between two adjacent frames. Measured as the refused run's implied speed over
    its own span, against the event's own measured speed.

    Constructed so PATH_COHERENCE still passes — otherwise this would only be
    re-testing the mandatory signal and would prove nothing about this one.
    """

    # Anchors creeping at 12.5 px/frame; refusals striding at 85 px/frame. The
    # reach radius at gap 1 is 1.5*12.5 + 73.4 + 5.9 = 98 px, so every stride is
    # individually survivable while the run as a whole implies ~6.8x the anchors.
    ANCHORS = [(400.0 + 10.0 * i, 500.0 - 7.5 * i) for i in range(2)]
    STRIDERS = [(400.0 + 68.0 * i, 500.0 - 51.0 * i) for i in range(6)]

    def setUp(self):
        self.seg = self.promote(
            refused=len(self.STRIDERS), refused_positions=self.STRIDERS,
            confirmed_positions=self.ANCHORS,
        )
        self.report = self.seg.events[0].weak_confirmation

    def test_the_path_stays_physically_possible_so_only_speed_is_at_fault(self):
        self.assertTrue(
            self.report["signals"][WeakSignal.PATH_COHERENCE]["pass"],
            "this test only isolates physical movement if the path is possible",
        )

    def test_the_impossible_speed_ratio_is_measured_and_named(self):
        movement = self.report["signals"][WeakSignal.PHYSICAL_MOVEMENT]
        self.assertFalse(movement["pass"])
        self.assertGreater(movement["speed_ratio"], movement["speed_ratio_band"][1])
        self.assertAlmostEqual(
            movement["reference_speed_px_per_frame"], 12.5, places=1
        )
        self.assertGreater(movement["implied_speed_px_per_frame"], 80.0)

    def test_the_event_still_carries_the_measurement_it_failed_on(self):
        """A failed signal is reported, not discarded: thin and disagreeing are
        different diagnoses and a diagnostic has to be able to tell them apart."""
        self.assertIsNotNone(self.seg.events[0].weak_confirmation)
        self.assertIn(WeakSignal.PHYSICAL_MOVEMENT, self.report["signals"])
        self.assertEqual(self.report["signals_agreed"], 5)


class TestPitchContextIsMeasured(WeakCase):
    """A static artefact cannot satisfy PITCH_CONTEXT, and the numbers say which
    half of the test failed."""

    def test_a_dead_quiet_frame_with_no_corridor_context_fails_the_signal(self):
        seg = self.promote(
            motion=0.0, pitch_motion=0.0,
            confirmed_positions=[(120.0 + 8 * i, 60.0) for i in range(2)],
            confirmed_in_corridor=False,
        )
        ctx = seg.events[0].weak_confirmation["signals"][WeakSignal.PITCH_CONTEXT]
        self.assertFalse(ctx["pass"])
        self.assertFalse(ctx["event_confirmed_in_corridor"])
        self.assertLess(ctx["peak_pitch_energy"], ctx["quiet_threshold"])
        self.assertEqual(ctx["quiet_threshold"], 1.5)

    def test_quiet_frames_still_pass_when_the_event_was_already_on_the_pitch(self):
        """The two halves of the test are independent, and either one suffices."""
        seg = self.promote(motion=0.0, pitch_motion=0.0)
        ctx = seg.events[0].weak_confirmation["signals"][WeakSignal.PITCH_CONTEXT]
        self.assertTrue(ctx["event_confirmed_in_corridor"])
        self.assertTrue(ctx["pass"])


# ═══════════════════════════════════════════════════════════════════════════
# 5. The floors are floors
# ═══════════════════════════════════════════════════════════════════════════
class TestFloorsAreNeverLowered(WeakCase):
    def test_too_few_refusals_is_refused_even_when_coherent(self):
        seg = self.promote(refused=3)
        report = seg.events[0].weak_confirmation
        self.assertFalse(report["promoted"])
        self.assertGreaterEqual(report["weak_evidence_count"], 3)
        self.assertTrue(
            any("gate-refused" in r for r in report["reasons"]), report["reasons"]
        )

    def test_too_few_anchor_detections_is_refused_even_when_coherent(self):
        seg = self.promote(confirmed=1, refused=20)
        report = seg.events[0].weak_confirmation
        self.assertFalse(report["promoted"])
        self.assertTrue(
            any("confirmed detections" in r for r in report["reasons"]),
            report["reasons"],
        )

    def test_total_measurement_floor_counts_both_channels_and_is_not_lowered(self):
        r = self.segmenter().resolved
        self.assertEqual(r.weak_min_total_measurements, 8)
        self.assertGreaterEqual(
            r.weak_min_total_measurements, r.min_event_confirmed_frames
        )
        self.assertEqual(r.weak_min_candidate_frames, 4)
        self.assertEqual(r.weak_min_confirmed_frames, 2)
        self.assertEqual(r.weak_min_signal_agreement, 4)

    def test_the_signal_agreement_floor_actually_bites(self):
        """Fewer than four agreeing signals is refused, however clean each is."""
        seg = self.promote()
        r = seg.resolved
        raised = SegmentationConfig()
        raised.weak_min_signal_agreement = 7      # more than there are signals
        strict = EventSegmenter(
            fps=FPS, width=W, height=H, config=raised
        )
        for i in range(2):
            self.feed_confirmed(strict, 40 + i, ball_at(i))
        for i in range(11):
            self.feed_refused(strict, 42 + i, ball_at(2 + i))
        self.expire(strict, after=80)
        report = strict.events[0].weak_confirmation
        self.assertFalse(report["promoted"])
        self.assertTrue(
            any("coherence signals agreed" in r for r in report["reasons"]),
            report["reasons"],
        )
        self.assertEqual(report["signals_agreed"], 6)


# ═══════════════════════════════════════════════════════════════════════════
# 6. A refused rescue is recorded as examined, not as absent
# ═══════════════════════════════════════════════════════════════════════════
class TestRefusedRescueIsAuditable(WeakCase):
    def test_an_ambiguous_event_says_the_rescue_was_attempted(self):
        seg = self.segmenter()
        self.feed_confirmed(seg, 40, ball_at(0))
        for i in range(11):
            self.feed_refused(seg, 41 + i, (1150.0, 40.0))
        self.expire(seg, after=90)
        ev = seg.events[0]
        self.assertTrue(ev.ambiguous)
        self.assertEqual(ev.confidence, 0.0)
        self.assertIsNotNone(ev.weak_confirmation)
        self.assertFalse(ev.weak_confirmation["promoted"])
        self.assertTrue(
            any("coherent weak confirmation was attempted" in r
                for r in ev.ambiguous_reasons),
            ev.ambiguous_reasons,
        )

    def test_an_event_with_no_refusals_says_there_was_nothing_to_rescue_with(self):
        seg = self.segmenter()
        for i in range(3):
            self.feed_confirmed(seg, 40 + i, ball_at(i))
        self.expire(seg, after=90)
        ev = seg.events[0]
        self.assertTrue(ev.ambiguous)
        self.assertIsNotNone(ev.weak_confirmation)
        self.assertFalse(ev.weak_confirmation["promoted"])
        self.assertEqual(ev.weak_confirmation["weak_evidence_count"], 0)
        self.assertTrue(
            any("no gate-refused evidence" in r for r in ev.ambiguous_reasons),
            ev.ambiguous_reasons,
        )

    def test_the_summary_separates_rescued_from_ordinary_events(self):
        ordinary = self.segmenter()
        for i in range(8):
            self.feed_confirmed(ordinary, 40 + i, ball_at(i))
        self.expire(ordinary, after=100)
        rescued = self.promote()
        combined = EventSegmenter(fps=FPS, width=W, height=H,
                                  config=SegmentationConfig())
        combined.events = list(ordinary.events) + list(rescued.events)
        summary = combined.summary()
        self.assertEqual(summary["confirmation_methods"], {
            "standard": 1, "coherent_weak_detection": 1
        })
        self.assertEqual(summary["weak_evidence_total"], 11)
        self.assertEqual(summary["events_primary"], 2)
        self.assertEqual(summary["events_ambiguous"], 0)


if __name__ == "__main__":
    unittest.main()