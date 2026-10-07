"""
test_pitch_detector.py - the optional Roboflow pitch response, parsed.

These are pure unit tests over `app.pitch.detector`: no network, no pipeline,
fixtures only. What they pin down:

  * every documented response shape is accepted (a bare list, a wrapper, a
    double wrapper, `points`/`keypoints` aliases, keypoints as dicts, arrays
    or `{name: [x, y]}` mappings);
  * names and confidences are read however the model spells them, and fall
    back to `point_N` rather than to an invented name;
  * a whole-set normalised response becomes pixel coordinates - and a mixed
    one never is scaled point by point;
  * anything unreadable is DROPPED, never repaired: a keypoint without a
    coordinate must not reach the calibration;
  * confidence is the mean over the keypoints that carried one, and is None
    (not 0.0) when the response carried none;
  * the API key never survives into an exception message.

The client tests inject a fake transport, so the thread/timeout policy (three
consecutive timeouts disable the detector) is asserted without a live endpoint.
"""

from __future__ import annotations

import dataclasses
import time
import unittest

from app.core.config import PitchConfig
from app.pitch.detector import (
    Keypoint,
    PitchDetectError,
    PitchDetectorUnavailable,
    PitchTimeout,
    RoboflowPitchDetector,
    parse_pitch_response,
    scrub,
)

SIZE = (640, 360)  # (width, height)
KEY = "SECRET_ROBOFLOW_KEY_123"


def parse(payload, image_size=SIZE, frame_number=7):
    return parse_pitch_response(
        payload, frame_number=frame_number, image_size=image_size
    )


def pairs(detection) -> dict[str, tuple[float, float]]:
    return detection.named


# ── Response shapes ───────────────────────────────────────────────────────────
class TestResponseShapes(unittest.TestCase):
    """One keypoint set, written the way each documented response writes it."""

    def test_bare_list_of_predictions(self):
        payload = [
            {
                "keypoints": [
                    {"name": "top_left", "x": 10.0, "y": 20.0},
                    {"name": "top_right", "x": 100.0, "y": 20.0},
                    {"name": "bottom_left", "x": 10.0, "y": 90.0},
                ]
            }
        ]
        detection = parse(payload)
        self.assertEqual(detection.frame_number, 7)
        self.assertEqual(detection.image_size, SIZE)
        self.assertEqual(len(detection.keypoints), 3)
        self.assertEqual(detection.keypoints[0].name, "top_left")
        self.assertEqual(pairs(detection)["top_right"], (100.0, 20.0))

    def test_predictions_wrapper(self):
        detection = parse(
            {
                "predictions": [
                    {"keypoints": [{"name": "a", "x": 5.0, "y": 6.0},
                                   {"name": "b", "x": 7.0, "y": 8.0}]}
                ]
            }
        )
        self.assertEqual(list(pairs(detection)), ["a", "b"])

    def test_double_nested_predictions(self):
        """`{"predictions": {"predictions": [...]}}` - seen in wrapped responses."""
        detection = parse(
            {
                "predictions": {
                    "predictions": [
                        {"keypoints": [{"name": "a", "x": 5.0, "y": 6.0}]}
                    ]
                }
            }
        )
        self.assertEqual(pairs(detection)["a"], (5.0, 6.0))

    def test_points_and_keypoints_aliases(self):
        for alias in ("points", "keypoints"):
            with self.subTest(alias=alias):
                detection = parse(
                    {alias: [{"name": "only", "x": 3.0, "y": 4.0}]}
                )
                self.assertEqual(pairs(detection)["only"], (3.0, 4.0))

    def test_top_level_object_with_x_and_y_alone_holds_no_keypoints(self):
        """A coordinate is not a keypoint set; nothing usable means None."""
        self.assertIsNone(parse({"x": 12.0, "y": 34.0, "confidence": 0.9}))

    def test_keypoint_as_dict_or_array(self):
        detection = parse(
            [
                {
                    "keypoints": [
                        {"name": "stump", "x": 1.0, "y": 2.0},
                        [3.0, 4.0],
                    ]
                }
            ]
        )
        names = list(pairs(detection))
        self.assertEqual(names, ["stump", "point_2"])
        self.assertEqual(pairs(detection)["point_2"], (3.0, 4.0))

    def test_unreadable_entry_is_dropped_not_guessed(self):
        """A name with no coordinate inside a list is not a keypoint."""
        self.assertIsNone(parse([{"keypoints": [{"top_left": [5.0, 6.0]}]}]))

    def test_top_level_name_mapping_container(self):
        """The container itself may be `{name: [x, y]}` rather than a list."""
        for alias in ("points", "keypoints"):
            with self.subTest(alias=alias):
                detection = parse({alias: {"tl": [10.0, 20.0], "br": [30.0, 40.0]}})
                self.assertEqual(
                    pairs(detection), {"tl": (10.0, 20.0), "br": (30.0, 40.0)}
                )

    def test_name_spellings_class_and_keypoint(self):
        detection = parse(
            [
                {
                    "keypoints": [
                        {"class": "long_stop", "x": 1.0, "y": 2.0},
                        {"keypoint": "middle_stump", "x": 3.0, "y": 4.0},
                    ]
                }
            ]
        )
        self.assertEqual(list(pairs(detection)), ["long_stop", "middle_stump"])

    def test_single_keypoint_container_as_array(self):
        """One keypoint written as a bare [x, y, confidence] triple."""
        detection = parse([{"keypoints": [12.0, 34.0, 0.75]}])
        self.assertEqual(list(pairs(detection)), ["point_1"])
        self.assertEqual(detection.keypoints[0].confidence, 0.75)

    def test_single_name_mapping_container(self):
        detection = parse(
            [{"keypoints": {"long_stop": {"x": 10.0, "y": 20.0, "confidence": 0.6}}}]
        )
        self.assertEqual(pairs(detection)["long_stop"], (10.0, 20.0))
        self.assertEqual(detection.keypoints[0].confidence, 0.6)

    def test_longest_prediction_wins(self):
        """Two objects in one response: the richer one is the pitch."""
        detection = parse(
            [
                {"keypoints": [{"name": "a", "x": 1.0, "y": 1.0}]},
                {
                    "keypoints": [
                        {"name": "a", "x": 10.0, "y": 10.0},
                        {"name": "b", "x": 20.0, "y": 10.0},
                        {"name": "c", "x": 20.0, "y": 90.0},
                        {"name": "d", "x": 10.0, "y": 90.0},
                    ]
                },
            ]
        )
        self.assertEqual(len(detection.keypoints), 4)
        self.assertEqual(pairs(detection)["a"], (10.0, 10.0))


# ── Normalised vs pixel coordinates ──────────────────────────────────────────
class TestNormalisedScaling(unittest.TestCase):
    def test_unit_coordinates_become_pixels(self):
        detection = parse(
            [
                {
                    "keypoints": [
                        {"name": "tl", "x": 0.1, "y": 0.1},
                        {"name": "br", "x": 0.9, "y": 0.9},
                    ]
                }
            ]
        )
        self.assertEqual(pairs(detection)["tl"], (64.0, 36.0))
        self.assertEqual(pairs(detection)["br"], (576.0, 324.0))

    def test_mixed_coordinates_are_never_scaled(self):
        """Scaling one point and not its neighbour would bend the quad."""
        detection = parse(
            [
                {
                    "keypoints": [
                        {"name": "tl", "x": 0.1, "y": 0.1},
                        {"name": "br", "x": 576.0, "y": 324.0},
                    ]
                }
            ]
        )
        self.assertEqual(pairs(detection)["tl"], (0.1, 0.1))

    def test_pixel_coordinates_are_left_alone(self):
        detection = parse(
            [{"keypoints": [{"name": "a", "x": 100.0, "y": 200.0}]}]
        )
        self.assertEqual(pairs(detection)["a"], (100.0, 200.0))

    def test_unknown_frame_size_blocks_scaling(self):
        detection = parse(
            [{"keypoints": [{"name": "a", "x": 0.5, "y": 0.5}]}],
            image_size=(0, 0),
        )
        self.assertEqual(pairs(detection)["a"], (0.5, 0.5))


# ── Malformed input is dropped, never repaired ───────────────────────────────
class TestMalformedInput(unittest.TestCase):
    def test_missing_coordinates_are_dropped(self):
        detection = parse(
            [
                {
                    "keypoints": [
                        {"name": "broken"},
                        {"name": "good", "x": 1.0, "y": 2.0},
                    ]
                }
            ]
        )
        self.assertEqual(list(pairs(detection)), ["good"])

    def test_nan_coordinates_are_dropped(self):
        detection = parse(
            [{"keypoints": [{"name": "nan", "x": float("nan"), "y": 2.0}]}]
        )
        self.assertIsNone(detection)

    def test_short_array_is_dropped(self):
        detection = parse([{"keypoints": [[1.0], [2.0, 3.0]]}])
        self.assertEqual(list(pairs(detection)), ["point_2"])

    def test_non_string_names_are_stringified(self):
        detection = parse([{"keypoints": [{"class": 4, "x": 1.0, "y": 2.0}]}])
        self.assertEqual(list(pairs(detection)), ["4"])

    def test_empty_and_absent_responses_yield_none(self):
        for payload in (None, {}, [], {"predictions": []}, {"predictions": {}}):
            with self.subTest(payload=payload):
                self.assertIsNone(parse(payload))

    def test_prediction_without_a_keypoint_container_yields_none(self):
        self.assertIsNone(parse([{"confidence": 0.9, "x": 1.0, "y": 2.0}]))
        self.assertIsNone(parse([{"class": "pitch", "confidence": 0.9}]))

    def test_unsupported_payload_type_yields_none(self):
        self.assertIsNone(parse("not a response"))
        self.assertIsNone(parse(42))


# ── Confidence ───────────────────────────────────────────────────────────────
class TestConfidence(unittest.TestCase):
    def test_mean_over_scored_keypoints(self):
        detection = parse(
            [
                {
                    "keypoints": [
                        {"name": "a", "x": 1.0, "y": 1.0, "confidence": 0.6},
                        {"name": "b", "x": 2.0, "y": 2.0, "confidence": 0.8},
                    ]
                }
            ]
        )
        self.assertEqual(detection.confidence, 0.7)

    def test_score_and_conf_spellings(self):
        detection = parse(
            [
                {
                    "keypoints": [
                        {"name": "a", "x": 1.0, "y": 1.0, "score": 0.5},
                        {"name": "b", "x": 2.0, "y": 2.0, "conf": 0.7},
                    ]
                }
            ]
        )
        self.assertEqual(detection.confidence, 0.6)

    def test_no_confidence_anywhere_is_none_not_zero(self):
        detection = parse(
            [{"keypoints": [{"name": "a", "x": 1.0, "y": 2.0}]}]
        )
        self.assertIsNone(detection.confidence)

    def test_prediction_level_confidence_wins_over_the_mean(self):
        detection = parse(
            [
                {
                    "confidence": 0.91,
                    "keypoints": [
                        {"name": "a", "x": 1.0, "y": 1.0, "confidence": 0.5},
                        {"name": "b", "x": 2.0, "y": 2.0, "confidence": 0.5},
                    ],
                }
            ]
        )
        self.assertEqual(detection.confidence, 0.91)

    def test_unscored_neighbours_do_not_drag_the_mean(self):
        detection = parse(
            [
                {
                    "keypoints": [
                        {"name": "a", "x": 1.0, "y": 1.0, "confidence": 0.8},
                        {"name": "b", "x": 2.0, "y": 2.0},
                    ]
                }
            ]
        )
        self.assertEqual(detection.confidence, 0.8)

    def test_as_pair_rounds(self):
        self.assertEqual(Keypoint("a", 1.23456, 7.891011).as_pair(), [1.23, 7.89])


# ── The credential never escapes ─────────────────────────────────────────────
class TestScrub(unittest.TestCase):
    def test_key_is_replaced_wherever_it_appears(self):
        text = f"POST https://host/{KEY}/infer failed"
        cleaned = scrub(text, KEY)
        self.assertNotIn(KEY, cleaned)
        self.assertIn("***", cleaned)

    def test_query_string_key_is_masked_and_the_rest_kept(self):
        cleaned = scrub(f"GET /infer?api_key={KEY}&model=x failed", KEY)
        self.assertNotIn(KEY, cleaned)
        self.assertIn("api_key=***", cleaned)
        self.assertIn("&model=x", cleaned)

    def test_masking_works_without_the_literal_key(self):
        cleaned = scrub("GET /infer?api_key=abc123&model=x", None)
        self.assertEqual(cleaned, "GET /infer?api_key=***&model=x")

    def test_empty_text_is_untouched(self):
        self.assertEqual(scrub("", KEY), "")
        self.assertEqual(scrub(None, KEY), None)


# ── The client ───────────────────────────────────────────────────────────────
class _GoodClient:
    def infer(self, frame, model_id=None):
        return {
            "predictions": [
                {
                    "confidence": 0.88,
                    "keypoints": [
                        {"name": "tl", "x": 0.1, "y": 0.1},
                        {"name": "tr", "x": 0.9, "y": 0.1},
                        {"name": "bl", "x": 0.1, "y": 0.9},
                        {"name": "br", "x": 0.9, "y": 0.9},
                    ],
                }
            ]
        }


class _FailingClient:
    def infer(self, frame, model_id=None):
        raise RuntimeError(f"https://host/infer?api_key={KEY}&model=x refused")


class _SlowClient:
    """A model that answers - eventually. Longer than any test ceiling."""

    def __init__(self, delay: float = 0.4) -> None:
        self.delay = delay

    def infer(self, frame, model_id=None):
        time.sleep(self.delay)
        return {"predictions": []}


def detector_with(client, **overrides) -> RoboflowPitchDetector:
    cfg = dataclasses.replace(PitchConfig(api_key=KEY), **overrides)
    det = RoboflowPitchDetector(cfg)
    det._client = client  # transport injected: no SDK, no network
    return det


class TestDetectorClient(unittest.TestCase):
    def test_no_credential_means_no_detector(self):
        det = RoboflowPitchDetector(PitchConfig(api_key=""))
        self.assertFalse(det.available)
        with self.assertRaises(PitchDetectorUnavailable):
            det.detect(None, 1, SIZE)

    def test_a_credential_alone_is_not_enough_without_the_sdk(self):
        """`available` must not raise just because the SDK is missing."""
        det = RoboflowPitchDetector(PitchConfig(api_key=KEY))
        try:
            from inference_sdk import InferenceHTTPClient  # noqa: F401
        except Exception:
            self.assertFalse(det.available)
        else:  # pragma: no cover - exercised only where the SDK is installed
            self.assertTrue(det.available)

    def test_successful_call_is_parsed(self):
        det = detector_with(_GoodClient())
        detection = det.detect(b"frame", 11, SIZE)
        self.assertEqual(detection.frame_number, 11)
        self.assertEqual(detection.image_size, SIZE)
        self.assertEqual(len(detection.keypoints), 4)
        self.assertEqual(pairs(detection)["tl"], (64.0, 36.0))
        self.assertEqual(detection.confidence, 0.88)
        self.assertIsNone(det.disabled_reason)

    def test_response_without_keypoints_is_an_error(self):
        det = detector_with(_GoodClient())
        det._client = type(
            "Empty", (), {"infer": lambda self, frame, model_id=None: {"predictions": []}}
        )()
        with self.assertRaises(PitchDetectError):
            det.detect(b"frame", 1, SIZE)

    def test_transport_errors_are_scrubbed(self):
        det = detector_with(_FailingClient())
        with self.assertRaises(PitchDetectError) as ctx:
            det.detect(b"frame", 1, SIZE)
        message = str(ctx.exception)
        self.assertNotIn(KEY, message)
        self.assertIn("api_key=***", message)

    def test_three_consecutive_timeouts_disable_the_detector(self):
        det = detector_with(_SlowClient(), request_timeout_seconds=0.05)
        for attempt in range(1, 4):
            with self.assertRaises(PitchTimeout) as ctx:
                det.detect(b"frame", attempt, SIZE)
            self.assertIn(str(attempt), str(ctx.exception))
            if attempt < 3:
                self.assertIsNone(det.disabled_reason)
        self.assertEqual(det.disabled_reason, "timeout")
        self.assertFalse(det.available)

    def test_a_successful_answer_resets_the_timeout_counter(self):
        """
        Timeouts 1 and 3 and 5, with answers in between: the counter must fall
        back to zero on every success, or the detector would disable itself on
        a model that simply is slow some of the time.
        """
        slow = _SlowClient(delay=0.4)

        class Alternating:
            def __init__(self) -> None:
                self.calls = 0

            def infer(self, frame, model_id=None):
                self.calls += 1
                if self.calls % 2 == 1:
                    return slow.infer(frame, model_id)
                return _GoodClient().infer(frame, model_id)

        det = detector_with(Alternating(), request_timeout_seconds=0.05)
        for attempt in range(1, 6):
            if attempt % 2 == 1:
                with self.assertRaises(PitchTimeout):
                    det.detect(b"frame", attempt, SIZE)
                self.assertIsNone(
                    det.disabled_reason,
                    f"timeout #{attempt} counted past a successful answer",
                )
            else:
                det.detect(b"frame", attempt, SIZE)
                self.assertIsNone(det.disabled_reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
