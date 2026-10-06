"""
test_frontend_serializer.py — The internal document → frontend schema projection.

These are unit tests over `app.schemas.frontend`: no HTTP, no pipeline, one
synthetic internal delivery record. What they pin down:

  * the exact field set the browser receives — nothing more, however rich the
    internal record is;
  * filesystem paths never survive into a response;
  * unavailable measurements come out null rather than invented;
  * the detected/predicted distinction is preserved point by point;
  * verbose validation collapses into concise flags plus one boolean;
  * the array order a Previous/Next client indexes by is the document's order.
"""

from __future__ import annotations

import unittest

from app.schemas.frontend import DeliveryOut, serialize_delivery, serialize_result_document

ANALYSIS_ID = "abc123"


def internal_delivery(**overrides) -> dict:
    """A complete internal delivery record, as `result.json` stores it."""
    delivery = {
        "delivery_id": 22,
        "delivery_ref": "delivery_022",
        "index": 22,
        "detection_start_frame": 7986,
        "detection_end_frame": 8081,
        "ball_lost_frame": 8060,
        "release_frame": 8001,
        "clip_path": "C:\\Users\\LENOVO\\Desktop\\New folder\\data\\runs\\x\\clips\\delivery_022.mp4",
        "clip_url": "/data/runs/x/clips/delivery_022.mp4",
        "clip": {
            "start_frame": 7986,
            "end_frame": 8081,
            "frame_count": 96,
            "frame_offset": 7985,
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "codec": "avc1",
            "frames_written": 96,
            "frame_map_valid": True,
            "frame_map_error": None,
            "pre_roll_actual_sec": 0.56,
            "post_roll_frames": 50,
            "post_roll_actual_frames": 45,
            "ball_lost_frame": 8060,
            "truncated_before_frame": None,
            "truncated_reason": None,
            "start_clamped_by_previous": False,
            "url": "/data/runs/x/clips/delivery_022.mp4",
            "file_name": "delivery_022.mp4",
        },
        "trajectory": {
            "points": [
                {
                    "original_frame": 8005,
                    "clip_frame": 20,
                    "x": 781,
                    "y": 290,
                    "source": "detected",
                    "confidence": 0.4544,
                },
                {
                    "original_frame": 8006,
                    "clip_frame": 21,
                    "x": 786,
                    "y": 301,
                    "source": "predicted",
                    "confidence": None,
                },
            ],
            "point_count": 2,
            "detected_count": 1,
            "predicted_count": 1,
            "continuity": "continuous",
            "gaps": [],
        },
        "speed_samples": [
            {
                "original_frame": 8005,
                "x": 781,
                "y": 290,
                "ground_x_m": 1.2,
                "ground_y_m": 3.4,
                "t_sec": 0.0,
            }
        ],
        "bounce": {
            "detected": True,
            "original_frame": 8013,
            "x_px": 799,
            "y_px": 459,
            "x_norm": 0.62422,
            "y_norm": 0.6375,
            "ground_x_m": None,
            "ground_y_m": None,
            "method": "score",
        },
        "bowling": {
            "speed_kmh": None,
            "speed_method": None,
            "speed_calibrated": False,
            "speed_imputed": False,
            "length": None,
            "line": None,
            "swing": None,
            "release_angle": -95.3,
            "bounce_angle": None,
            "geometry_calibrated": False,
        },
        "shot": {
            "type": None,
            "confidence": None,
            "class_probabilities": {},
            "input_source": None,
            "frames_sampled": None,
            "unique_frames_sampled": None,
            "frames_repeated": None,
            "sampled_clip_frames": None,
            "sampled_original_frames": None,
            "error": None,
        },
        "overlay": {
            "path": "C:\\Users\\LENOVO\\Desktop\\New folder\\data\\runs\\x\\overlays\\delivery_022.mp4",
            "url": "/data/runs/x/overlays/delivery_022.mp4",
            "rendered": True,
            "trajectory_source": "stored",
        },
        "overlay_url": "/data/runs/x/overlays/delivery_022.mp4",
        "tracking_source": "single_pass",
        "fallback_reason": None,
        "provenance": {
            "tracking_source": "crickettracker",
            "tracking_pass": "single_pass",
            "trajectory_source": "stored_tracking",
            "shot_model": "cricket_model_transformer",
            "shot_model_input_frames": 30,
            "source_video_decodes": 1,
            "yolo_passes": 1,
            "kalman_passes": 1,
        },
        "quality": {
            "trajectory_valid": True,
            "analytics_complete": True,
            "flags": ["pitch_calibration_required"],
            "trajectory_points": 2,
            "trajectory_detected_points": 1,
            "trajectory_predicted_points": 1,
            "tracking_confidence": 0.6306,
            "frame_map_valid": True,
        },
        "event": {
            "event_index": 1,
            "start_frame": 7990,
            "end_frame": 8070,
            "boundary": {"start": 7986, "end": 8081},
            "evidence": ["confirmed", "coasting"],
            "clip_window": [7986, 8081],
            "confirmed_frame_ranges": [[7990, 8020]],
            "predicted_frame_ranges": [[8021, 8040]],
            "reacquired_frames": [8045],
            "merge_checks": [],
            "weak_confirmation": False,
        },
        "validation": {
            "delivery_id": 22,
            "clip_start": 7986,
            "clip_end": 8081,
            "classification": "SINGLE_EVENT",
            "event_count_estimate": 1,
            "event_ranges": [[7990, 8070]],
            "confidence": 1.0,
            "reason": "clip 7986-8081 contains one tracked event",
            "flags": [],
            "own_event_index": 1,
            "start_margin_frames": 14,
            "end_margin_frames": 11,
            "requires_human_review": False,
        },
    }
    delivery.update(overrides)
    return delivery


def frontend(**overrides) -> dict:
    return serialize_delivery(internal_delivery(), analysis_id=ANALYSIS_ID).model_dump()


TOP_LEVEL = {
    "delivery_id",
    "delivery_ref",
    "video",
    "trajectory",
    "bounce",
    "bowling",
    "shot",
    "quality",
}


class TestResponseShape(unittest.TestCase):
    def test_the_top_level_keys_are_exactly_the_frontend_contract(self):
        self.assertEqual(set(frontend()), TOP_LEVEL)

    def test_nested_blocks_have_exactly_the_documented_keys(self):
        body = frontend()
        self.assertEqual(
            set(body["video"]),
            {"clip_url", "start_frame", "end_frame", "fps", "width", "height",
             "duration_seconds"},
        )
        self.assertEqual(
            set(body["trajectory"]),
            {"points", "detected_points", "predicted_points", "continuity"},
        )
        self.assertEqual(
            set(body["trajectory"]["points"][0]),
            {"frame", "x", "y", "source", "confidence"},
        )
        self.assertEqual(
            set(body["bounce"]),
            {"detected", "frame", "x", "y", "ground_x_m", "ground_y_m"},
        )
        self.assertEqual(
            set(body["bowling"]),
            {"speed_kmh", "line", "length", "swing", "release_angle",
             "bounce_angle", "calibrated"},
        )
        self.assertEqual(set(body["shot"]), {"classification", "confidence"})
        self.assertEqual(
            set(body["quality"]),
            {"trajectory_valid", "tracking_confidence",
             "requires_human_review", "flags"},
        )

    def test_no_internal_field_reaches_the_response(self):
        """Whatever the pipeline records, none of it is serialised."""
        body = frontend()
        serialised = repr(body)
        for leaked in (
            "clip_path", "original_frame", "clip_frame", "speed_samples",
            "provenance", "validation", "event", "boundary", "evidence",
            "clip_window", "reacquired_frames", "confirmed_frame_ranges",
            "predicted_frame_ranges", "merge_checks", "weak_confirmation",
            "class_probabilities", "overlay", "tracking_source",
            "detection_start_frame", "ball_lost_frame", "release_frame",
            "event_ranges", "frame_offset", "gaps",
        ):
            self.assertNotIn(leaked, serialised, leaked)

    def test_a_model_rejects_an_undeclared_field(self):
        with self.assertRaises(Exception):
            DeliveryOut(**{"delivery_id": 1, "delivery_ref": "delivery_001",
                           "video": {}, "trajectory": {}, "bounce": {},
                           "bowling": {}, "shot": {}, "quality": {},
                           "provenance": {}})

    def test_values_are_carried_across_unchanged(self):
        body = frontend()
        self.assertEqual(body["delivery_id"], 22)
        self.assertEqual(body["delivery_ref"], "delivery_022")
        self.assertEqual(body["video"]["start_frame"], 7986)
        self.assertEqual(body["video"]["end_frame"], 8081)
        self.assertEqual(body["video"]["fps"], 25.0)
        self.assertEqual(body["video"]["width"], 1280)
        self.assertEqual(body["video"]["height"], 720)
        self.assertEqual(body["video"]["duration_seconds"], 3.84)
        self.assertEqual(body["trajectory"]["continuity"], "continuous")

    def test_a_point_reports_its_source_frame_and_nothing_else(self):
        point = frontend()["trajectory"]["points"][0]
        self.assertEqual(point["frame"], 8005)      # source timebase
        self.assertEqual(point["source"], "detected")
        self.assertEqual(point["confidence"], 0.4544)

    def test_detected_and_predicted_are_both_counted(self):
        traj = frontend()["trajectory"]
        self.assertEqual(traj["detected_points"], 1)
        self.assertEqual(traj["predicted_points"], 1)
        self.assertEqual(
            {p["source"] for p in traj["points"]}, {"detected", "predicted"}
        )
        self.assertEqual(
            traj["detected_points"] + traj["predicted_points"],
            len(traj["points"]),
        )


class TestPathsNeverEscape(unittest.TestCase):
    def test_the_stored_url_is_used_when_it_is_already_a_url(self):
        self.assertEqual(frontend()["video"]["clip_url"],
                         "/data/runs/x/clips/delivery_022.mp4")

    def test_a_filesystem_path_is_replaced_by_the_api_route(self):
        """A record whose clip_url is a path must not hand that path out."""
        body = serialize_delivery(
            internal_delivery(
                clip_url="C:\\Users\\LENOVO\\Desktop\\New folder\\data\\clips\\delivery_022.mp4",
                clip={**internal_delivery()["clip"], "url": None},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertEqual(
            body["video"]["clip_url"],
            f"/api/analysis/{ANALYSIS_ID}/deliveries/22/clip",
        )

    def test_no_absolute_path_survives_anywhere_in_the_payload(self):
        import json

        text = json.dumps(frontend())
        self.assertNotIn("\\", text)
        self.assertNotIn("C:", text)
        self.assertNotIn("Desktop", text)

    def test_a_delivery_without_a_clip_has_a_null_url(self):
        body = serialize_delivery(
            internal_delivery(clip=None, clip_url=None, clip_path=None),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertIsNone(body["video"]["clip_url"])
        self.assertIsNone(body["video"]["start_frame"])
        self.assertIsNone(body["video"]["duration_seconds"])
        self.assertIn("clip_unavailable", body["quality"]["flags"])


class TestNullHandling(unittest.TestCase):
    def test_uncalibrated_analytics_are_null_not_guessed(self):
        bowling = frontend()["bowling"]
        self.assertIsNone(bowling["speed_kmh"])
        self.assertIsNone(bowling["line"])
        self.assertIsNone(bowling["length"])
        self.assertIsNone(bowling["swing"])
        self.assertIsNone(bowling["bounce_angle"])
        self.assertEqual(bowling["release_angle"], -95.3)
        self.assertFalse(bowling["calibrated"])

    def test_calibrated_analytics_set_the_flag(self):
        body = serialize_delivery(
            internal_delivery(
                bowling={**internal_delivery()["bowling"],
                         "geometry_calibrated": True, "speed_kmh": 132.4},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertTrue(body["bowling"]["calibrated"])
        self.assertEqual(body["bowling"]["speed_kmh"], 132.4)

    def test_a_missing_shot_is_null_null(self):
        shot = frontend()["shot"]
        self.assertIsNone(shot["classification"])
        self.assertIsNone(shot["confidence"])

    def test_a_shot_that_failed_to_classify_is_null_null(self):
        body = serialize_delivery(
            internal_delivery(
                shot={**internal_delivery()["shot"],
                      "type": "Cover", "confidence": 0.4,
                      "error": "checkpoint missing"},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertIsNone(body["shot"]["classification"])
        self.assertIsNone(body["shot"]["confidence"])

    def test_a_ground_plane_coordinate_is_null_when_uncalibrated(self):
        bounce = frontend()["bounce"]
        self.assertIsNone(bounce["ground_x_m"])
        self.assertIsNone(bounce["ground_y_m"])
        self.assertEqual(bounce["frame"], 8013)
        self.assertEqual(bounce["x"], 799)
        self.assertEqual(bounce["y"], 459)
        self.assertTrue(bounce["detected"])

    def test_an_undetected_bounce_keeps_its_real_coordinate(self):
        """The fallback is a real tracked point; the flag says it is not a bounce."""
        body = serialize_delivery(
            internal_delivery(
                bounce={**internal_delivery()["bounce"],
                        "detected": False, "method": "fallback_max_y"},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertFalse(body["bounce"]["detected"])
        self.assertEqual(body["bounce"]["frame"], 8013)
        self.assertEqual(body["bounce"]["x"], 799)


class TestQualityCollapse(unittest.TestCase):
    def test_validation_collapses_to_flags_and_one_boolean(self):
        quality = frontend()["quality"]
        self.assertTrue(quality["trajectory_valid"])
        self.assertEqual(quality["tracking_confidence"], 0.6306)
        self.assertFalse(quality["requires_human_review"])
        self.assertEqual(quality["flags"], ["pitch_calibration_required"])

    def test_an_unambiguous_delivery_gets_no_validation_flag(self):
        flags = frontend()["quality"]["flags"]
        self.assertNotIn("multiple_events_in_clip", flags)

    def test_multiple_events_becomes_a_single_concise_flag(self):
        body = serialize_delivery(
            internal_delivery(
                validation={**internal_delivery()["validation"],
                            "classification": "MULTIPLE_EVENTS",
                            "event_count_estimate": 2,
                            "event_ranges": [[7990, 8010], [8050, 8070]],
                            "reason": "two events in the clip",
                            "requires_human_review": True},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        quality = body["quality"]
        self.assertTrue(quality["requires_human_review"])
        self.assertIn("multiple_events_in_clip", quality["flags"])
        # The verdict's working is not exposed alongside it.
        self.assertNotIn("MULTIPLE_EVENTS", repr(body))
        self.assertNotIn("event_ranges", repr(body))

    def test_an_ambiguous_clip_neither_hides_nor_overstates_itself(self):
        body = serialize_delivery(
            internal_delivery(
                validation={**internal_delivery()["validation"],
                            "classification": "AMBIGUOUS",
                            "requires_human_review": True},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertIn("ambiguous_event", body["quality"]["flags"])
        self.assertTrue(body["quality"]["requires_human_review"])

    def test_an_unvalidated_record_is_flagged_and_never_trusted(self):
        body = serialize_delivery(
            internal_delivery(validation=None), analysis_id=ANALYSIS_ID
        ).model_dump()
        self.assertTrue(body["quality"]["requires_human_review"])
        self.assertIn("unvalidated_clip", body["quality"]["flags"])

    def test_an_invalid_trajectory_forces_human_review(self):
        body = serialize_delivery(
            internal_delivery(
                quality={**internal_delivery()["quality"],
                         "trajectory_valid": False},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertFalse(body["quality"]["trajectory_valid"])
        self.assertTrue(body["quality"]["requires_human_review"])

    def test_a_broken_frame_map_is_flagged(self):
        body = serialize_delivery(
            internal_delivery(
                quality={**internal_delivery()["quality"],
                         "frame_map_valid": False},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        self.assertIn("frame_map_invalid", body["quality"]["flags"])
        self.assertTrue(body["quality"]["requires_human_review"])

    def test_flags_are_deduplicated_and_ordered(self):
        body = serialize_delivery(
            internal_delivery(
                quality={**internal_delivery()["quality"],
                         "flags": ["speed_unavailable", "speed_unavailable"]},
                validation={**internal_delivery()["validation"],
                            "classification": "AMBIGUOUS",
                            "flags": ["speed_unavailable"]},
            ),
            analysis_id=ANALYSIS_ID,
        ).model_dump()
        flags = body["quality"]["flags"]
        self.assertEqual(flags.count("speed_unavailable"), 1)
        self.assertEqual(len(flags), len(set(flags)))


class TestArrayOrdering(unittest.TestCase):
    def test_the_array_is_the_document_order_so_index_navigation_works(self):
        doc = {
            "analysis_id": ANALYSIS_ID,
            "deliveries": [internal_delivery(delivery_id=i) for i in (1, 2, 3)],
        }
        body = serialize_result_document(doc)
        self.assertEqual([d.delivery_id for d in body], [1, 2, 3])
        # Previous/Next: index ± 1 yields the neighbouring delivery whole.
        current = body[1].model_dump()
        self.assertEqual(current["delivery_id"], 2)
        self.assertEqual(set(current), TOP_LEVEL)

    def test_a_document_with_no_deliveries_is_an_empty_array(self):
        self.assertEqual(serialize_result_document({"analysis_id": "x"}), [])


if __name__ == "__main__":
    unittest.main()
