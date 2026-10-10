"""
test_pitch_detection_source.py - WHICH detector the pipeline runs.

The local pitch-box detector is the pitch detection source: the shipped
`backend/models/pitch_detect.pt` is the default, so a checkout with the weights
and no configuration runs it. The abandoned Roboflow keypoint path is the other
way round: a credential in `.env` is no longer enough to make the pipeline call
it, because an explicit opt-in (`CRICKET_ROBOFLOW_PITCH`) is required too.

These tests hold both defaults, plus the OFF spelling the suite itself relies
on. What the detector produces once it runs - the estimate, its refusal rules,
and the fact that it can never calibrate anything - is proven by
test_pitch_box_slot.py, test_pitch_box_corners.py and
tests/integration/test_pitch_box_feature.py.
"""

from __future__ import annotations

import json
import unittest

from app.core.config import (
    MODELS_DIR,
    PITCH_BOX_CONFIG,
    PitchBoxConfig,
    PitchConfig,
)
from app.core.constants import PITCH_BOX_DEFAULT_WEIGHTS, PITCH_BOX_OFF_TOKENS
from app.pitch.local_box_detector import describe_pitch_box, pitch_box_enabled
from app.pitch.service import PitchService

W, H = 320, 180
KEY = "a-key-that-must-never-be-persisted"


class TestLocalBoxIsTheDefault(unittest.TestCase):
    def test_the_shipped_weights_are_the_default_model(self):
        self.assertEqual(PitchBoxConfig().model, PITCH_BOX_DEFAULT_WEIGHTS)

    def test_the_default_is_enabled(self):
        self.assertTrue(PitchBoxConfig().enabled)

    def test_the_default_resolves_inside_the_models_directory(self):
        path = PitchBoxConfig().model_path
        self.assertIsNotNone(path)
        self.assertEqual(path.parent, MODELS_DIR)

    def test_the_shipped_weights_really_exist(self):
        """A default that names a file this checkout does not have would log
        an error on every analysis instead of running the detector."""
        self.assertTrue(PitchBoxConfig().model_path.is_file())

    def test_the_resolver_reports_the_default_as_on(self):
        self.assertTrue(pitch_box_enabled(PitchBoxConfig()))
        self.assertIn("ON", describe_pitch_box(PitchBoxConfig()))


class TestLocalBoxCanBeTurnedOff(unittest.TestCase):
    def test_the_off_spelling_disables_the_feature(self):
        for token in sorted(PITCH_BOX_OFF_TOKENS):
            cfg = PitchBoxConfig(model=token)
            self.assertFalse(cfg.enabled, token)
            self.assertIsNone(cfg.model_path, token)

    def test_an_empty_model_is_also_off(self):
        self.assertFalse(PitchBoxConfig(model="").enabled)
        self.assertIsNone(PitchBoxConfig(model="").model_path)

    def test_the_off_message_names_the_switch(self):
        self.assertIn("OFF", describe_pitch_box(PitchBoxConfig(model="off")))

    def test_another_weights_file_still_enables_it(self):
        cfg = PitchBoxConfig(model="some_other.pt")
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.model_path, MODELS_DIR / "some_other.pt")

    def test_the_suite_runs_with_the_feature_off(self):
        """conftest.py sets `off`: without it every tracking test would load a
        22 MiB model on a worker thread."""
        self.assertFalse(PITCH_BOX_CONFIG.enabled)


class TestRoboflowIsOptIn(unittest.TestCase):
    def test_a_credential_alone_enables_nothing(self):
        cfg = PitchConfig(api_key=KEY)
        self.assertFalse(cfg.remote_enabled)
        self.assertFalse(cfg.enabled)

    def test_the_opt_in_needs_a_credential_too(self):
        self.assertFalse(PitchConfig(remote_enabled=True).enabled)
        self.assertTrue(PitchConfig(api_key=KEY, remote_enabled=True).enabled)

    def test_the_production_service_submits_nothing_with_a_credential(self):
        """`.env` still holds a ROBOFLOW_API_KEY. Without the opt-in the
        service must not build a client and must not submit a single frame."""
        svc = PitchService(PitchConfig(api_key=KEY), width=W, height=H)
        self.assertFalse(svc.enabled)
        for frame in range(5):
            svc.on_frame(frame, None)
        summary = svc.summary()
        self.assertEqual(summary["submitted"], 0)
        self.assertEqual(summary["diagnostics"]["gated"], {"disabled": 5})
        self.assertIsNone(
            svc._thread, "a disabled service must not start a worker"
        )

    def test_the_metadata_says_the_path_is_off(self):
        metadata = PitchConfig(api_key=KEY).metadata()
        self.assertFalse(metadata["enabled"])
        self.assertFalse(metadata["remote_enabled"])
        self.assertTrue(metadata["api_key_configured"])

    def test_the_credential_never_reaches_the_metadata(self):
        payload = json.dumps(PitchConfig(api_key=KEY).metadata())
        self.assertNotIn(KEY, payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
