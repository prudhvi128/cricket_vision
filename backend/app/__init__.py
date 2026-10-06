"""
app — Cricket Video Analysis backend.

Single-pass cricket analysis:
    one decode + one YOLO/Kalman pass over the uploaded video, from which
    tracking data, delivery clips, shot labels and overlays are all produced.

Package layout, and who owns what:

  core/               configuration, constants, logging. No ML logic.
  schemas/            the persisted contract. Imported by services AND the API.
  pipeline/           `UnifiedPipeline` — stage order and persistence only.
  tracking/           YOLO detection, Kalman tracking, and the single decode.
  segmentation/       where a delivery starts and ends, and whether to serve it.
  shot_classification/ the Adarsh shot model, unmodified.
  analytics/          speed/bounce/length/line, and the calibration honesty gate.
  video/              frames as pixels: buffer, clips, overlay, probing.
  storage/            durable JSON writes.
  api/                HTTP. `routes/` per resource, `helpers.py` for shared rules.
"""

__all__ = []