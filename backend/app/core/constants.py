"""
constants.py — Project-wide constants.

Three categories:

  1. Values ported verbatim from CricketTracker-main/backend/model.py and
     tracker_modules/*. These define the DETECTOR and the short-range Kalman
     smoother. Changing one changes how well the ball is followed.

  2. Values introduced by docs/UNIFIED_ARCHITECTURE.md for the single-pass
     architecture (ring buffer sizing, clip window, codec fallback chain).

  3. Delivery-segmentation thresholds (SEG_*). These define where ONE BATTING
     EVENT ends and the next begins. They are expressed in SECONDS and in
     FRACTIONS OF THE FRAME DIAGONAL, then converted to frames/pixels from the
     video's own fps and resolution.

     THEY ARE NOT DELIVERY COUNTS. No constant anywhere in this project states
     how many deliveries a video contains, because that number is an OUTPUT of
     the segmentation, never an input. A video with 41 contiguous footage
     regions may contain any number of batting events; the algorithm decides.

     Nothing here was fitted to a particular video. See
     docs/DELIVERY_SEGMENTATION_DIAGNOSTIC.md.
"""

# ── Delivery segmentation (ported verbatim from CricketTracker model.py) ───────
# A track with fewer confirmed detections than this is not a delivery. This is a
# MINIMUM QUALITY BAR, not a segmenter: it says "there is not enough evidence
# here to call this a delivery", not "deliveries are N frames long".
MIN_DELIVERY_FRAMES = 8

# DEPRECATED — retained only so existing imports keep resolving. It is NO LONGER
# USED AS A SEGMENTATION RULE.
#
# It used to DISCARD any delivery starting within 60 frames (2.4 s) of the
# previous one. That was wrong in two separate ways:
#
#   * it silently DELETED a real batting event that happened to follow quickly,
#     so a delivery could vanish from the output entirely; and
#   * 60 is a bare frame count, i.e. 2.4 s at 25 fps but only 1.2 s at 50 fps,
#     so the same video segmented differently at a different frame rate.
#
# Segmentation is now event-based (services/event_segmenter.py) and driven by
# seconds plus kinematic evidence. Boundary decisions are recorded per event
# with their reasons; uncertain ones are flagged, never dropped.
MIN_FRAMES_BETWEEN_DELIVERIES = 60

# ── Ball detection (ported verbatim from CricketTracker tracker_modules/detector.py) ──
DETECTOR_CONF_THRESHOLD = 0.30
DETECTOR_RESIZE_W = 640
DETECTOR_RESIZE_H = 360
DETECTOR_MIN_DIAG_PX = 4
DETECTOR_MAX_DIAG_PX = 80

# ── Kalman tracking (ported verbatim from CricketTracker tracker_modules/tracker.py) ──
MAX_TRAIL = 120
MAX_MISSED = 8
GATE_RADIUS = 120

# ── Speed fitting (ported verbatim from CricketTracker tracker_modules/trajectory.py) ──
MAX_EARLY_POSITIONS = 12
SPEED_MIN_KMH = 40.0
SPEED_MAX_KMH = 200.0

# Speed calibration, ported verbatim from trajectory.py:110.
#     kmph = speed_mps * 3.6 * 2.0 - 20.0
# This is a heuristic fudge factor, NOT a physical measurement — see
# UNIFIED_ARCHITECTURE.md §10 D2. It is persisted in the analysis metadata and
# surfaced in the API so no consumer can mistake the result for a calibrated
# speed reading.
SPEED_CALIBRATION_SCALE = 2.0
SPEED_CALIBRATION_OFFSET_KMH = -20.0

# ── Pitch geometry (ported verbatim from CricketTracker tracker_modules/utils.py) ──
PITCH_LENGTH_M = 20.12
CREASE_WIDTH_M = 2.64

# Homography corner fractions, measured off one specific side-on broadcast
# template. Re-measure if camera framing changes; every speed is wrong otherwise.
HOMOGRAPHY_CORNER_FRACTIONS = {
    "TL": (0.418, 0.268),
    "TR": (0.560, 0.268),
    "BL": (0.432, 0.655),
    "BR": (0.572, 0.655),
}

# ── Shot classification (ported verbatim from CricketShot-Classification) ─────
SHOT_CLASSES = [
    "Cover",
    "Defense",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted",
    "Pull",
    "Square Cut",
    "Straight",
    "Sweep",
]
SHOT_N_FRAMES = 30
SHOT_IMAGE_SIZE = 224
SHOT_TEMPORAL_DIM = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Length / line zones (ported verbatim from CricketTracker trajectory.py) ────
LENGTH_ZONES = [
    ("Beamer", 0.00, 0.30),
    ("Bouncer", 0.30, 0.45),
    ("Short", 0.45, 0.58),
    ("Good Length", 0.58, 0.72),
    ("Full", 0.72, 0.83),
    ("Yorker", 0.83, 1.00),
]

LINE_ZONES = [
    ("Wide Leg", 0.00, 0.18),
    ("Leg Side", 0.18, 0.38),
    ("Middle", 0.38, 0.62),
    ("Off Side", 0.62, 0.82),
    ("Wide Off", 0.82, 1.00),
]

# FIVE zones, expressed where cricket expresses them: METRES from the batting
# crease. This is the table used whenever the geometry is genuinely measured
# (operator-measured corners, or the pitch-keypoint quad), because only then is
# there a ground plane to measure along.
#
# It is not a restatement of the six image-fraction zones above: a Beamer and a
# Bouncer cannot be told apart by metres from the crease, so they are dropped
# rather than guessed, and a "Very Short" band covers a bounce that lands behind
# the good-length zone. The labels that differ here are the point — the metre
# table only claims what the measurement can support.
#
# Thresholds are coaching-standard distances — a yorker lands in the blockhole,
# a good-length ball forces the batter to play from roughly 4-7 m, a short ball
# arrives after 7 m — NOT values fitted to this footage. They live here, in one
# place, so they can be audited or replaced without touching the code that
# applies them. `LENGTH_ZONES` above (fractions of FRAME HEIGHT) is the ported
# upstream rule and is no longer used for length.
LENGTH_ZONES_M = [
    ("Yorker", 0.0, 1.5),
    ("Full", 1.5, 4.0),
    ("Good Length", 4.0, 7.0),
    ("Short", 7.0, 10.0),
    ("Very Short", 10.0, PITCH_LENGTH_M),
]

# ── Pitch-keypoint quad → ground-plane metres (app/analytics/pitch_geometry.py) ──
# The quad's LONG axis must clearly dominate its SHORT axis before the axis
# assignment ("this is the 20.12 m direction") is believed. A real pitch seen
# end-on spans several times more of the frame along its length than across its
# width; a quad that is roughly square cannot say which axis is which, and
# guessing would swap the length and width scales.
PITCH_GEOMETRY_ASPECT_MIN_RATIO = 1.2

# How far the ball must travel ALONG the chosen axis, in unit-square units,
# before its direction of travel is believed. This is what decides which end of
# the quad the batter stands at; a ball whose detections barely move cannot say.
PITCH_GEOMETRY_TRAVEL_MIN_DELTA = 0.12

# Detected points required to vote on the direction of travel at all.
PITCH_GEOMETRY_MIN_TRAVEL_POINTS = 3

# ── Single-pass architecture (UNIFIED_ARCHITECTURE.md §4) ────────────────────
# Rolling window of recent frames kept so a delivery clip can start BEFORE the
# delivery was known to have started. Raw 1080p is 5.93 MiB/frame, so frames are
# JPEG-compressed; ~180 frames then costs roughly 45 MiB instead of 1.04 GiB.
RING_BUFFER_CAPACITY_FRAMES = 180
RING_BUFFER_MAX_BYTES = 512 * 1024 * 1024
RING_BUFFER_JPEG_QUALITY = 92

# Clip window around a delivery. POST_ROLL is load-bearing: the stroke happens
# AFTER the ball is lost, so a clip ending at ball_lost_frame would feed the shot
# model footage in which the bat has not yet moved.
PRE_ROLL_SECONDS = 4.0
POST_ROLL_SECONDS = 2.0

# Codec fallback chain. avc1 is preferred for browser playback but the bundled
# OpenH264 is frequently broken; mp4v is always available via OpenCV.
VIDEO_CODEC_PREFERENCE = ("avc1", "mp4v")

# Upper bound on the memory used to hold sampled clip frames for in-pass shot
# inference. Beyond this the classifier falls back to reading the written clip,
# which is explicitly permitted (clip-level fallback, not source-video re-read).
SHOT_SAMPLING_MAX_IN_MEMORY_BYTES = 256 * 1024 * 1024

# ── Shot checkpoint resolution ───────────────────────────────────────────────
# The Adarsh checkpoint is `cricket_model_transformer.ckpt`. This repository
# carries it byte-identical under `shot_classifier.ckpt` (verified by MD5 against
# CricketShot-Classification-main/backend/models/cricket_model_transformer.ckpt),
# so both names resolve to the same weights. Listed in preference order so the
# provenance block can report the upstream name even when the local copy is used.
SHOT_MODEL_FILENAMES = (
    "cricket_model_transformer.ckpt",
    "shot_classifier.ckpt",
)
# What provenance reports, so a consumer sees the model it was trained as rather
# than the local filename.
SHOT_MODEL_PROVENANCE_NAME = "cricket_model_transformer"

# ── API processing stages ────────────────────────────────────────────────────
# The vocabulary published to clients. A client switches on these, so the set is
# closed and every value is emitted by the pipeline.
#
# `segmentation` and `analytics` are reported as the SUBSTAGE while the single
# tracking pass is running: in this architecture segmentation, per-delivery
# analytics, clip writing and validation are all interleaved inside that one
# decode, so promoting them to top-level stages would mean reporting a phase
# change that never happened. See `STAGE_PROGRESS_BANDS` for how progress is
# derived without faking a phase boundary.
PIPELINE_STAGES = (
    "uploading",
    "tracking",
    "validation",
    "shot_classification",
    "overlay",
    "persisting",
    "completed",
    "failed",
)

# Sub-activities reported alongside `stage` while inside the tracking pass.
TRACKING_SUBSTAGES = ("segmentation", "analytics", "clip_writing")

# Progress bands. Progress is monotonic: each band maps a STAGE's own real
# completed-unit count onto a slice of 0..100, so a later stage always starts at or
# above where the previous one finished and the number can never run backwards.
#
# These are WEIGHTS over measured work, not a claim about elapsed time. The
# tracking band is the widest because the single decode + YOLO + Kalman pass
# dominates real runtime (measured on real_cricket.mp4: ~400 s of a ~470 s run).
STAGE_PROGRESS_BANDS = {
    "uploading": (0.0, 5.0),
    "tracking": (5.0, 62.0),
    "validation": (62.0, 68.0),
    "shot_classification": (68.0, 86.0),
    "overlay": (86.0, 96.0),
    "persisting": (96.0, 100.0),
}

SCHEMA_VERSION = "2.1"
PIPELINE_VERSION = "1.1.0"


# ═══════════════════════════════════════════════════════════════════════════
# EVENT-BASED DELIVERY SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════
# The invariant these serve:
#
#     ONE DELIVERY CLIP = ONE BATTING EVENT
#
# A "batting event" is one continuous ball: released, bowled, struck or missed,
# lost. Two such events separated by a camera cut with no black frame are still
# two events, and the pixel gap between them is NOT evidence either way.
#
# HOW A BOUNDARY IS DECIDED
# -------------------------
# Evidence is per-frame tracking provenance, all of it produced by the single
# pass: was the ball CONFIRMED by YOLO, was the position only PREDICTED, how
# long has it been since the last confirmation, and does the new position lie
# where the ball could physically have got to.
#
# GAP LENGTH IS A COAST, AND A COAST HAS A CEILING
# ------------------------------------------------
# A missed detection during the SAME delivery is normal — the ball goes behind
# the bat, a fielder's arm, or simply out of focus. That must NOT split the
# delivery. But a Kalman filter will happily extrapolate forever, so prediction
# must be given a maximum meaningful continuation period, after which the event
# is over regardless of what the filter believes.
#
# All three limits below are in SECONDS and are converted with the video's own
# fps, so a 50 fps and a 25 fps video of the same match segment identically.

# A gap no longer than this is treated as a SHORT TRACKING INTERRUPTION: the
# same ball, momentarily lost. Kinematics may still override this (see the
# reachability test below), but silence alone never splits here.
SEG_MAX_COAST_SECONDS = 0.8

# A gap longer than this is a NEW BATTING EVENT, unconditionally. At 25 fps this
# is 50 frames; a cricket ball that has not been seen for 0.8 s is behind a
# player, in a pocket, or already dead.
SEG_HARD_SPLIT_SECONDS = 2.0

# Between the two limits above sits the AMBIGUOUS band: too long to be an
# occlusion, too short to trust blindly. Here the kinematics decide, and the
# event is flagged for review either way.

# ── Kinematic reachability ───────────────────────────────────────────────────
# Where could the ball physically have got to during the gap? The reachable
# radius is the last measured speed times the gap, plus slack. Tolerances are
# fractions of the frame DIAGONAL so the test behaves identically on 320x180
# test footage and 1080p broadcast footage.
SEG_REACH_BASE_FRACTION = 0.05       # 5% of the diagonal, constant slack
SEG_REACH_RATE_FRACTION = 0.004      # +0.4% of the diagonal per coasted frame

# A confirmed detection beyond the reachable radius cannot be the same ball: it
# either teleported or it is a different ball. This is the test that splits two
# events inside one contiguous footage region, with no black gap anywhere.
#
# Direction is checked too, because a ball CAN legitimately change direction —
# off the bat, off the pitch. What it cannot do is reverse. The threshold is
# generous (120 degrees) so ordinary bounce scatter does not trip it.
SEG_REVERSAL_DEGREES = 120.0

# A confirmed detection whose implied speed is wildly above or below the
# event's recent speed is a kinematic jump. A ball off the bat speeds up or
# slows down; it does not change order of magnitude frame to frame.
SEG_SPEED_JUMP_UP = 4.0
SEG_SPEED_JUMP_DOWN = 0.15

# Confirmed points used to estimate the event's reference velocity. A least
# squares fit over a window is robust to the bounce scatter that makes any
# two-point estimate useless.
SEG_VELOCITY_WINDOW = 8

# An event whose own confidence falls below this is reported as AMBIGUOUS by the
# validator rather than being asserted SINGLE_EVENT.
SEG_AMBIGUOUS_CONFIDENCE = 0.6

# A clip whose own event sits closer than this to the clip edge (in seconds) is
# reported AMBIGUOUS: the boundary may well be wrong, and no amount of
# arithmetic on the frame map can prove otherwise.
SEG_CLIP_MARGIN_SECONDS = 0.15

# ── Dynamic Bounded Clip Extraction Barriers ──────────────────────────────────
# Hard barriers prevent pre-roll and post-roll from blindly extending across
# transitions, black gaps, scene cuts, or neighboring deliveries.
BLACK_FRAME_LUMINANCE_THRESHOLD = 3.0   # Grayscale mean < 3.0 is a black frame
BLACK_FRAME_RUN_BARRIER = 3            # >= 3 consecutive black frames forms an impermeable barrier
SCENE_CUT_DIFF_THRESHOLD = 35.0         # Downsampled frame difference threshold for camera cut
MAX_PRE_ROLL_SECONDS = 3.0             # Safety ceiling for run-up backward search
MAX_POST_ROLL_SECONDS = 1.8            # Safety ceiling for follow-through forward search
PITCH_ACTIVITY_QUIET_THRESHOLD = 1.5   # Inactivity valley threshold in pitch corridor
PITCH_ACTIVITY_ACTIVE_THRESHOLD = 4.0  # Significant activity threshold in pitch corridor

# Activity windows (Tier-2 event horizons). A run of motion above the active
# threshold that ends in a sustained quiet run is one activity cluster. When it
# contains no confirmed ball detection at all it is an UNTRACKED ACTIVITY
# WINDOW: a real delivery the detector missed. It is never promoted to a
# delivery, but it becomes an exclusion zone a neighbouring clip cannot cross.
# Expressed in SECONDS so a 50 fps video behaves identically to a 25 fps one.
ACTIVITY_MIN_WINDOW_SECONDS = 0.16      # shortest motion burst worth a horizon
ACTIVITY_QUIET_HOLD_SECONDS = 0.16      # quiet run that closes an activity window

# ── Coherent weak confirmation ──────────────────────────────────────────────────
# A batting event is occasionally shredded by the association gate rather than by
# segmentation: the filter's prediction drifts, and a run of REAL YOLO detections
# of the same ball lands beyond GATE_RADIUS and is refused one frame after
# another. Those frames are genuine measurements — they are just ones the filter
# declined to use — so they are recorded as a separate, clearly labelled evidence
# channel (`FrameEvidence.candidate`, `TrackResult.gated_out`).
#
# WHEN AN EVENT MAY BE CONFIRMED ON THAT EVIDENCE
# -----------------------------------------------
# Only when the event would otherwise fail the 8-frame CONFIRMED bar, and only
# when several INDEPENDENT signals agree that the refused measurements belong to
# the open ball. An event promoted this way is marked
# `confirmation_method = "coherent_weak_detection"` and never reported as a
# standard confirmation.
#
# NOTHING HERE RELAXES A THRESHOLD
# --------------------------------
#   * DETECTOR_CONF_THRESHOLD is untouched — weak evidence is produced by the gate,
#     not by lowering the detector's confidence bar;
#   * the weak measurements never become normal detections, trajectory points,
#     Kalman inputs, or confirmed-frame counts;
#   * the TOTAL number of real measurements still has to reach
#     MIN_DELIVERY_FRAMES, so a weak confirmation is not a shorter delivery, it is
#     a delivery whose evidence is distributed differently. Asserted at import
#     below, in the same spirit as the fragment-merge floor.

# Total REAL measurements (confirmed + gate-refused) a weakly confirmed event must
# carry. Equal to MIN_DELIVERY_FRAMES on purpose: the quantity of evidence required
# does not drop, only where some of it came from.
WEAK_MIN_TOTAL_MEASUREMENTS = MIN_DELIVERY_FRAMES

# Below this many confirmed detections there is no anchor: nothing to measure a
# refused detection against, so no coherence claim can be made at all.
WEAK_MIN_CONFIRMED_FRAMES = 2

# How many gate-refused measurements the promotion actually needs. Without this an
# event holding two confirmed frames and one refusal could reach the total by
# accident.
WEAK_MIN_CANDIDATE_FRAMES = 4

# How many of the six coherence signals must agree. Two of them (temporal
# continuity and path coherence) are mandatory; the rest must supply the balance.
WEAK_MIN_SIGNAL_AGREEMENT = 4

# Largest share of the measurement span that may be silent, and the smallest share
# of consecutive measurements. Both are fractions, not frame counts.
WEAK_MAX_SILENCE_FRACTION = 0.4
WEAK_MIN_CONTIGUOUS_FRACTION = 0.6

# Slack on the weak path-coherence reachability test, as a multiplier on the
# measured travel term only (the constant + per-frame slack is unchanged). The
# test is over an event the gate already shredded, so the measured term is
# allowed to be generous; the constant terms still carry the absolute tolerance.
WEAK_REACH_TOLERANCE = 1.5

# Corridor residency of the refused measurements, and how far apart they may sit
# before they read as two different balls rather than one.
WEAK_CORRIDOR_RATIO = 0.5
WEAK_SPREAD_FRACTION = 0.35

# Implied speed of the refused measurements, relative to the event's own measured
# speed. A ball off the bat accelerates and decelerates; a ratio of 5x or 1/5th
# between adjacent frames is not cricket.
WEAK_SPEED_RATIO_DOWN = 0.2
WEAK_SPEED_RATIO_UP = 5.0

# Confidence stamped on a promoted event. Below a standard confirmation (1.0) and
# above SEG_AMBIGUOUS_CONFIDENCE, because the boundary is not in doubt — the
# boundary is measured — but the detection path is weaker and a consumer should be
# able to see that.
WEAK_CONFIRMATION_CONFIDENCE = 0.7

# ── Fragment merging ──────────────────────────────────────────────────────────
# A real batting event is occasionally cut into two or three candidates, because
# the reference velocity fitted to the open event can be badly wrong across a
# bounce and the reachability test then declares the returning ball physically
# impossible. Measured on real_cricket.mp4: candidates #34+#35+#36 held 2+7+2
# confirmed detections of ONE stroke and #43+#44 held 5+1.
#
# Merging is a REPAIR for that shredding, not a second segmentation. It runs
# only between ADJACENT candidates, only when one of them is too thin to be a
# delivery on its own, and only when continuity is re-verified from BOTH sides.
# Two candidates that each already qualify as deliveries are never merged.

# Longest silence that may be bridged by a merge. Strictly inside the hard-split
# limit: beyond that a new ball has certainly been bowled.
FRAGMENT_MERGE_MAX_GAP_SECONDS = 1.2

# A merge needs this many confirmed detections in total. It EQUALS
# MIN_DELIVERY_FRAMES and is asserted as such at import: merging may not become a
# back door around the delivery threshold.
FRAGMENT_MERGE_MIN_COMBINED_FRAMES = MIN_DELIVERY_FRAMES

# Corridor residency at which a fragment is treated as "in the pitch corridor"
# for the compatibility test. A ball that travels the pitch spends nearly its
# whole flight inside it; a fielder or a fielder's boot does not.
FRAGMENT_MERGE_CORRIDOR_RATIO = 0.55

# Slack on the merge reachability test, as fractions of the frame diagonal. A
# little tighter than the in-event test on purpose: merging has to survive being
# wrong in a way that splitting cannot, since it produces one clip from two
# candidates instead of merely dropping one.
FRAGMENT_MERGE_REACH_BASE_FRACTION = 0.05
FRAGMENT_MERGE_REACH_RATE_FRACTION = 0.004

# The pitch corridor, in fractions of frame width and height. Same box the
# per-frame pitch-motion ROI uses (tracking_service), stated as fractions so it
# means the same thing on 320x180 test footage and 1080p broadcast footage.
PITCH_CORRIDOR = (0.30, 0.70, 0.20, 0.80)  # x0, x1, y0, y1

# Merging may never become a way around the delivery threshold. Asserted at import
# so lowering one constant without the other fails loudly instead of quietly
# admitting thinner deliveries.
assert FRAGMENT_MERGE_MIN_COMBINED_FRAMES == MIN_DELIVERY_FRAMES, (
    "fragment merging must not lower the delivery threshold: "
    f"{FRAGMENT_MERGE_MIN_COMBINED_FRAMES} != {MIN_DELIVERY_FRAMES}"
)

# Coherent weak confirmation may not become a way around the delivery threshold
# either. The floor counts EVERY real measurement — confirmed plus gate-refused —
# so a weakly confirmed event carries at least as much evidence as a standard one;
# it simply carries part of it through the evidence-only channel.
assert WEAK_MIN_TOTAL_MEASUREMENTS == MIN_DELIVERY_FRAMES, (
    "coherent weak confirmation must not lower the delivery threshold: "
    f"{WEAK_MIN_TOTAL_MEASUREMENTS} != {MIN_DELIVERY_FRAMES}"
)


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONAL PITCH KEYPOINT DETECTION (Roboflow)
# ═══════════════════════════════════════════════════════════════════════════
# An auxiliary, NON-BLOCKING detector that finds the pitch's keypoints in a
# frame so a client can draw the pitch boundary and place the ball on it. It is
# strictly auxiliary: the ball detector, the Kalman tracker, segmentation and
# every analytic measurement behave identically whether this runs or not, and a
# failure here can never fail an analysis.
#
# The model is optional. Without ROBOFLOW_API_KEY every value below is inert
# (see app/pitch/service.py), and the response simply reports
# `detected: false, calibrated: false`.

# Hosted inference endpoint. The SDK treats any URL under these hosts as the
# legacy (v0) API, which is where `project/version` model ids are served.
ROBOFLOW_API_URL_DEFAULT = "https://serverless.roboflow.com"
PITCH_MODEL_ID_DEFAULT = "cricketpitchkeypointdetections/3"

# Frames between submissions. The loop calls `PitchService.on_frame()` every
# frame and the service drops all but one in N, so the tracking pass pays a
# couple of integer comparisons per frame rather than an HTTP round trip.
# 10 frames ≈ one submission every 0.4 s at 25 fps.
PITCH_DETECTION_INTERVAL_FRAMES = 10

# A keypoint must clear this confidence to count. Sent to the server as
# `keypoint_confidence` AND re-checked locally, because the server's threshold
# is not this codebase's threshold.
PITCH_MIN_CONFIDENCE = 0.5

# Fewer than this many valid keypoints means the pitch quad cannot be trusted
# (and below four there is no homography at all — see PITCH_HOMOGRAPHY_MIN_KPS).
PITCH_MIN_KEYPOINTS = 3
PITCH_HOMOGRAPHY_MIN_KPS = 4

# The keypoints' convex hull must cover at least this share of the frame.
# Guards against a model that answers with near-degenerate points (all four in
# a corner, or all collapsed onto one another), which would produce a homography
# that maps the whole video into a sliver of the pitch.
PITCH_MIN_AREA_FRACTION = 0.01

# How many frames after the last GOOD detection the calibration is still served
# (marked `using_previous_calibration`) before it expires outright. At 25 fps,
# 50 frames = 2 s of camera stability after which a stale quad is no longer
# assumed to describe the current view.
PITCH_MAX_LOST_FRAMES = 50

# EMA weight for temporal smoothing of the keypoints. 1.0 would be raw model
# output; lower values trade responsiveness for a boundary that does not jitter
# frame to frame.
PITCH_SMOOTHING_ALPHA = 0.4

# Longest a single Roboflow call may run before it is recorded as a failure.
# Requests are made on a background thread, so a slow call never stalls the
# tracking loop — this only bounds how long the state machine waits before
# giving up on that attempt.
PITCH_REQUEST_TIMEOUT_SECONDS = 15.0

# Frames the client may downsize an input to before sending. Smaller payload,
# faster upload; the SDK maps the returned keypoints back to the original frame
# coordinates, so the numbers in the response are full-resolution pixels.
PITCH_MAX_INPUT_SIZE = 1024

# ── Submission strategy ──────────────────────────────────────────────────────
# Three modes, both measured rather than assumed:
#
#   interval  every `PITCH_DETECTION_INTERVAL`-th frame is offered, whatever
#             happens. Predictable, and the model is asked for frames the
#             picture cannot answer (crowd shots, replays, black frames).
#   adaptive  the same base interval, multiplied by what the loop already
#             knows: is this frame part of a delivery (open event, or inside
#             the guard that runs past one), did a quad already cover this
#             view, does the pitch corridor even contain a readable pitch, and
#             has the model been answering "nothing there" for long enough that
#             the scene itself is what should change. The queue's back-pressure
#             then caps the rate at what one worker thread can actually answer,
#             so nothing is submitted only to be thrown away.
#   sequential backpressure-aware sequential processing: one inference at a
#             time, and never faster than the detector answers. A frame is
#             offered only when no call is in flight and nothing is queued
#             (so the queue never holds a backlog of stale frames), then a
#             wall-clock cooldown holds the next offer back until
#             `1 / PITCH_MAX_ATTEMPTS_PER_SEC` seconds have passed. When the
#             detector slows down the cooldown stops binding and the rate
#             follows the latency; when it speeds up the rate rises only to
#             the configured maximum.
#
# The default is `interval` — a bare `PitchConfig()` behaves exactly as before.
# `.env` opts a real analysis into `sequential`.
PITCH_SAMPLING_MODE_INTERVAL = "interval"
PITCH_SAMPLING_MODE_ADAPTIVE = "adaptive"
PITCH_SAMPLING_MODE_SEQUENTIAL = "sequential"
PITCH_SAMPLING_MODE_DEFAULT = PITCH_SAMPLING_MODE_INTERVAL
PITCH_SAMPLING_MODES = (
    PITCH_SAMPLING_MODE_INTERVAL,
    PITCH_SAMPLING_MODE_ADAPTIVE,
    PITCH_SAMPLING_MODE_SEQUENTIAL,
)

# `sequential` mode's ceiling, in detector attempts per second. It is also the
# cooldown: two submissions are at least `1 / this` apart on the wall clock.
# Measured on `real_cricket.mp4` the detector answers in ~0.48 s (p95 0.61 s),
# so 2.0/s sits at the top of the intended 1.5–2 attempts/s band, and a slower
# answer stretches the interval by itself — the sampler never outruns it.
PITCH_MAX_ATTEMPTS_PER_SEC = 2.0
# Floor for that ceiling: a configured rate below this is a misconfiguration
# rather than a policy (it would also divide by zero). The detector's own
# latency — not this value — is what slows the sampler down when answers take
# longer than the cooldown.
PITCH_ATTEMPTS_PER_SEC_FLOOR = 0.1

# Multipliers on the base interval (all in frames, all ≥ 1).
PITCH_IDLE_INTERVAL_MULTIPLIER = 8    # no delivery in progress
PITCH_COVERED_INTERVAL_MULTIPLIER = 6  # a quad already describes this view
PITCH_DIM_INTERVAL_MULTIPLIER = 2     # the corridor holds no readable pitch
PITCH_FORCE_INTERVAL_MULTIPLIER = 40  # never starve: attempt at least this often

# How far past a delivery's last frame the footage still belongs to that
# delivery. The clip window runs past the segmenter's event, and those
# trailing frames are where the model answers: every keypoint response that
# arrived outside an open event but inside a delivery window landed here, and
# no response at all ever arrived from further away than the guard.
PITCH_EVENT_GUARD_FRAMES = 50

# The model's answer rate is a property of the scene, not of the sampler.
# Measured over one instrumented run of `real_cricket.mp4`: 38% of attempts
# inside a delivery carried keypoints, 3% of attempts outside one, and 0% of
# either kind beyond the video's first third. After this many empty answers
# in a row the sampler treats the frame as a scene the model has already
# answered and goes quiet — until a scene cut opens a new one, or a delivery
# makes asking worthwhile again.
PITCH_DROUGHT_BACKOFF = 6
PITCH_DROUGHT_INTERVAL_MULTIPLIER = 40          # between deliveries: keepalive
PITCH_DROUGHT_INTERVAL_MULTIPLIER_IN_EVENT = 4  # inside a delivery: still worth asking

# Three consecutive timeouts disable the client — and used to disable it for
# the rest of the video, because nothing ever asked it again. A timeout that
# came from one bad minute of the service is not a dead credential: this many
# frames later the service gets exactly one more attempt, and three more
# timeouts put it back to sleep for another window.
PITCH_DETECTOR_REARM_FRAMES = 600

# The pitch is a bright, washed-out surface. Measured over the pitch corridor
# of `real_cricket.mp4`: frames where the model answered had a median corridor
# luminance of 126, frames where it answered with nothing had 113 — and every
# one of the sampled answers cleared 116. Below this the corridor is a crowd,
# a graphic or a night shot, and an attempt is worth half as often, not never.
PITCH_CORRIDOR_MIN_LUMINANCE = 115.0