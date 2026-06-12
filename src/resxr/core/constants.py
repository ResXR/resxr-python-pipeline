"""
Constants and enumerations for the ResXR pipeline.

This module defines tracking system types and their associated column mappings
for converting Unity/Meta Quest XR data to BIDS format.
"""

from __future__ import annotations

from enum import Enum

# Name of the recorder's global engine clock column (ResXR CSV format
# contract). Every validation check, masking step, and report onset
# computation keys on this single constant.
GLOBAL_CLOCK_COLUMN = "timeSinceStartup"


class TrackingSystem(Enum):
    """
    Enumeration of XR tracking subsystems.

    Each system represents a distinct tracking modality from the Meta Quest
    headset with its own sampling characteristics.
    """

    HEAD = "Head"
    HANDS = "Hands"
    EYES = "Eyes"
    FACE = "Face"
    BODY = "Body"
    CONTROLLERS = "Controllers"


# Column prefixes that identify which tracking system a column belongs to
SYSTEM_COLUMN_PREFIXES: dict[TrackingSystem, list[str]] = {
    TrackingSystem.HEAD: [
        "Node_Head_",
        "FocusedObject",
        "RecenterCount",
        "TrackingLost",
        "UserPresent",
        "recenterEvent",
        "shouldRecenter",
        GLOBAL_CLOCK_COLUMN,
        "TrackingOriginChange_",
        "TrackingTransform_",
    ],
    TrackingSystem.EYES: [
        "EyeGazeHitPosition_",
        "RightEye_",
        "LeftEye_",
        "Node_EyeCenter_",
        "Eyes_Time",
        "LeftEyeGazeHitPosition_",
        "RightEyeGazeHitPosition_",
        "LeftFocusedObject",
        "RightFocusedObject",
        "HasLeftEyeHit",
        "HasRightEyeHit",
    ],
    TrackingSystem.HANDS: [
        "Node_HandLeft_",
        "Node_HandRight_",
        "LeftHand_",
        "RightHand_",
        "Left_XRHand_",
        "Right_XRHand_",
    ],
    TrackingSystem.FACE: [
        "Face_",
        "Brow_",
        "Cheek_",
        "Chin_",
        "Dimpler",
        "Eyes_Closed",
        "Eyes_Look",
        "Inner_Brow",
        "Jaw_",
        "Lid_",
        "Lip_",
        "Lips_",
        "Lower_Lip",
        "Mouth_",
        "Nose_",
        "Outer_Brow",
        "Upper_Lid",
        "Upper_Lip",
        "Tongue_",
        "FaceRegionConfidence",
    ],
    TrackingSystem.BODY: [
        "Body_Time",
        "Body_Confidence",
        "Body_Fidelity",
        "Body_CalibrationStatus",
        "Body_SkeletonChangedCount",
        "Body_",
    ],
    TrackingSystem.CONTROLLERS: [
        "Node_ControllerLeft_",
        "Node_ControllerRight_",
    ],
}


# BIDS channel type inference patterns: (suffix, channel_type, component, units)
BIDS_CHANNEL_PATTERNS: list[tuple[str, str, str, str]] = [
    # Position columns
    ("_px", "POS", "x", "m"),
    ("_py", "POS", "y", "m"),
    ("_pz", "POS", "z", "m"),
    ("HitPosition_X", "POS", "x", "m"),
    ("HitPosition_Y", "POS", "y", "m"),
    ("HitPosition_Z", "POS", "z", "m"),
    # Orientation/quaternion columns
    ("_qx", "ORNT", "quat_x", "n/a"),
    ("_qy", "ORNT", "quat_y", "n/a"),
    ("_qz", "ORNT", "quat_z", "n/a"),
    ("_qw", "ORNT", "quat_w", "n/a"),
    # Position (XRHand-style _x/_y/_z; after quaternion so Palm_qx matches ORNT first)
    ("_x", "POS", "x", "m"),
    ("_y", "POS", "y", "m"),
    ("_z", "POS", "z", "m"),
    # Latency/timing columns
    ("_Time", "LATENCY", "n/a", "s"),
    # Validity/status columns
    ("_Present", "MISC", "n/a", "boolean"),
    ("_Flags_OrientationValid", "MISC", "n/a", "boolean"),
    ("_Flags_PositionValid", "MISC", "n/a", "boolean"),
    ("_Flags_OrientationTracked", "MISC", "n/a", "boolean"),
    ("_Flags_PositionTracked", "MISC", "n/a", "boolean"),
    ("_Valid_Position", "MISC", "n/a", "boolean"),
    ("_Valid_Orientation", "MISC", "n/a", "boolean"),
    ("_Tracked_Position", "MISC", "n/a", "boolean"),
    ("_Tracked_Orientation", "MISC", "n/a", "boolean"),
    ("_IsValid", "MISC", "n/a", "boolean"),
    ("_Confidence", "MISC", "n/a", "normalized"),
    ("_Status", "MISC", "n/a", "boolean"),
    # Face expression columns
    ("Confidence_Lower", "MISC", "n/a", "normalized"),
    ("Confidence_Upper", "MISC", "n/a", "normalized"),
]


# Column suffixes to strip when extracting tracked point names
COLUMN_SUFFIXES: list[str] = [
    # Position
    "_px",
    "_py",
    "_pz",
    "HitPosition_X",
    "HitPosition_Y",
    "HitPosition_Z",
    # Orientation
    "_qx",
    "_qy",
    "_qz",
    "_qw",
    # Status flags
    "_Flags_OrientationValid",
    "_Flags_PositionValid",
    "_Flags_OrientationTracked",
    "_Flags_PositionTracked",
    "_Valid_Position",
    "_Valid_Orientation",
    "_Tracked_Position",
    "_Tracked_Orientation",
    "_Present",
    "_Time",
    "_IsValid",
    "_Confidence",
    # Generic xyz
    "_x",
    "_y",
    "_z",
    "_X",
    "_Y",
    "_Z",
]


# BIDS channel type count keys for motion.json sidecar
BIDS_CHANNEL_TYPE_COUNTS: dict[str, str] = {
    "ACCEL": "ACCELChannelCount",
    "ANGACCEL": "ANGACCELChannelCount",
    "GYRO": "GYROChannelCount",
    "JNTANG": "JNTANGChannelCount",
    "LATENCY": "LATENCYChannelCount",
    "MAGN": "MAGNChannelCount",
    "MISC": "MISCChannelCount",
    "ORNT": "ORNTChannelCount",
    "POS": "POSChannelCount",
    "VEL": "VELChannelCount",
}
