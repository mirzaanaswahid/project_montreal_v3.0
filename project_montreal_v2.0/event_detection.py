#!/usr/bin/env python3
"""Event detection module for UAV simulation"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any, Tuple

from events import GroundEvent

class FOVCone:
    """Field of View cone for UAV with altitude-dependent characteristics"""
    def __init__(
        self,
        angle_deg: float = 90.0,         # full cone angle in degrees
        min_range: float = 50.0,         # Minimum slant-range (m)
        max_slant_range: float = 1000.0, # Maximum slant-range (m)
        min_alt: float = 50.0            # Minimum altitude for detection (m)
    ):
        # half‑angle in radians
        self.half_angle = math.radians(angle_deg) / 2.0
        self.cos_half  = math.cos(self.half_angle)
        self.min_range = min_range
        self.max_slant_range = max_slant_range
        self.min_alt   = min_alt

    def get_ground_fov(self, altitude: float) -> float:
        """Horizontal footprint radius on ground at given altitude."""
        if altitude < self.min_alt:
            return 0.0
        return altitude * math.tan(self.half_angle)

    def contains_circle(
        self,
        uav_pos: np.ndarray,      # [x, y, z]
        uav_heading: float,       # radians, 0 = +y axis (North)
        center: np.ndarray,       # [x, y, 0]
        radius: float             # event radius on ground (m)
    ) -> Tuple[bool, float]:
        """
        Check if any part of a circle (center + radius) lies within the FOV cone.
        Returns (is_detected, slant_range_to_center).
        """
        # vector from UAV to event center
        v = center - uav_pos
        slant_center = np.linalg.norm(v)

        # quick reject if entire disk is out of slant-range band
        if slant_center - radius > self.max_slant_range or (slant_center + radius) < self.min_range:
            return False, slant_center

        # horizontal projection & distance
        hor = np.array([v[0], v[1], 0.0])
        d_hor = np.linalg.norm(hor)
        ground_fov = self.get_ground_fov(uav_pos[2])
        # reject if entire disk is outside horizontal footprint
        if d_hor - radius > ground_fov:
            return False, slant_center

        # forward unit vector in ENU with 0 rad = +Y/North (matches agent.py)
        forward = np.array([math.sin(uav_heading), math.cos(uav_heading), 0.0])

        # angle between forward and horizontal vector to center
        if d_hor > 1e-8:
            cos_center = np.dot(hor / d_hor, forward)
            cos_center = max(-1.0, min(1.0, cos_center))
            ang_center = math.acos(cos_center)
            # angular radius subtended by the event disk at UAV
            ang_radius = math.asin(min(1.0, radius / d_hor))
        else:
            # UAV is almost directly above center → full coverage
            ang_center = 0.0
            ang_radius = math.pi / 2.0

        # detect if the cone half-angle overlaps [ang_center ± ang_radius]
        return (ang_center - ang_radius) <= self.half_angle, slant_center


class DetectionRecord:
    """Record of a single event detection"""
    def __init__(
        self,
        event: GroundEvent,
        first_detection_time: float,
        detection_range: float,
        uav_altitude: float
    ):
        self.event = event
        self.first_detection_time = first_detection_time
        self.detection_range     = detection_range
        self.uav_altitude        = uav_altitude


class EventDetector:
    """Detects ground events within UAV's field of view"""
    def __init__(
        self,
        fov_angle: float       = 90.0,    # degrees
        min_alt:   float       = 50.0,    # m
        max_alt:   float       = 800.0,   # m
        min_range: float       = 50.0,    # m
        max_slant_range: float = 1000.0   # m
    ):
        self.fov = FOVCone(
            angle_deg       = fov_angle,
            min_range       = min_range,
            max_slant_range = max_slant_range,
            min_alt         = min_alt
        )
        self.min_alt = min_alt
        self.max_alt = max_alt
        self.detected_events: Dict[str, DetectionRecord] = {}

    def check_events(
        self,
        pos: np.ndarray,            # UAV position [x, y, z]
        heading: float,             # UAV heading in radians
        events: List[GroundEvent],
        current_time: float
    ) -> List[Tuple[GroundEvent, float]]:
        """
        Scan `events` and return newly detected ones as (event, slant_range_to_center).
        An event is marked detected as soon as any part of its ground disk overlaps the FOV cone.
        """
        # only detect when UAV within altitude envelope
        if not (self.min_alt <= pos[2] <= self.max_alt):
            return []

        new_detections: List[Tuple[GroundEvent, float]] = []

        for ev in events:
            if not ev.active(current_time):
                continue

            center = np.array([ev.cx, ev.cy, 0.0])
            detected, slant = self.fov.contains_circle(pos, heading, center, ev.radius)
            if detected and ev.id not in self.detected_events:
                new_detections.append((ev, slant))
                self.detected_events[ev.id] = DetectionRecord(
                    event               = ev,
                    first_detection_time= current_time,
                    detection_range     = slant,
                    uav_altitude        = pos[2]
                )

        return new_detections

    def get_detection_stats(self) -> Dict[str, Any]:
        """Return summary of all events detected so far."""
        return {
            'total_detections': len(self.detected_events),
            'detections': [
                {
                    'event_id'      : eid,
                    'level'         : rec.event.level,
                    'detection_time': rec.first_detection_time,
                    'range'         : rec.detection_range,
                    'altitude'      : rec.uav_altitude
                }
                for eid, rec in self.detected_events.items()
            ]
        }
