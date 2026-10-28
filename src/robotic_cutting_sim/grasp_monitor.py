"""Operational grasp status from measured motion, separate from task scoring.

The reference is a *measured, already verified* settled grasp. No body pose or
drive target is changed here. Exact poses/contact data remain privileged debug
information. These operational loss gates do not replace the frozen physics
acceptance tests or turn an episode into a successful cut.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .scene_frames import invert_transform, matrix_pose, pose_matrix


@dataclass(frozen=True)
class GraspLossConfig:
    translation_warning_m: float = .002
    rotation_warning_rad: float = np.pi/60
    translation_loss_m: float = .020
    rotation_loss_rad: float = np.pi/6
    unsupported_loss_s: float = .1

    def __post_init__(self):
        values = np.asarray(list(self.__dict__.values()), float)
        if (not np.isfinite(values).all() or np.any(values <= 0)
                or self.translation_warning_m >= self.translation_loss_m
                or self.rotation_warning_rad >= self.rotation_loss_rad):
            raise ValueError("Grasp warning/loss gates must be positive and ordered")


class GraspMonitor:
    """Track slip since settling; loss remains latched until an explicit reset."""

    def __init__(self, hand_pose_xyzw, tool_pose_xyzw, spec, *, config=None):
        self.reference = invert_transform(pose_matrix(hand_pose_xyzw)) @ pose_matrix(tool_pose_xyzw)
        self.nominal = spec.T_hand_tool.copy()
        self.config = config or GraspLossConfig()
        self.last_time_s, self.unsupported_s, self.lost = 0., 0., False
        self.last = self._report(self.reference, "HOLDING", False, 1., True)

    def _report(self, relative, status, lost, fraction, valid):
        delta = invert_transform(self.reference) @ relative
        nominal_delta = invert_transform(self.nominal) @ relative
        return {"status": status, "grasp_lost": bool(lost),
            "slip_since_settling_m": float(np.linalg.norm(delta[:3, 3])),
            "rotation_since_settling_rad": float(Rotation.from_matrix(delta[:3, :3]).magnitude()),
            "nominal_grasp_offset_m": float(np.linalg.norm(nominal_delta[:3, 3])),
            "nominal_grasp_offset_rad": float(Rotation.from_matrix(nominal_delta[:3, :3]).magnitude()),
            "bilateral_loaded_tool_fraction": float(fraction), "contacts_valid": bool(valid),
            "unsupported_duration_s": self.unsupported_s,
            "reference_hand_from_tool_xyz_xyzw": matrix_pose(self.reference),
            "loss_scope": "operational episode termination; independent physics/task scores remain separate"}

    def update(self, hand_pose_xyzw, tool_pose_xyzw, *, time_s,
               bilateral_loaded_tool_fraction, contacts_valid=True, gripper_command="hold"):
        now, fraction = float(time_s), float(bilateral_loaded_tool_fraction)
        if (not np.isfinite([now, fraction]).all() or now <= self.last_time_s
                or not 0 <= fraction <= 1):
            raise ValueError("Grasp samples need increasing time and a contact fraction in [0, 1]")
        dt, self.last_time_s = now-self.last_time_s, now
        relative = invert_transform(pose_matrix(hand_pose_xyzw)) @ pose_matrix(tool_pose_xyzw)
        delta = invert_transform(self.reference) @ relative
        translation = np.linalg.norm(delta[:3, 3])
        angle = Rotation.from_matrix(delta[:3, :3]).magnitude()
        g = self.config
        self.unsupported_s = self.unsupported_s+dt if fraction == 0 else 0.
        warning = translation > g.translation_warning_m or angle > g.rotation_warning_rad
        self.lost |= bool(translation > g.translation_loss_m or angle > g.rotation_loss_rad
                         or (warning and self.unsupported_s >= g.unsupported_loss_s))
        status = ("INVALID_CONTACT_DATA" if not contacts_valid else "LOST" if self.lost
                  else "RELEASING" if gripper_command in ("open", "release")
                  else "SLIPPING" if warning or fraction < .95 else "HOLDING")
        self.last = self._report(relative, status, self.lost, fraction, contacts_valid)
        return self.last.copy()
