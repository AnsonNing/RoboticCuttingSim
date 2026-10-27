"""CSV trajectory with linear translation and quaternion SLERP, XYZW in world frame."""
import csv
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FIELDS = ["time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"]


@dataclass
class Trajectory:
    data: np.ndarray

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if (self.data.ndim != 2 or self.data.shape[1] != 8 or len(self.data) < 2
                or not np.isfinite(self.data).all() or np.any(np.diff(self.data[:, 0]) <= 0)):
            raise ValueError("Trajectory needs finite (N,8) rows and increasing times")
        if not np.allclose(np.linalg.norm(self.data[:, 4:], axis=1), 1, atol=1e-5):
            raise ValueError("Trajectory quaternions must be unit XYZW")
        self._rotations = Slerp(self.data[:, 0], Rotation.from_quat(self.data[:, 4:]))

    @classmethod
    def load(cls, path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError(f"CSV header must be {FIELDS}")
            return cls(np.array([[float(row[k]) for k in FIELDS] for row in reader]))

    def save(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(self.data)

    def at(self, time):
        t = np.clip(time, self.data[0, 0], self.data[-1, 0])
        i = min(np.searchsorted(self.data[:, 0], t, side="right")-1, len(self.data)-2)
        a, b = self.data[i:i+2]
        v = (b[1:4]-a[1:4]) / (b[0]-a[0])
        w = (Rotation.from_quat(b[4:]) * Rotation.from_quat(a[4:]).inv()).as_rotvec() / (b[0]-a[0])
        if time < self.data[0, 0] or time >= self.data[-1, 0]:
            v, w = np.zeros(3), np.zeros(3)
        return np.r_[a[1:4] + (t-a[0])*(b[1:4]-a[1:4])/(b[0]-a[0]),
                     self._rotations([t]).as_quat()[0]], np.r_[v, w]


def reference_trajectory(case):
    if case == "settle":
        positions = [(0, -.08, 0, .05), (2, -.08, 0, .05)]
    elif case == "indent_unload":
        # Flat punch bottom reaches 18 mm above the tray then unloads completely.
        positions = [(0, 0, 0, .040), (.30, 0, 0, .040),
                     (.80, 0, 0, .023), (1.1, 0, 0, .023),
                     (1.6, 0, 0, .040), (2, 0, 0, .040)]
    elif case in ("shallow_scrape", "through_bump_cut"):
        z = .020 if case == "shallow_scrape" else .0155
        # Thin box blade extends upwards 10 mm from its lower edge.
        positions = [(0, -.045, 0, z+.010), (.30, -.045, 0, z+.010),
                     (1.3, .045, 0, z+.010), (1.6, .060, 0, .055), (2, .060, 0, .055)]
    else:
        raise ValueError(f"Unknown case: {case}")
    return Trajectory(np.array([[t, x, y, z, 0, 0, 0, 1] for t,x,y,z in positions]))
