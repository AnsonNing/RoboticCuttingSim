"""Small command queue for a GUI controlling the existing coupled environment.

UI callbacks only enqueue commands. All reset, action, material, articulation
and RGB-D work runs synchronously on the same Kit thread in tick().
"""
from collections import deque
import json

import numpy as np


class RemoteControlLoop:
    def __init__(self, env, reset_task, *, set_camera=None, on_event=None):
        self.env, self.reset_task = env, reset_task
        self.set_camera = set_camera or (lambda name: None)
        self.on_event = on_event or (lambda row: None)
        self.commands = deque()
        self.playing = False
        self.loading = False
        self.closed = False
        self.camera = "global_d455"
        self.error = ""
        self.step_count = 0
        self.last_evaluation = None

    def request(self, command, **values):
        self.commands.append({"command": command, **values})

    def tick(self):
        command = self.commands.popleft() if self.commands else None
        action, advance = None, False
        if command:
            name = command["command"]
            try:
                if name == "play":
                    self.playing = True
                elif name == "pause":
                    self.playing = False
                elif name == "step":
                    self.playing, advance = False, True
                elif name == "action":
                    # Validate before advancing the original environment.
                    value = command["target"]
                    target = json.loads(value) if isinstance(value, str) else value
                    action = self.env.task.validate_action({"mode": command["mode"],
                        "target": target, "cut_enabled": bool(command.get("cut_enabled", False))})
                    self.playing, advance = False, True
                elif name == "reset":
                    self.playing = False
                    self.loading = True
                    try:
                        self.reset_task(command.get("task_id", self.env.task.task_id))
                    finally:
                        self.loading = False
                    self.step_count, self.last_evaluation = 0, None
                elif name == "camera":
                    self.set_camera(command["camera"])
                    self.camera = command["camera"]
                elif name == "evaluate":
                    self.last_evaluation = self.env.evaluate()
                elif name == "close":
                    self.playing, self.closed = False, True
                else:
                    raise ValueError("Unknown viewer command: "+name)
                self.error = ""
                self.on_event({**command, "status": "ACCEPTED", "simulation_s": self.env.time_s})
            except (ValueError, TypeError, KeyError) as error:
                self.playing = False
                self.error = str(error)
                self.on_event({**command, "status": "REJECTED", "error": self.error})
                return
        if (advance or self.playing) and not self.closed:
            if self.env.time_s >= self.env.task.duration_s-1e-9:
                self.playing = False
                return
            before = self.env.time_s
            self.env.step(action)
            if not np.isclose(self.env.time_s-before, self.env.task.control_dt_s, rtol=0, atol=1e-9):
                raise RuntimeError("The coupled environment did not advance exactly one control interval")
            self.step_count += 1
            self.on_event({"event": "coupled_step", "simulation_s": self.env.time_s,
                "action": "scripted_reference" if action is None else action,
                "task_id": self.env.task.task_id})
            if self.env.time_s >= self.env.task.duration_s-1e-9:
                self.playing = False

    def snapshot(self):
        return {"task_id": self.env.task.task_id, "simulation_s": self.env.time_s,
            "playing": self.playing, "loading": self.loading, "camera": self.camera,
            "control_steps_since_reset": self.step_count, "error": self.error,
            "evaluation": self.last_evaluation}
