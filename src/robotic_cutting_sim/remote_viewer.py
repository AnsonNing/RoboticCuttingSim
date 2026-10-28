"""Kit GUI for ONE existing LiveIsaacSimCutting / TLMPM environment."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import numpy as np

from .isaac_sim_runner import LiveIsaacSimCutting
from .live_env import CuttingTaskEnv
from .recording import manifest, write_json
from .remote_control import RemoteControlLoop
from .remote_runtime import ViewportCameraPreview, configure_manual_physics


class RemoteCuttingSession:
    def __init__(self, app, config, output):
        import omni.timeline
        self.app, self.config, self.output = app, config, Path(output)
        self.reset_count = 0
        self.viewport = None
        self.last_view = {}
        self.events = (self.output/"events.jsonl").open("a", buffering=1)
        self.backend = LiveIsaacSimCutting(app, config, self.output)
        self.env = CuttingTaskEnv(self.backend, config.task_id)
        self.loop = RemoteControlLoop(self.env, self.reset_task, set_camera=self.select_camera, on_event=self.event)
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline_subscription = None
        self.reset_task(config.task_id)
        self.timeline_subscription = self.timeline.get_timeline_event_stream().create_subscription_to_pop(self.timeline_event)
        self.widgets = {}
        self.build_ui()
        self.last_status_wall = 0.

    def event(self, row):
        def default(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(type(value).__name__)
        self.events.write(json.dumps({"wall_time_s": time.time(), "pid": os.getpid(), **row}, default=default)+"\n")

    def reset_task(self, task_id):
        if task_id not in ("CUT-01", "TRIM-01"):
            raise ValueError("Choose CUT-01 or TRIM-01")
        self.loop.loading = True
        try:
            if self.reset_count:
                self.archive_episode()
            episode = self.output/"episodes"/f"{self.reset_count:03d}_{task_id.lower()}"
            episode.mkdir(parents=True)
            self.config = replace(self.config, task_id=task_id,
                                  particle_spacing_m=.002 if task_id == "TRIM-01" else None)
            self.backend.config, self.backend.output = self.config, episode
            write_json(episode/"manifest.json", manifest(asdict(self.config), "live_isaac_sim_remote_interactive"))
            self.env.reset(seed=self.config.seed)
            self.manual_physics = configure_manual_physics()
            self.viewport = ViewportCameraPreview(self.backend)
            self.last_view = self.viewport.select(self.loop.camera)
            self.reset_count += 1
            if hasattr(self, "action_text"):
                self.current_action(False)
            self.record_observation()
            self.event({"event": "task_reset", "task_id": task_id, "episode": str(episode),
                        "simulation_s": self.env.time_s, "same_application_pid": os.getpid()})
        finally:
            self.loop.loading = False

    def archive_episode(self):
        self.loop.last_evaluation = self.env.evaluate()
        write_json(self.backend.output/"task_evaluation.json", self.loop.last_evaluation)
        write_json(self.backend.output/"camera_pose_trace.json", {"samples": self.backend.camera_pose_trace})
        self.backend.stage.Export(str(self.backend.output/"final_live_scene.usda"))

    def record_observation(self):
        frame = self.backend.output/"rgbd"
        frame.mkdir(exist_ok=True)
        index = self.backend.step_index
        payload = {"timestamp_s": self.env.time_s}
        for name, values in self.env._frames.items():
            for field in ("rgb", "depth_m", "valid_mask"):
                payload[name+"_"+field] = values[field]
        np.savez_compressed(frame/f"frame_{index:05d}.npz", **payload)
        write_json(frame/f"calibration_{index:05d}.json", self.backend.calibration)

    def select_camera(self, name):
        self.last_view = self.viewport.select(name)

    def timeline_event(self, event):
        import omni.timeline
        if self.loop.loading:
            return
        names = {int(omni.timeline.TimelineEventType.PLAY): "play",
                 int(omni.timeline.TimelineEventType.PAUSE): "pause",
                 int(omni.timeline.TimelineEventType.STOP): "reset"}
        if event.type in names:
            self.loop.request(names[event.type], source="native_timeline")

    def request(self, command, **values):
        self.loop.request(command, source="omni_ui", **values)

    def build_ui(self):
        import omni.ui as ui
        self.window = ui.Window("RoboticCuttingSim Controls", width=374, height=650)
        self.window.position_x, self.window.position_y = 12, 70
        def button(key, label, callback):
            widget = ui.Button(label, height=28, clicked_fn=callback)
            self.widgets[key] = widget
            return widget
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Live robot + material + RGB-D", height=24)
                with ui.HStack(height=30, spacing=6):
                    for task in ("CUT-01", "TRIM-01"):
                        button("task_"+task, task, lambda task=task: self.request("reset", task_id=task))
                self.status_label = ui.Label("Loading", height=48, word_wrap=True)
                with ui.HStack(height=30, spacing=4):
                    for name in ("play", "pause", "step", "reset"):
                        button(name, name.title(), lambda name=name: self.request(name))
                ui.Separator(height=8)
                ui.Label("Physical tool / finger contact", height=22)
                with ui.HStack(height=30, spacing=4):
                    for name in ("hold", "close", "open", "release"):
                        button("gripper_"+name, name.title(), lambda name=name: self.request("gripper",mode=name))
                self.grasp_label=ui.Label("Legacy tool or grasp diagnostics pending",height=40,word_wrap=True)
                ui.Separator(height=8)
                ui.Label("Viewport", height=22)
                with ui.HStack(height=30, spacing=4):
                    for label, camera in (("Free", "free"), ("Global", "global_d455"),
                                          ("Wrist 0", "wrist_d405_0"), ("Wrist 1", "wrist_d405_1")):
                        button("camera_"+camera, label, lambda camera=camera: self.request("camera", camera=camera))
                ui.Label("Free view supports orbit / pan / zoom.", height=22)
                ui.Separator(height=8)
                ui.Label("Action: XYZ + XYZW (metres), or 7 joints (rad)", height=24, word_wrap=True)
                self.action_mode = ui.ComboBox(0, "Cartesian knife-edge pose", "Joint positions", height=26)
                self.action_text = ui.StringField(height=30)
                self.widgets["action_text"] = self.action_text
                with ui.HStack(height=30, spacing=6):
                    button("current_cartesian", "Current Cartesian", lambda: self.current_action(False))
                    button("current_joints", "Current joints", lambda: self.current_action(True))
                    button("clear_action", "Clear", self.clear_action)
                with ui.HStack(height=24):
                    self.cut_enabled = ui.CheckBox(width=24)
                    ui.Label("Enable cutting for explicit action", height=24)
                button("apply_action", "Apply action (1 step)", self.submit_action)
                button("evaluate", "Evaluate task", lambda: self.request("evaluate"))
                self.result_label = ui.Label("Task success is evaluated separately.", height=46, word_wrap=True)
                ui.Label("D405: ideal prototype\nGlobal: provisional native D455", height=40, word_wrap=True)
                self.error_label = ui.Label("", height=54, word_wrap=True)
                button("exit", "Close simulation", lambda: self.request("close"))
        self.current_action(False)

    def current_action(self, joints):
        actual = self.backend.proprioception()
        values = actual["q"][0] if joints else actual["tool_pose"]
        self.action_mode.model.get_item_value_model().set_value(1 if joints else 0)
        self.action_text.model.set_value(json.dumps(np.asarray(values).round(7).tolist()))

    def submit_action(self):
        mode = self.action_mode.model.get_item_value_model().get_value_as_int()
        self.request("action", mode="joint_position_rad" if mode else "cartesian_edge_pose_world",
                     target=self.action_text.model.get_value_as_string(),
                     cut_enabled=self.cut_enabled.model.get_value_as_bool())

    def clear_action(self):
        # A visible GUI operation also works when a browser/SDK consumes the
        # remote text field's Ctrl+A or Home/End selection shortcuts.
        self.action_text.model.set_value("")
        self.event({"event": "action_field_cleared", "source": "omni_ui"})

    def physics_clock(self):
        manager = self.backend.SimulationManager
        return int(manager.get_num_physics_steps()), float(manager.get_simulation_time())

    def update(self):
        before_control = self.backend.step_index
        before_clock = self.physics_clock()
        before_reset = self.reset_count
        self.loop.tick()
        after_clock = self.physics_clock()
        if self.reset_count == before_reset:
            steps = self.backend.step_index-before_control
            if after_clock[0]-before_clock[0] != steps*self.config.robot_substeps:
                raise RuntimeError("GUI control advanced native physics outside the coupled environment")
        if self.backend.step_index != before_control and self.reset_count == before_reset:
            self.record_observation()
            if not self.loop.playing and hasattr(self, "action_text"):
                self.current_action(bool(self.action_mode.model.get_item_value_model().get_value_as_int()))
        self.last_view = self.viewport.update()
        state = "PLAYING" if self.loop.playing else "PAUSED"
        self.status_label.text = f"{self.env.task.task_id} | {state}\nSimulation time: {self.env.time_s:.3f} s | Step: {self.backend.step_index}"
        self.error_label.text = self.loop.error
        if self.loop.last_evaluation:
            self.result_label.text = "Task score: "+self.loop.last_evaluation["status"]
        # Rendering and WebRTC input may update the GUI, but never step PhysX.
        frozen_clock = self.physics_clock()
        self.app.update()
        if self.physics_clock() != frozen_clock:
            raise RuntimeError("A GUI/render update advanced physics independently of TLMPM")
        if time.monotonic()-self.last_status_wall >= .2:
            self.write_status()
            self.last_status_wall = time.monotonic()

    def write_status(self):
        rectangles = {}
        for name, widget in self.widgets.items():
            rectangles[name] = {"x": float(widget.screen_position_x), "y": float(widget.screen_position_y),
                                "width": float(widget.computed_width), "height": float(widget.computed_height)}
        actual = self.backend.proprioception()
        grasp = {}
        if self.backend.config.tool_attachment_mode == "physical_grasp":
            monitor = getattr(self.backend, "grasp_monitor", None)
            grasp = monitor.last if monitor is not None else {"status":"FREE_TOOL"}
            self.grasp_label.text=("%s | %s\nSlip since settling %.3fmm / %.2fdeg" % (
                grasp["status"], self.backend.gripper_command,1000*grasp.get("slip_since_settling_m",0.),
                np.degrees(grasp.get("rotation_since_settling_rad",0.))))
        status = {"phase": "READY", "pid": os.getpid(), **self.loop.snapshot(),
            "tool_attachment_mode":self.backend.config.tool_attachment_mode,"privileged_grasp_debug":grasp,
            "episode": str(self.backend.output), "native_physics_steps": self.physics_clock()[0],
            "native_physics_time_s": self.physics_clock()[1], "material_time_s": self.backend.material.time_s,
            "surface_timestamp_s": self.backend.surface_receipt.get("physics_timestamp_s", self.env.time_s),
            "robot_q": actual["q"], "actual_tool_pose_world_xyzw": actual["tool_pose"],
            "rgbd_timestamps_s": {name: values.get("physics_timestamp_s", values.get("timestamp_s")) for name, values in self.env._frames.items()},
            "camera_ids": list(self.env._frames), "viewport": self.last_view,
            "action_input": {"mode": self.action_mode.model.get_item_value_model().get_value_as_int(),
                "json_text": self.action_text.model.get_value_as_string(),
                "cut_enabled": self.cut_enabled.model.get_value_as_bool()},
            "ui_regions": rectangles, "ui_region_coordinates": "Kit framebuffer pixels, top-left origin",
            "manual_physics": self.manual_physics, "same_live_simulation": True,
            "policy_force_oracle": False, "wall_time_s": time.time()}
        temporary = self.output/"status.tmp.json"
        write_json(temporary, status)
        temporary.replace(self.output/"status.json")

    def close(self):
        self.timeline_subscription = None
        self.archive_episode()
        self.events.close()
