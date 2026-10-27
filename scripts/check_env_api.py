#!/usr/bin/env python3
"""Actual GPU integration: direct pose action, truncation, D1 seed reset and close."""
import sys
import numpy as np

from robotic_cutting_sim.env import CuttingEnv,EnvConfig
from robotic_cutting_sim.recording import new_output,write_json,manifest

output=new_output(sys.argv[1])
write_json(output/"manifest.json",manifest({"seed":91,"variant":"D1","max_steps":2},"live_api_integration"))
env=CuttingEnv(EnvConfig(case="settle",variant="D1",max_steps=2))
obs,info=env.reset(seed=91)
initial=env.debug_material_state()["positions_m"].copy()
target=env.tool_pose.copy();target[0]+=.001
obs,_,_,done,_=env.step(target)
assert not done and np.allclose(env.tool_pose,target)
obs,_,_,done,_=env.step(target)
assert done and set(obs)=={"rgb","depth_m","valid_mask","calibration","timestamp_s"}
for name in ["d455_0","d455_1"]:
    assert obs["rgb"][name].shape==(480,640,3)
    assert obs["depth_m"][name].dtype==np.float32
    assert np.isfinite(obs["depth_m"][name]).all()
    assert np.array_equal(obs["valid_mask"][name],obs["depth_m"][name]>0)
env.reset(seed=91)
assert np.array_equal(initial,env.debug_material_state()["positions_m"])
env.close()
try:
    env.step()
    raise AssertionError("Step after close did not fail")
except RuntimeError:
    pass
result={"status":"PASS","scope":"GPU direct blade action, live observation, truncation, same-D1-seed reset and close",
        "particle_count":len(initial),"seed":91,"force_in_observation":False,"robot_dynamics":False}
write_json(output/"result.json",result)
print(result)
