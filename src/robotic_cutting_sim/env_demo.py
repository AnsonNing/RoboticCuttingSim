"""Record a short real environment episode without training any policy."""
import csv
import time
from dataclasses import asdict

import numpy as np
from PIL import Image

from .env import CuttingEnv,EnvConfig
from .recording import manifest,write_json


def run(args,output):
    config=EnvConfig(case=args.case,max_steps=args.steps,device=args.device,voxel_size=args.voxel_size)
    write_json(output/"manifest.json",manifest({**asdict(config),"seed":args.seed},"live_newton_ideal_rgbd_env"))
    camera_path=output/"observations"; camera_path.mkdir()
    global_path=output/"global_debug"; global_path.mkdir()
    def save_observation(obs,index,global_rgb):
        np.savez_compressed(camera_path/f"{index:04d}.npz",timestamp_s=obs["timestamp_s"],
                            rgb=np.stack(list(obs["rgb"].values())),depth_m=np.stack(list(obs["depth_m"].values())),
                            valid_mask=np.stack(list(obs["valid_mask"].values())))
        for name,rgb in obs["rgb"].items():
            Image.fromarray(rgb).save(camera_path/f"{name}_{index:04d}.png")
        Image.fromarray(global_rgb).save(global_path/f"{index:04d}.png")
    started=time.perf_counter()
    with CuttingEnv(config) as env:
        obs,info=env.reset(seed=args.seed)
        write_json(output/"reset_info.json",info); write_json(output/"calibration.json",obs["calibration"])
        save_observation(obs,0,env.render())
        rows=[]
        for i in range(args.steps):
            obs,reward,terminated,truncated,step_info=env.step()
            rows.append({"step":i+1,"time_s":obs["timestamp_s"],"reward":reward,
                         "physics_wall_s":step_info["physics_wall_s"],"render_wall_s":step_info["render_wall_s"],
                         "max_projection_correction_m":step_info["max_projection_correction_m"]})
            if (i+1)%15==0 or i+1==args.steps:
                save_observation(obs,i+1,env.render())
                print(f"live step={i+1}/{args.steps} t={obs['timestamp_s']:.3f}s",flush=True)
            if terminated or truncated:
                break
        oracle=env.debug_material_state()
        oracle_path=output/"oracle_debug"; oracle_path.mkdir()
        np.savez_compressed(oracle_path/"final_material.npz",**oracle)
        keys=set(obs)
        expected={"rgb","depth_m","valid_mask","calibration","timestamp_s"}
        if keys!=expected or set(obs["rgb"])!={"d455_0","d455_1"}:
            raise AssertionError("Unexpected official observation fields")
    with open(output/"timing.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={"status":"PASS","scope":"live reset/step/observe/render/close with real MPM and ideal RGB-D",
            "steps":len(rows),"simulation_s":rows[-1]["time_s"],"episode_wall_s":time.perf_counter()-started,
            "observation_keys":sorted(keys),"rgb_shape_per_camera":[480,640,3],"depth_shape_per_camera":[480,640],
            "global_in_observation":False,"force_in_observation":False,"oracle_in_observation":False,
            "material_deformation":True,"ideal_rgbd":True,"camera_contains":["material","kinematic tool","tray","table"],
            "robot_visuals_in_ideal_camera":False,"d455_rtx":False,"robot_dynamics":False,
            "persistent_cutting":False,"truncated":bool(truncated),"reward_semantics":"constant0, no training/scoring claim"}
    write_json(output/"result.json",result)
    return result
