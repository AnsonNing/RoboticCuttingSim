# RoboticCuttingSim

已建立可供後續研究接續的模擬原型：**雙 Franka Panda、雙 RealSense D455 官方 USD 資產、Global Camera、桌面／治具、連續軟質試件和刀具**，並提供真正執行 Newton MPM 的 Python 環境介面及即時理想 RGB-D。沒有加入 RSI 或訓練任何 Policy。

目前刀具由運動學軌跡或直接 pose action 控制；兩臂是工作站中的靜態資產。材料可變形、壓入與刮推；**持久材料切斷／切屑分離、機械臂雙向耦合及 Isaac RTX D455 深度仍待後續完成**。這些能力清楚標記，不阻止使用現有場景開展研究。

原始[實作規劃](RoboticCuttingSim_Implementation_Plan.md)保留供參考。依使用者後續指示，先交付可用場景和介面，完整科學驗收另行記錄於 [PROGRESS](docs/PROGRESS.md)。

![本機組合的官方雙臂與雙相機工作站](docs/evidence/workcell_preview.png)

安裝見 [SETUP](docs/SETUP.md)。在本機已建立的隔離環境內，一條指令生成完整工作站、跑一集並輸出相機資料和 USD 回放：

```bash
cd /home/ningan/RoboticCuttingSim
CUDA_VISIBLE_DEVICES=1 bash scripts/run_research_demo.sh artifacts/my_research_demo
```

第一次會下載約21 MiB官方 USD 組合資產至 gitignored cache。請先確認 GPU1空閒。輸出的 research_workcell_replay.usda 包含雙臂／雙相機的靜態工作站及實測材料／刀具運動；它是錄製的回放，開啟 USD 本身不會啟動 MPM。也可分別操作：

```bash
.venv/bin/python -m robotic_cutting_sim.cli scene --preset workcell --output-dir artifacts/my_workcell
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m robotic_cutting_sim.cli env-demo --steps 120 --output-dir artifacts/my_episode
.venv/bin/python -m pytest -q
```

研究程式可直接呼叫：

```python
from robotic_cutting_sim.env import CuttingEnv, EnvConfig

with CuttingEnv(EnvConfig(case="indent_unload")) as env:
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step()  # reference trajectory
    # Or env.step([x_m, y_m, z_m, qx, qy, qz, qw]) for a direct blade pose.
    rgb = obs["rgb"]["d455_0"]
    depth_m = obs["depth_m"]["d455_0"]
    frame = env.render()  # global debug view, outside formal observation
```

d455_0/1 是固定的觀測接口名稱；目前輸出來自明確標記的理想 Warp 相機，包含材料、刀具、治具和桌面，尚未渲染 Franka。完整工作站 USD 保留真實 D455 數位資產及其原生相機 hierarchy，方便後續接入 RTX。正式 observation 不含 force 或 oracle。

目前主機可跑 standalone MPM＋理想 RGB-D；Isaac Lab3.0／Sim6.2完整 CUDA13/RTX 環境仍受 driver 與安裝限制，見 [COMPATIBILITY](docs/COMPATIBILITY.md)。

詳見 [環境接口](docs/ENVIRONMENT_API.md)、[工作站](docs/WORKCELL.md)、[材料實驗](docs/reports/WP-01.md)、[solver 調查](docs/reports/SOLVER_SELECTION.md)、[相機管線](docs/CAMERA_PIPELINE.md)。所有輸出使用新路徑，保留失敗紀錄；大型 artifacts、第三方資產和虛擬環境不提交 Git。
