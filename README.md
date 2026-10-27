# RoboticCuttingSim

獨立的可變形材料表面切削研究 testbed。依原始
[實作規劃](RoboticCuttingSim_Implementation_Plan.md)先驗證連續材料、運動學刀具、
Newton/CRESSim 和動態 RGB-D，再整合雙 Franka。

目前階段為 **blade-only prototype**；完整 Isaac Lab/Isaac Sim、持久切屑分離、
D455 RTX、雙臂和正式 benchmark 的驗收狀態見 [PROGRESS](docs/PROGRESS.md)。
沒有訓練 Policy、RL 或 RSI。

安裝與執行：[SETUP](docs/SETUP.md)。環境稽核：[COMPATIBILITY](docs/COMPATIBILITY.md)。
實驗結果：[WP-01](docs/reports/WP-01.md)、[solver 比較](docs/reports/SOLVER_SELECTION.md)、
[相機管線](docs/CAMERA_PIPELINE.md)。

```bash
source .venv/bin/activate
python -m robotic_cutting_sim.cli scene --output-dir artifacts/my_scene
python -m robotic_cutting_sim.cli physics-probe --case indent_unload --output-dir artifacts/my_indent
python -m robotic_cutting_sim.cli camera-check --input artifacts/my_indent --output-dir artifacts/my_rgbd
```

`scene` 輸出 `physics_testbed.usda`：桌面、剛性托盤、單一連續凸起試件、刀具和三個
理想相機的靜態預覽。開啟 USD 本身不會執行 MPM；`physics-probe` 才會推進物理。
本期參數為 **uncalibrated**。塑性變形或看得見刀痕，不代表已通過材料分離驗收。

不覆寫既有實驗輸出。`artifacts/`、`.deps/`、`.venv/` 不提交 Git。
Public repo 只包含程式、設定、重現工具與精簡實測報告。
