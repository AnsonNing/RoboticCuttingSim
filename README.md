# RoboticCuttingSim

雙 Franka Panda 的物理切削研究環境：連續軟質材料、固定刀具與治具、TLMPM 永久材料分離、雙路 RealSense D455 模擬 RGB-D、Global Camera，以及共用控制與任務評分 API。第二隻 Franka 目前停泊；本期沒有訓練模型。

**CUT-01 與 TRIM-01 已各自在 Isaac Lab 3.0 GA 與 Isaac Sim 6.2 完整實跑 400 步／8 秒，產生永久材料分離。** 刀具由實際 Franka 關節與末端控制帶動，量測刀刃驅動材料，材料反作用回饋手臂。沒有刪除粒子或預分割凸起。

| 固定任務 | 實際物理結果，Lab 與 Sim 各一集 | 獨立任務評分 |
| --- | --- | --- |
| CUT-01：160×30×20 mm 長條完整切斷 | 144 g 連續長條 → 兩個 72 g 部分；切割位置誤差上界 1.5 mm | **PASS_TASK_METRICS** |
| TRIM-01：連續單凸起表面切削 | 明示 2 mm 粒子配置；1.608 g 切屑；移除 98.53% 初始凸起材料 | 永久分離通過；嚴格全表面精度 **FAIL_TASK_METRICS** |

固定任務都保持質量／粒子數，撤刀後的分離持續 2.4 秒。原始 1 mm TRIM 未完全斷開、延長刀路與第二刀的失敗均保留。材料與斷裂參數未經實物校準，尚未證明解析度或耦合能量收斂。完整 benchmark 的表面精度、耦合功、原生深度尾端與相機 reset 仍有 FAIL，不能將 episode 完成當成全面驗收通過。

![TRIM-01：Live Isaac Sim 6.2，雙 Franka 與長條切削場景，t=4.8s](docs/evidence/trim01_live_kit.png)

以下影片直接編碼自 **Live Isaac Sim 執行時捕捉的原生相機陣列**，每集 41 幀；來源與 SHA256 隨附。不是 USD replay。

| 任務 | Global | 雙 RGB | 雙量測 Depth | 同步視角／軌跡／來源 |
| --- | --- | --- | --- | --- |
| CUT-01 | [影片](docs/media/cut01/global.mp4) | [D455 0](docs/media/cut01/d455_0_rgb.mp4)／[D455 1](docs/media/cut01/d455_1_rgb.mp4) | [D455 0](docs/media/cut01/d455_0_depth.mp4)／[D455 1](docs/media/cut01/d455_1_depth.mp4) | [同步](docs/media/cut01/synchronized_multiview.mp4)／[Commanded vs Actual](docs/media/cut01/trajectory.png)／[來源](docs/media/cut01/provenance.json) |
| TRIM-01 | [影片](docs/media/trim01/global.mp4) | [D455 0](docs/media/trim01/d455_0_rgb.mp4)／[D455 1](docs/media/trim01/d455_1_rgb.mp4) | [D455 0](docs/media/trim01/d455_0_depth.mp4)／[D455 1](docs/media/trim01/d455_1_depth.mp4) | [同步](docs/media/trim01/synchronized_multiview.mp4)／[Commanded vs Actual](docs/media/trim01/trajectory.png)／[來源](docs/media/trim01/provenance.json) |

## 執行兩個場景

使用本機已建立的隔離環境與新的輸出目錄；安裝及版本見 [SETUP](docs/SETUP.md)、[Isaac Lab](docs/ISAAC_LAB_INTEGRATION.md)、[Isaac Sim](docs/ISAAC_SIM_VALIDATION.md)。下列會真正推進機器人與材料，保存實際軌跡、物理 trace、RGB-D、Global 與獨立評分。

```bash
# Isaac Sim 6.2 / PhysX / 原生 SingleView RGB-D
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id CUT-01 --steps 400 --gpu 1 --device cuda:1 \
  --camera-mode native_single_view --output-dir artifacts/cut01_fresh

.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id TRIM-01 --particle-spacing 0.002 --steps 400 --gpu 1 --device cuda:1 \
  --camera-mode native_single_view --output-dir artifacts/trim01_fresh

# 相同 Task，在 Isaac Lab 3.0 / NewtonManager 實跑；Kit-less 相機為 ideal Warp
.deps/isaaclab-venv/bin/python scripts/run_isaac_lab_cutting.py \
  --task-id TRIM-01 --particle-spacing 0.002 --steps 400 --reset-check \
  --output-dir artifacts/lab_trim01_fresh

# 依序執行兩個任務的固定與隨機化場景；不進行訓練
.venv/bin/python scripts/run_cutting_task_benchmark.py \
  --runtime isaac_sim --variants fixed randomized --seeds 42 --gpu 1 \
  --output-dir artifacts/task_matrix_fresh
```

CUT 使用 1 mm 粒子，成功 TRIM 使用明示的 2 mm 初始取樣；材料網格均為 4 mm。API／Task 原始 1 mm 預設仍保留，重跑成功 TRIM 必須指定 `--particle-spacing 0.002`。目前單集 Sim 約 86–253 秒、Lab 約 197–458 秒；首次啟動與硬體負載另計，不保證即時效能。

隨機化支援不同長度、局部切割位置、靜態工件位置／yaw、1–3 個 Dome／Gaussian／Ridge／不規則連續凸起。已實跑 native seed42：CUT 分成 66.6／59.4 g；TRIM 三個凸起產生三塊切屑，移除 96.28% 目標材料。這是單一 seed 的工程驗證，不是隨機任務成功率。見 [Task benchmark](docs/CUTTING_BENCHMARK.md) 與[隨機化設定](docs/TASK_RANDOMIZATION.md)。

## 共用研究 API

兩個任務共用 `reset()`、`step(action)`、`observe()`、`render()`、`close()`、`evaluate()`；支援 Cartesian 刀刃 pose 與 joint position actions。正式觀測只有雙 RGB-D／校正／時間與可選 robot proprioception，force／材料 oracle／Global 留給獨立 debug 與評分。

```python
from robotic_cutting_sim.live_env import open_cutting_env

# 使用該 runtime 的 Python；完整設定見 SHARED_TASK_API.md
with open_cutting_env("TRIM-01", runtime="isaac_sim",
                      particle_spacing_m=0.002,
                      output_dir="artifacts/my_trim_api") as env:
    obs, info = env.reset(seed=42)
    for _ in range(400):
        obs, reward, terminated, truncated, info = env.step()  # scripted reference
    result = env.evaluate()  # 獨立評分；完整表面標準仍可能 FAIL
```

Object、Material、Fixture、Robot、Camera、Physics 與 Task Goal 分離；目標定義在 workpiece local frame。後續 RL／IL／VLA／Agent 可接入對應控制與評分 adapter，沿用目前切削物理。動態持物、任意新切面及雙臂協作仍需擴充。

詳細：[共用 API](docs/SHARED_TASK_API.md)、[Task 定義／成功標準](docs/CUTTING_TASKS.md)、[實際 episodes 與負結果](docs/reports/CUTTING_TASK_EPISODES.md)、[相機驗證](docs/TASK_CAMERA_VALIDATION.md)、[表面表示誤差](docs/REPRESENTED_TARGET_VALIDATION.md)、[完整進度](docs/PROGRESS.md)。

## 回歸與既有基線

```bash
.venv/bin/python -m pytest -q
```

205 項 CPU 測試通過；與實際 GPU／Isaac episodes 分開計算。D0–D3 的固定種子物體取樣 hash 保持一致；另完成 Lab D0 的 5 步實際物理前綴回歸，不冒充完整重跑。歷史 standalone、D0–D3、Newton implicit MPM 與 CRESSim 證據全部保留，入口見 [原 benchmark](docs/BENCHMARK_REPORT.md)、[TLMPM 驗證](docs/reports/TLMPM_VALIDATION.md)、[D0 Live 驗收](docs/reports/ISAAC_LIVE_ACCEPTANCE.md) 與 [D0 影片](docs/media/live_isaac_kit_d0/synchronized_multiview.mp4)。大型原始資料與官方資產留在本機 ignored cache。
