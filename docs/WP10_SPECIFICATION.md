# WP-10 — 尺狀刀具與可交給 Policy／RL／Agent 的切割場景

Repository: https://github.com/AnsonNing/RoboticCuttingSim
工作分支：`claude/lucid-mendel-vt7n82`（A、B 部分已完成，見第 2 節；第 3 節是剩下的工作）

## 0. 目標（使用者原話整理）

Franka 夾著刀具切割物體，完成任務（切斷、切除凸起物等）。

1. 刀具要像尺：長而扁平的直刃，由 Franka 0 兩指**物理夾持**（不是固定連接）。尺寸與次要目標由實作者判斷。
2. 機器人要能用刀具完成**各種動作**：不同角度切入、下切、推切、拉切、來回鋸切、沿表面削、斜切、用刀面推擠等。物理上做不到的動作（例如側推）要正確地「沒切開」，或出現滑刀、掉刀，而不是照樣切開。
3. 後續會用 policy、RL、RSI agent 輸出動作，狀態觀察與影像觀察都會用到。場景要可執行、不崩潰、有安全限制、失敗有回報，速度要足以做 RL。
4. 最後要有**實際的影片**：腳本 policy、學習型 policy、RL、LLM agent 實際輸出動作控制機器人、完成任務。

共通規則照 `AGENTS.md`：v1 的物理、門檻與紀錄不改；新東西版本化並可選；PASS／FAIL／BLOCKED／NOT_RUN 分開；失敗保留；不覆寫輸出；記錄指令、commit、seed、版本、wall time 與 artifact 路徑；不改驅動、不改 RSI。

## 1. 診斷（2026-10-28，更正版）

證據來源：讀程式、CPU 測試、CPU 上的 TLMPM 探測（刀刃以運動學驅動、不含機器人）、獨立的 Franka DH 運動學掃描（不是 repo 的 Newton 模型、不含碰撞）。

- **運動學**：Franka 0（base (−0.46, 0.12, −0.01)）在切割點用 CUT 的手到刀刃安裝，斜切（繞刃口線）±60°、水平轉向 360°、刃口傾斜 ±45° 都有 IK 解，沿下刀路徑每段關節變化 ≤ 0.02 rad；yaw −150° 時離關節限位只剩 0.16 rad。碰撞未檢查。
- **v1 切削規則**：斜切、轉向、刃口傾斜都能切開，切面跟著刀的方向。**更正**：先前寫「移動方向偏離刀面 15° 也照樣切斷，是錯誤」並不正確；那其實是正常的斜向推切。真正把刀面側推、或刀面平壓進材料時，v1 也**沒有切斷任何連結**（只留下單側標記）。v1 對「各種刀具動作」真正的缺口是：(a) 沿刃口的拉切／鋸切不算切削功，所以拉切幾乎不能降低下壓力；(b) 只要開啟切削，刃口後方整個窗口都會寫入單側標記，同一個 digit 之後再切時會繼承這些標記；(c) 只有刀刃會碰到材料，刀柄和手指會穿過材料；(d) runtime 只用 cut digit 0。
- **材料**：任務材料 E = 50 kPa、屈服 1 kPa、Gc = 0.05 J/m²，極軟極弱（比豆腐還軟），反作用力 < 1 N。要做真實一點的食材，需要另外的材料設定；材料越硬，顯式 MPM 的時間步就要越小（CFL）。
- **速度**：完整一集 8 s／400 步約 164 s wall；`num_envs = 1`。
- **觀察**：physical_grasp 模式本來就不把刀具真實姿態放進觀察（只給手部 FK 與手指編碼器），沒有特權資訊外洩。

問題清單（✅ = 已在分支完成並經 CPU 驗證；GPU／live runtime 一律 NOT_RUN）：

| # | 問題 | 狀態 |
|---|---|---|
| 1 | Native Cartesian IK 不收斂時丟 `RuntimeError`，整集崩潰 | ✅ 改丟 `IKFailure`；wrapper 可設為拒絕並保持（`ik_failure="reject"`） |
| 2 | Lab IK 只檢查位置 | ✅ 記錄方向誤差；`ik_rotation_tolerance_rad` 可開啟（預設 None = 舊行為） |
| 3 | 沒有相對動作、每步上限、工作空間 | ✅ `cartesian_edge_delta`、`ActionLimits`；動作可自帶 `frame` |
| 4 | reward 固定 0 | ✅ `cutting_reward_v1`（以實際被切斷的連結計算進度，單側標記不給分） |
| 5 | 速度、單一環境 | 部分：kinematic backend（無機器人）CPU 4 mm 約 0.2 s/步；Isaac 快速模式與多環境未做 |
| 6 | CLI 預設 `legacy_fixed_tool` | 未做（WP-10B） |
| 7 | 刀柄、手指穿過材料 | ✅ 刀柄為鈍障礙物（v2 任務預設）；手指障礙物要用實際手指幾何（WP-10C） |
| 8 | 拉切／鋸切不算切削 | ✅ 法則 2：沿刃口的拖曳功也算 |
| 9 | 開啟切削就寫單側標記 | ✅ 法則 2：只有刃口真的切開的分段才寫標記 |
| 10 | 一集只能一道切口 | 部分：動作可帶 `cut_index`，兩個 runtime 照傳；`max_cuts=2` 可用；v1 oracle 評估只支援一道切口（WP-10D） |
| 11 | 材料太軟、力量級偏小 | 未做（WP-10D 材料設定） |
| 12 | `tlmpm.py` kernel 參數叫 `new`，Warp CPU 不能編譯 | ✅ 在 `tlmpm_v2.py` 改名；v1 不動 |
| 13 | IK 不修正刀在手中滑動 | 未做（WP-10E，可選） |
| 14 | 刀具相容性只接受舊楔形 | ✅ `tool_profile.require_profile_consistency` |
| 15 | 新任務在 runtime 裡還不能建立 | 未做（WP-10C：Lab 設定、相機 rig、USD／Newton 刀具都還只認 CUT-01／TRIM-01） |

## 2. 已完成（分支 `claude/lucid-mendel-vt7n82`；CPU 驗證；GPU／live NOT_RUN）

使用說明：[POLICY_CONTROL.md](POLICY_CONTROL.md)。決定：[ADR 0003](adr/0003-ruler-knife-and-policy-interface.md)。報告與數字：[WP-10A](reports/WP-10A.md)。

- **控制介面**（`control.py`、`live_env.py`）：`cartesian_edge_delta`（edge 或 world 座標）、`ActionLimits`（每步 2 mm、2°，工作空間框）、IK 失敗拒絕並保持、連續失敗終止、`state` 觀察、`info` 的 `action_report`／`ik_status`、可選 reward。全部 opt-in，預設行為不變。
- **Gym**（`gym_env.py`）：Box[−1,1]^6 動作；`state` 向量或 Dict（state＋腕部 RGB-D）觀察；通過 gymnasium 的 env_checker。
- **尺狀刀**（`tool_profile.py`）：`ruler_knife_v1`，刃長 120 mm、深 40 mm、厚 1.5 mm、雙斜刃 25°、沿刀背的刀柄、刃口中心在夾持軸上、約 150 g。碰撞體、質量、TLMPM 刀刃多邊形與 `hand_from_edge` 全部由同一個 profile 產生。
- **求解器**（`tlmpm_v2.py`）：凸多邊形刀刃、CPU 可編譯；法則 1（預設）與改名後的凍結 v1 逐 bit 相同（含切割）。法則 2（opt-in，v2 任務預設）：刃口區、推切功＋沿刃口拖曳功、乘以運動與刀面夾角的 cos²、只在真的切開處寫標記；鈍障礙物（刀柄）只推不切。
- **任務 v2**（`configs/cutting_tasks_v2.json`、`tasks.ProfileTaskSpec`）：`CUT-01-R`、`TRIM-01-R`（前傾 25°，刃口是最低點、刀柄離削平面 > 6 mm）、`CUT-02`（160 × 60 × 25 mm）。門檻繼承自 v1 原任務；v1 任務的類別、收據與雜湊不變。
- **kinematic backend**（`kinematic_env.py`）：刀刃完全照指令走的理想追蹤後端，和 live runtime 共用 `LiveCuttingMaterial`；結果標 `kinematic_ideal_tracking`。
- **policy／agent**（`policies.py`、`agent_api.py`）：只讀觀察、輸出有界相對動作的 `WaypointPolicy`；`through_cut`（下切、斜切、轉向、刃口傾斜、鋸切）與 `skim`；給 LLM agent 的 JSON 工具函式。
- **驗證腳本**：`scripts/run_kinematic_validation.py`（任務 × 控制方式完整一集，用凍結的 oracle 評分）、`scripts/render_kinematic_episode.py`（粒子＋刀具影片，**不是**機器人影片）、`blade_probe.py`（切削規則探測）。
- **測試**：CPU 全套通過（`cpu_sim` 標記的模擬測試約 6 分鐘）；細節與數字在 WP-10A 報告。

## 3. 剩下的工作包（依序；完成條件就是測試標準）

所有 GPU 實跑先 `nvidia-smi`，不要碰別人的行程（GPU0 上可能有別人的 vla-sim）。如果執行環境看不到 GPU 或不能開 Isaac，把確切指令寫進報告並標 BLOCKED，不要用其他方式冒充實跑。

### WP-10B：在 GPU 上驗證控制介面

1. 兩個 runtime 各跑「隨機動作模糊測試」≥ 1000 步：`action_mode="cartesian_edge_delta"`、`ik_failure="reject"`、`ActionLimits()`，Lab 加 `ik_rotation_tolerance_rad=0.01`；動作含刻意不可達的目標、超大位移、方向翻轉。不能崩潰，每次拒絕都有 `ik_status`；Native／Lab 在相同動作序列下的 `ik_status` 一致。
2. 舊腳本基準（`action=None`、預設參數）在兩個 runtime 重現原紀錄的切斷結果與 strict 分數。
3. `cut_index` 在兩個 runtime 照實傳到材料（`max_cuts=1` 時給 1 必須報錯，不能被默默忽略）。
4. CLI preset（例如 `--preset policy`：physical_grasp、pregrasped、grip 0.0112、finite_pad_patch_v3、delta 動作、reject、Lab 方向檢查），不改舊旗標的預設值。
5. 用 Gym 介面在兩個 runtime 各跑一集（`state` 與 `vision_proprio` 各一）。

### WP-10C：尺狀刀與 v2 任務接進 runtime

1. 讓兩個 runtime 接受 v2 任務：`LabCuttingConfig.__post_init__` 目前只接受 CUT-01／TRIM-01；相機 rig（`make_rig(task_id)`）等以任務 id 查表的地方，對 v2 任務改用 `task.base_task_id`；`deployment_validation` 等同理。
2. 物理夾刀建立：兩個 runtime 已改用 `tool_profile.tool_spec_for_task(task)`（v1 任務仍是 `make_tool_spec(task)`，結果逐項相同；v2 任務是 profile 刀具），Native 另用 `require_task_tool` 檢查一致性（CPU 已測，GPU NOT_RUN）。要確認 USD／Newton 碰撞體與視覺 mesh 在 runtime 裡正確（建立程式依 `role` 通用處理 Handle／Blade）。材料直接用 `LiveCuttingMaterial(task=...)`（v2 任務自動帶 profile、法則 2、刀柄障礙物）。
3. 手指障礙物：用實際 Franka 手指墊的幾何，在刀刃座標加成 `obstacle_boxes_edge`，避免手指穿過材料；同時用實際幾何確認 TRIM-01-R 的 25° 前傾讓手指、刀柄、腕部相機都不會碰到削平面（目前的間隙是用估計的手指尺寸算的）。
4. 夾持：重做 pregrasp 初始化，以 `configs/physical_tool_controls_*.json` 的控制組（hold／load／torque／weak-grip／low-friction／release）出尺狀刀的新版本並實跑；計算預期切削力與力矩下的夾持力預算，必要時在 Franka Hand 規格內調高指力（寫 ADR），不能用固定連接代替。
5. CUT-01-R、TRIM-01-R、CUT-02 的腳本參考動作在 Native 與 Lab 各跑完整 400 步，用凍結的 oracle 評分；相機 rig v5 不和新刀具碰撞（`wrist_camera_clearance`）；README 放實際截圖與影片。

完成條件：各項 GPU 實跑逐項 PASS／FAIL；CPU 測試全過。

### WP-10D：切削物理與材料的 GPU 驗證

1. 在 GPU、正式解析度（1 mm、0.1 ms）重跑 `tests/test_cutting_law_v2.py` 的情境（用 `blade_probe.drive_blade(device="cuda:0", spacing=.001)`）：對齊推切切開、側推與平壓不切斷任何連結、拉切的下壓阻力 < 純推切的 25%、拔出不增加分離、刀柄只推不切、兩個 digit 切成三段。
2. 材料設定：除了現在的軟材料，再定義至少兩組暫定的食材材料（例如軟乳酪、較硬的蔬果），寫 ADR 並引用文獻量級；對每組材料找出穩定的 `material_dt`（CFL），量 1 mm 下的力曲線，做合理性比較（不宣稱已校正）。
3. 多道切口的評估：v1 oracle 寫死一道切口（`mechanical_components(..., 1, ...)`）。為多刀情境做新的評估（或擴充 Evaluation v2），v1 結果必須逐位相同。
4. 「各種動作」的 GPU 情境表（kinematic 與 live 機器人各一次）：下切、推切、斜切 ±20°／±40°、轉向 30°／60°、刃口傾斜 ±15°、拉切、來回鋸切、削平、側推、刀面平壓、切到一半拔出。每格記錄是否切斷、被切斷的連結數、切面與刀刃平面的距離、阻力。預期：會切的動作要切開；側推、平壓不切斷；拉切和鋸切需要的下壓力明顯較低。結果不符預期就照實記錄為 FAIL 並分析，不能改門檻。

### WP-10E：完整機器人的動作與角度矩陣

1. 可達性與碰撞預檢工具：用 repo 的 Newton 模型（不是 DH 近似），掃描刀刃姿態，含關節限位餘量、手／腕部相機／刀具與桌面、夾具、工件的碰撞，以及沿路徑的關節連續性；輸出可行角度地圖。
2. 動作矩陣（CUT-02 為主，CUT-01-R、TRIM-01-R 抽查；兩個 runtime；physical_grasp），動作與角度同 WP-10D-4，兩種速度。每格記錄：是否完成任務、切面誤差、滑刀量、掉刀、碰撞、IK 拒絕次數、最大反作用力與指力、wall time；部分格子存 RGB-D 影片。
3. 可選：IK 目標用量到的刀具相對手部姿態修正滑刀（opt-in，不影響舊行為）。

完成條件：每格 PASS／FAIL／BLOCKED／NOT_RUN 與原因；失敗格保留原始紀錄；整理「Franka 做不到的角度、動作與原因」表。

### WP-10F：RL 速度

1. kinematic backend 在 GPU（1 mm 與 2 mm）量 s/步，作為 RL 預訓練與篩選用的快速環境。
2. Isaac 的快速模式：不建相機（state 觀察）、可選粗粒子；量 s/步。目標 ≤ 0.05 s/步；做不到照實記錄並指出瓶頸。
3. 一個行程內多個獨立材料實例，或 Isaac Lab `num_envs > 1`；能做就做，做不到寫清楚是「做不到」還是「沒做」。
4. 保真度對照：同一組腳本動作，kinematic、fast、完整設定三者的切斷結果與主要指標差異表。kinematic 與 fast 只用於訓練與篩選，正式評估一律用完整設定。

### WP-10G：policy、RL、agent 控制的端到端驗證與影片（使用者要求）

目的：證明四種控制方式都能**正確輸出動作**、環境**正確執行**，而且**真的完成任務**（CUT-01-R／CUT-02 切斷、TRIM-01-R 削除凸起），並有影片。這次使用者明確允許為了驗證而訓練小模型（只證明管線能用，不宣稱研究結果）。

1. **動作正確性**（每種控制方式都要做）：每步記錄送出的動作、解析後的刀刃目標（`action_report`）、IK 結果、量到的手部與刀具姿態；報告追蹤誤差分佈與最大值，並和每步上限比對。delta 語意用已知序列確認（沿刀刃 +x、沿 world z、繞刃口線轉 10°、來回鋸切）。夾爪指令、`cut_enabled`、`cut_index`、IK 拒絕、clamp、終止原因都要有實跑案例。
2. **腳本 policy**：`WaypointPolicy` 加 `through_cut`／`skim`（下切、鋸切、斜切）經 Gym 介面在兩個 runtime 完成任務；每個任務至少 3 個 seed；用凍結的 oracle 與 Evaluation v2 Research profile 評分。
3. **學習型 policy**：用腳本 policy 收集示範（state 與腕部 RGB-D 都存），訓練一個小的行為複製 policy（state 版必做；影像版至少要能訓練與推論），閉迴路推論；報告動作是否在合法範圍、推論延遲、成功率（成功率低就照實記錄）。
4. **RL**：PPO 先在 kinematic backend 預訓練（state），再到 live runtime 跑幾集；確認 reward 有變化、不崩潰、episode 正常結束。不宣稱學會。
5. **LLM agent**：`agent_api.CuttingAgentAPI` 的工具交給 agent（Claude 或 Codex），只靠觀察（影像與本體感測）和 `describe_task` 決定動作，CUT-01-R、TRIM-01-R 各至少 3 集；記錄每集的工具呼叫序列、是否完成、失敗原因。如果需要跨行程呼叫，就做一個薄的 CLI 或服務，和 Gym 共用後端。
6. **影片（必交）**：上面每種控制方式 × 每個任務至少一集**實際機器人**的影片（Global D455 與兩個腕部 D405 同步 RGB-D，沿用現有錄影流程），畫面上標示控制方式、時間、切削開關與最終評估結果；放在 `docs/media/wp10/`，在 README 與報告中連結。kinematic 的粒子影片只能當補充，不能取代。
7. **RSI 相容**：用 RSI 的 `run(env, ctx)` 控制程式形式跑一集（在本 repo 內模擬那個呼叫方式），寫 RSI 對接說明（`lab/sim/envs/rcs_cut01_{lab,sim}.yaml` 要改什麼；不改 RSI）。

完成條件：一張總表，列出控制方式 × 任務 × runtime 的「動作正確性」與「任務完成」結果（PASS／FAIL／BLOCKED／NOT_RUN 與數字），附所有原始紀錄與影片路徑。

## 4. 交付

- 在 `claude/lucid-mendel-vt7n82` 繼續，每個工作包完成就 commit、push。
- 每個工作包寫 `docs/reports/WP-10x.md`，更新 `docs/PROGRESS.md` 與 README；方向性決定寫 `docs/adr/`。
- 全部完成後開一個 PR 到 main，列出每個工作包的 PASS／FAIL／BLOCKED／NOT_RUN、報告與影片連結；不要自己 merge。

## 5. 已知限制與風險

- kinematic backend 沒有機器人、IK、夾持、滑刀與相機；它上面的成功只代表控制與物理管線正確。
- CPU 驗證用的是 4 mm 粒子、0.5 ms 材料時間步（協定是 1 mm、0.1 ms）；結論要在 GPU 正式解析度重現。
- 法則 2 的參數（`slice_efficiency`、刃口區大小）是暫定值；材料完全沒有校正。
- `TRIM-01-R` 的 25° 前傾是用估計的手指尺寸算的；`CUT-02` 每集約 24 萬個粒子（1 mm），GPU 記憶體與速度要量。
- reward 是學習用的特權訊號，不是成功判定；成功一律看 `evaluate()`。
