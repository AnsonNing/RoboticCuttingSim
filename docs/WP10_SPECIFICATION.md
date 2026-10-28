# WP-10 — 尺狀刀具與可交給 Policy／RL／Agent 的切割場景

Repository: https://github.com/AnsonNing/RoboticCuttingSim
工作分支：`claude/lucid-mendel-vt7n82`（已完成 A 部分，見第 2 節）

## 0. 目標（使用者原話整理）

Franka 夾著刀具，切割物體，完成任務（切斷、切除凸起物等）。

1. 刀具要像尺：長而扁平的直刃，由 Franka 0 兩指**物理夾持**（不是固定連接）。
2. 機器人要能用刀具完成**各種動作**：不同角度切入、下壓、推切、拉切、來回鋸切、沿表面削、斜切、刀面接觸推擠等。不是只支援某一種切法；物理上要合理，做不到的動作要能正確地「沒切開」或「滑刀、掉刀」，而不是照樣切開。
3. 後續會用 policy、RL、RSI agent 輸出動作；狀態觀察與影像觀察都會用到。整個場景要可執行、不崩潰、有安全限制、失敗有回報，速度要足以做 RL。

尺寸與次要目標由實作者判斷，決定寫進 ADR（`docs/adr/0003-ruler-knife-and-policy-interface.md` 已記錄 A 部分的決定）。

共通規則照 `AGENTS.md`：v1 物理、門檻與紀錄不改；新東西版本化並可選；PASS／FAIL／BLOCKED／NOT_RUN 分開；失敗保留；不覆寫輸出；記錄指令、commit、seed、版本、wall time 與 artifact 路徑；不改驅動、不改 RSI。

## 1. 診斷（2026-10-28，A 部分開始前）

證據：讀程式、CPU 測試、CPU 上的 TLMPM 角度探測（2 mm 粒子、刀刃以運動學驅動、不含機器人）、獨立的 Franka DH 運動學掃描（不是 repo 的 Newton 模型、不含碰撞）。

- 運動學：Franka 0（base (−0.46, 0.12, −0.01)）在切割點用 CUT 的手到刀刃安裝，斜切（繞刃口線）±60°、水平轉向 360°、刃口傾斜 ±45° 都有 IK 解，沿下刀路徑每段關節變化 ≤ 0.02 rad；yaw −150° 時離關節限位只剩 0.16 rad。碰撞未檢查。
- TLMPM 角度探測（CUT-01 棒，舊楔形刀頭）：垂直、斜切 20°、轉向 30°、刃口傾斜 15° 都切成兩段，切面跟著刀的方向。但**刀垂直、移動方向偏離刀面 15° 也照樣切斷**（0.53／0.47），反作用力只從 0.25 N 增加到 0.45 N。所有反作用力 < 1 N：材料 E = 50 kPa（很軟，像豆腐），而且是粗粒子。
- 速度：完整一集 8 s / 400 步約 164 s wall；`num_envs = 1`。

問題清單（A 部分已處理的標 ✅）：

| # | 問題 | 位置 | 狀態 |
|---|---|---|---|
| 1 | Native Cartesian IK 不收斂時丟 `RuntimeError`，整集崩潰 | `isaac_sim_runner.py` step | ✅ 改丟 `IKFailure`（`RuntimeError` 子類別）；wrapper 可拒絕並保持 |
| 2 | Lab IK 只檢查位置，不檢查方向 | `isaac_lab_env.py` `_solve_ik` | ✅ 記錄方向誤差；`ik_rotation_tolerance_rad` 可開啟（預設 None = 舊行為） |
| 3 | 動作只有絕對姿態，沒有每步上限與工作空間 | `tasks.validate_action` | ✅ `cartesian_edge_delta` 與 `ActionLimits`（opt-in） |
| 4 | reward 固定 0 | `live_env` | ✅ `reward.CuttingReward`（暫定、特權、opt-in） |
| 5 | 速度、單一環境 | runtime | 未做（WP-10F） |
| 6 | CLI 預設 `legacy_fixed_tool` | CLI | 未做（WP-10B 加 preset） |
| 7 | 材料只碰得到刀刃，刀柄、手指穿過材料 | `tlmpm._rigid_contacts` | 未做（WP-10D） |
| 8 | 切削只算刀刃 local +x 推切；拉切、鋸切不算 | 同上 | 未做（WP-10D） |
| 9 | 分離是幾何標記：刀刃掃過就分離，不檢查能量（側推也切開的原因） | `tlmpm._assign` | 未做（WP-10D） |
| 10 | 一集只能一道切口（外部動作固定 cut digit 0） | `live_material`、兩個 runtime | 未做（WP-10D） |
| 11 | 力量級偏小、材料極軟 | `material_config` | 未做（WP-10D 合理性檢查與材料預設） |
| 12 | `tlmpm.py` 的 kernel 參數叫 `new`（C++ 保留字），Warp CPU 編譯失敗 | `tlmpm.py` | ✅ 在 `tlmpm_v2.py` 改名；v1 不動 |
| 13 | IK 對準名目安裝，不修正刀在手中滑動 | runtime | 未做（WP-10E，可選） |
| 14 | 刀具相容性只接受舊楔形 | `tool_compatibility` | ✅ `tool_profile.require_profile_consistency`（以 profile 一致性取代尺寸相等） |

更正：physical_grasp 模式的觀察本來就不含刀具真實姿態（只給手部 FK 與手指編碼器），沒有特權資訊外洩問題。

## 2. A 部分已完成（分支 `claude/lucid-mendel-vt7n82`，CPU 驗證，GPU NOT_RUN）

- `src/robotic_cutting_sim/control.py`：`cartesian_edge_delta`（[dx,dy,dz,rx,ry,rz]，edge 或 world 座標）、`ActionLimits`（預設每步 2 mm、2°，工作空間框以工件座標表示）、`IKFailure`、`normalized_delta`、`nominal_edge_pose`。
- `live_env.CuttingTaskEnv` 新參數（預設全部保持舊行為）：`action_limits`、`delta_frame`、`ik_failure="raise"|"reject"`、`max_consecutive_ik_failures`、`reward`；觀察 profile 多 `state`（無影像）；`info` 多 `ik_status`、`ik_info`、`action_report`、`consecutive_ik_failures`。`open_cutting_env` 會把這些參數轉給 wrapper。
- `reward.py`：`CuttingReward(task)`，`cutting_reward_v0`：目標帶分離比例的增量、目標外切割、TRIM 的過切（刀刃低於目標面）、刀面法向的側移、IK 拒絕、clamp、終止、時間；`info["reward_terms"]` 有每一項。只用來學習，成功判定仍是 `evaluate()`。
- `gym_env.py`：`CuttingGymEnv`（gymnasium 可選），動作 Box[−1,1]^6，觀察 `state` 向量或 Dict（state＋各腕部相機 RGB、深度）；通過 `gymnasium.utils.env_checker`（fake backend）。
- `tool_profile.py`：`ToolProfile` 與 `RULER_KNIFE_V1`（刃長 120 mm、刀身深 40 mm、厚 1.5 mm、雙斜刃 25°；刀柄沿刀背，手座標半尺寸 (50, 11, 20) mm，刃口中心在夾持軸上；總質量約 150 g）。同一個 profile 產生 ToolSpec（碰撞體、質量慣量）、TLMPM 多邊形與 `hand_from_edge`。舊楔形可表示成 `legacy_profile(task)`。
- `tlmpm_v2.py`：v1 的副本，只改三件事（見檔頭）；預設三角形與改名後的 v1 在 CPU 上**逐 bit 相同（含實際切割）**。`LiveCuttingMaterial(..., tool_profile=...)` 選用 v2；CPU 裝置不用 CUDA graph。
- 測試：`tests/test_policy_interface.py`（9）、`tests/test_tool_profile.py`（7，含尺狀刀在 CPU 上垂直與 25° 斜切切斷一根棒）。全套 CPU：428 PASS；2 個 `test_bootstrap` 失敗只因該機器沒有 Python 3.12。

尚未做：任何 GPU 實跑；runtime（Native、Lab、USD）還不會建立尺狀刀具；任務定義還沒有使用 profile。

## 3. 剩下的工作包（依序；完成條件就是測試標準）

### WP-10B：GPU 驗證 A 部分

1. 兩個 runtime 各跑「隨機動作模糊測試」≥ 1000 步，用 `action_mode="cartesian_edge_delta"`、`ik_failure="reject"`、`ActionLimits()`、Lab 加 `ik_rotation_tolerance_rad=0.01`；動作含刻意不可達的目標、超大位移、方向翻轉。不能崩潰，每次拒絕都有 `ik_status`。
2. 舊腳本基準（`action=None`、預設參數）在兩個 runtime 重現原紀錄的切斷結果與 strict 分數（證明預設行為沒變）。
3. 加 CLI preset（例如 `--preset policy`：physical_grasp、pregrasped、grip 0.0112、finite_pad_patch_v3、delta 動作、reject、Lab 方向檢查），不改舊旗標的預設值。
4. 用 Gym 介面在兩個 runtime 各跑一集（state 與 vision_proprio 各一）。

### WP-10C：尺狀刀具接進 runtime

1. 任務層：在不改 `cutting_tasks_v1.json` 既有任務雜湊與 `asdict` 收據的前提下，讓任務可以選刀具 profile（例如新的 protocol 檔或 task 變體）。CUT-01、TRIM-01 各一個尺狀刀變體；新增 CUT-02（較寬工件，例如 160 × 60 × 25 mm）。`TaskSpec.__post_init__` 目前只接受 CUT-01／TRIM-01，要擴充。
2. Native 與 Lab 的物理夾刀建立改用 `make_profile_tool_spec`；USD／Newton 碰撞體、視覺 mesh 由 profile 產生；`require_matching_blade` 對 profile 刀具改用 `require_profile_consistency`。材料一律用 `LiveCuttingMaterial(..., tool_profile=...)`。
3. 夾持：重做 pregrasp 初始化；沿用 `configs/physical_tool_controls_*.json` 的控制組（hold／load／torque／weak-grip／low-friction／release）針對尺狀刀出新版本設定並實跑。計算預期切削力與力矩下的夾持力預算；必要時在 Franka Hand 規格內調高指力（寫 ADR），不能用固定連接代替。
4. 腳本參考動作：CUT（垂直下切）、TRIM（尺狀刀需要前傾約 15–20° 讓刀柄離開已削平的表面，或改用其他合理姿態；照實驗結果決定），在 Native 與 Lab 跑完整 400 步。
5. 相機：rig v5 不和新刀具碰撞（`wrist_camera_clearance`），README 放實際截圖與影片。

完成條件：上述各項 GPU 實跑結果逐項 PASS／FAIL；CPU 測試全過。

### WP-10D：通用刀具動作的切削物理（`cutting_law_v2`，在 `tlmpm_v2.py`；v1 保留為預設，直到 v2 驗證）

目標：任何刀具相對材料的運動都有物理上說得通的結果。

1. **分離以能量為門檻**：粒子只有在刃口附近累積的切削功達到門檻時才標記分離（目前 `_assign` 只看幾何）；刀面側推只產生接觸阻力與變形，不產生分離。
2. **刃口切削同時算法向與切向**：刃口相對材料的速度分成「垂直刃口、在刀面內」（推切／下切）與「沿刃口」（拉切／鋸切）。兩者都能做功；沿刃口的分量依 slice/push 比例降低所需的法向力（參考 Atkins 等人的切割力學；參數暫定、寫 ADR）。垂直刀面的分量（側推）不切削。
3. **非刃口部件**：刀背、刀柄、手指作為鈍剛體障礙物加入材料接觸（只推擠、不切削），不再穿過材料。
4. **多道切口**：外部動作可指定 `cut_index`，或每次刃口離開材料再重新切入時自動遞增；至少支援 TLMPM 現有上限 2 道；評估提高上限的記憶體與速度代價。
5. **材料**：現在 E = 50 kPa；為切斷與削除任務選一組合理的暫定材料（例如軟食材與較硬的蔬果各一），量測 1 mm 解析度下的力曲線，與文獻量級做合理性比較（不宣稱已校正）。
6. CPU 測試（`tlmpm_v2` 已可在 CPU 跑）與 GPU 1 mm 探測都要做。

完成條件（刀具以運動學驅動與完整機器人各做一次；尺狀刀）：
- 對齊的推切、斜切（±20°、±40°）、轉向（30°、60°）、刃口傾斜（±15°）：切斷，切面在刀刃平面 2 mm 內。
- 純拉切與來回鋸切（下壓力小）：能切開，所需下壓力低於純推切。
- 側推 15°、30°：分離明顯不完整或延遲、阻力顯著增加；若仍照樣切開，記為 FAIL 並分析原因。
- 刀柄、刀背不再穿過材料（報告最大穿透深度）。
- 同一集兩道切口：產生三段，質量正確。
- `cutting_law_v1` 在舊設定下重現所有既有紀錄。

### WP-10E：完整機器人的動作與角度矩陣

1. 可達性與碰撞預檢工具：用 repo 的 Newton 模型（不是 DH 近似），掃描刀刃姿態，含關節限位餘量、手／腕部相機／刀具 vs 桌面、夾具、工件的碰撞、沿路徑的關節連續性；輸出可行角度地圖。
2. 動作矩陣（CUT-02 為主，CUT-01、TRIM-01 抽查；兩個 runtime；尺狀刀；physical_grasp）：下切、推切、拉切、來回鋸切、沿表面削、斜切 0／±15／±30／±45°、轉向 0／30／60／90°、刃口傾斜 0／±15／±30°（取正交子集）、兩種速度。每格記錄：是否完成任務、切面誤差、滑刀量、掉刀、碰撞、IK 拒絕次數、最大反作用力與指力、wall time；部分格子存 RGB-D 影片。
3. 可選：IK 目標用量到的刀具相對手部姿態修正滑刀（opt-in，不影響舊行為）。

完成條件：每格 PASS／FAIL／BLOCKED／NOT_RUN 與原因；失敗格保留原始紀錄；整理「Franka 做不到的角度與原因」表。

### WP-10F：RL 速度

1. `fast` 設定：不建相機（state 觀察）、可選 2 mm 粒子；量測 s/步。目標 ≤ 0.05 s/步；做不到照實記錄並指出瓶頸。
2. 一個行程內多個獨立材料實例，或 Isaac Lab `num_envs > 1`；能做就做，做不到寫清楚是「做不到」還是「沒做」。
3. 保真度對照：同一組腳本動作（垂直、斜切 20°、轉向 30°），fast 與完整設定的結果差異表。fast 只用於訓練與篩選。

### WP-10G：policy、RL、agent 控制的端到端驗證（使用者要求）

目的：證明三種控制方式都能**正確輸出動作**、環境**正確執行**，而且能**真的完成任務**（CUT 切斷、TRIM 削除凸起），不只是不崩潰。本工作包是使用者明確要求，允許為驗證而訓練小模型（只用來證明管線，不宣稱研究結果）。

1. **動作正確性（每種控制方式都要做）**
   - 指令與執行一致：每步記錄送出的動作、解析後的刀刃目標（`action_report`）、IK 結果、實際量到的手部與刀具姿態；報告追蹤誤差（位置、方向）的分佈與最大值，並和每步上限比對。
   - delta 語意：在 GPU 上用已知序列（沿刀刃 +x、沿 world z、繞刃口線轉 10°、來回鋸切）確認實際運動方向與大小正確（容差寫清楚）。
   - 夾爪指令（hold／open／release）、`cut_enabled` 開關、IK 拒絕、clamp、終止原因都有實跑案例且回報正確。
2. **腳本 policy（oracle 參考）**：經 Gym 介面完成 CUT-01、CUT-02、TRIM-01（尺狀刀）；`evaluate()`（v1 與 Evaluation v2 Research profile）判定完成；每個任務至少 3 個 seed，報告成功率。這是後面兩項的上限參考。
3. **學習型 policy 推論管線**：用腳本 policy 收集示範（state 與腕部 RGB-D 都存），訓練一個小的行為複製 policy（state 版必做；影像版至少確認能訓練與推論），在環境裡閉迴路推論；報告動作輸出是否在合法範圍、推論延遲、任務成功率（成功率低照實記錄，重點是管線正確）。
4. **RL**：PPO（fast 模式，約 2 萬步）確認 reward 有變化、不崩潰、episode 正常結束；再用完整設定跑訓練後的 policy 幾集看行為。不宣稱學會。
5. **LLM agent**：用第 6 項的函式或 CLI，讓 agent（Claude 或 Codex）各做 CUT-01、TRIM-01 至少 3 集，只靠觀察（影像與本體感測）決定動作；記錄每集的指令序列、是否完成、失敗原因。
6. 給 agent 的簡單函式或 CLI（reset、observe、move_edge_by、cut_along、set_gripper、status、evaluate），和 Gym 共用後端；寫使用說明。
7. RSI 相容：用 RSI 的 `run(env, ctx)` 控制程式形式跑一集（在本 repo 內模擬那個呼叫方式），並寫 RSI 對接說明（`lab/sim/envs/rcs_cut01_{lab,sim}.yaml` 要改什麼；不改 RSI）。

完成條件：一張總表，列每種控制方式 × 任務 × runtime 的「動作正確性」與「任務完成」結果（PASS／FAIL／BLOCKED／NOT_RUN 與數字）；所有原始紀錄與影片路徑。

## 4. 交付

- 在 `claude/lucid-mendel-vt7n82` 繼續（或從它開新分支），每個工作包完成就 commit、push。
- 每個工作包寫 `docs/reports/WP-10x.md`，更新 `docs/PROGRESS.md` 與 README；方向性決定寫 `docs/adr/`。
- 全部完成後開一個 PR 到 main，列出每個工作包的 PASS／FAIL／BLOCKED／NOT_RUN 與報告連結；不要自己 merge。
- GPU：先 `nvidia-smi` 看誰在用，不要碰別人的行程（這台伺服器 GPU0 上可能有別人的 vla-sim）；如果執行環境看不到 GPU，就把需要的確切指令列出來並記為 BLOCKED，不要改用別的方法冒充。
