# Progress — 2026-10-29 Asia/Seoul

## 進行中：WP-11（2026-10-30，分支 `claude/wp11-cutting-refinement`）

- 計畫：[WP-11](WP11_PLAN.md)。刀具：[ADR 0013 `ruler_knife_v2`](adr/0013-ruler-knife-v2-and-trim-orientation-metrics.md)。v2 改成像參考圖那樣夾住刀身一端的細握柄，刀身沿用 v1 的 120×40×1.5 mm。另外定義了削切姿態指標 τ／ψ／ρ／μ。
- Baseline：main `3e26d6b`；本機 780 passed／5 xfail；CI 5 failed（4 個 hash、1 個 matplotlib）。`final_v4` 不動。
- 補充需求（Project Goal & Generalization）已整合進 [WP-11 計畫](WP11_PLAN.md)：Phase S（場景／目標分離）、Phase M（[ADR 0014](adr/0014-scalable-cut-history-sparse-logical-fields.md)：`tlmpm_v5` 打包 digit 與稀疏 slot）、Tests A–F。
- 已 commit：Phase A 第一版（`97d56d3`：rollout pin 改為與環境無關，測試移除 matplotlib 依賴；CI 結果待確認）；Phase B 的 v2 profile（`a8e4d25`：`ruler_knife_v2_y`／`_x`，v1 pinned）與 Native grasp 工具（`c6eea75`）。
- Phase B 完成（`2d13d30`、`712e106`）：`ruler_knife_v2_y`／`_x`，握柄縮為 60 mm（實測手掌 collider 在手 z = 0.070 m，70 mm 會頂到）。
  - Native 7 項夾持控制兩種排列都 PASS_FROZEN_RIGID_CONTROLS（門檻同 v5；hold 時兩指各 18 N，滑移 ≤ 0.001 mm；穿透 ≤ 12 µm）。
  - −10 N stress 下 v2 比 v1 鬆（A 268 mrad、B 35 mrad，v1 33 mrad；此項只報告）。
  - [證據](evidence/wp11/phaseB_knife_v2_001.json)、[圖](media/wp11/knife_v2_y.png)。與參考圖相比，握柄偏大、刀身不是亮金屬色；待 C2 結果後一併定案。
- Phase S 完成（`4aef698`，另有部分修改被併入 `98d5c3f`）：`scene_config.py`（`rcs_scene_v1`、`compile_task` → `ConfiguredTaskSpec`）、`goals.py`（sever／divide／remove_protrusions／trim_plane／move_piece／sequence，三層評估）、`goal_planner.py`（只讀設定，τ/ψ/ρ/μ）。
  - 設定可完全重現 CUT-01-R／TRIM-01-R。
  - 4 mm kinematic 下三凸起逐一移除（tlmpm_v5，K=3）全部 1.0。
  - 限制：Isaac runtime 尚不接受 compiled task；`apply_scenario` 仍只接受 identity 方向。
  - 流程注意：平行代理共用 git index，修改曾被併進別人的 commit，以及一次誤清空後已還原。之後一律用 `git commit -- <paths>`。
- C2 預檢完成（`98d5c3f`，[證據](evidence/wp11/phaseC2_preflight_001.json)）。
  - ρ < 12.5° 時雙斜面最低點低於刃口，0°／5° 會刮進底材 0.55／0.25 mm；10° 剛好過。
  - 全任務共用配置：`v2_y`、刀尖 +y、工件抬高 +100 mm、TRIM ρ = 10°。決定已記入 ADR 0013。
- Phase M 完成（`be1d859`、`b2c1d09`，[證據](evidence/wp11/phaseM_tlmpm_v5_001.json)）。
  - `tlmpm_v5` 與 v4 law 8 等價：CPU 逐位相同，GPU M1a／M8b 在非確定性範圍內。
  - 成本與 K 無關：K=1–8 都是 1.30–1.34 ms/substep、38 MB。
  - 夾住的長條：3／5 刀得 4／6 塊。三凸起逐刀移除 0.98／0.97／0.98。
  - 待處理：自由長條每刀滑約 1 mm，宣告切面超界（與 M8 同現象）；兩次削切重疊處約 49 顆底材頂層粒子連起 chip 1、2；`validate_action` 只收 `cut_index` 0/1（Phase G 要改）；reward／v1 oracle 仍是 ternary；S=8 已用到 7 個；v5 的 M3–M6、M9 未重跑。
- Phase A（`97d56d3`、`cce395e`、`ee2a285`，[報告](reports/WP11_PHASE_A_CI.md)）。
  - Hash 不一致的根因是 OpenBLAS 依 CPU 選 kernel（SkylakeX 與 Haswell），只差 1 ulp，不是行為改變。
  - 改用參考軌跡＋推導出的容差 N·16·eps；matplotlib 改為 NumPy 判斷；CI 改完整 checkout（原本 26 個 skip 是歷史 law 測試沒跑）。
  - `test_tool_profile_v2.py` 的主軸改用推導出的 ulp 容差（`f7dfef5`）。
  - **GitHub Actions run 40934578313 PASS**：888 passed、1 skipped（需要 git-ignore 掉的 live 收據）、5 xfailed、0 failed，在 Haswell runner 上。
- C4 完成（`aa0dfdc`、`20101c1`、`78c0641`，[證據](evidence/wp11/phaseC4_trim_material_001.json)）。
  - 雙斜面 ρ = 5–35° 都分離（93–98%，0 橋接）；0° FAIL（下斜面壓陷）。
  - 單斜面只有 0° 過，因為 law 8 閘門量刀身平面而不是推 chip 的面角，這是 Phase F 的發現。
  - 負對照：平壓與鈍握柄兩項的平均力窗口含空氣段，依原門檻 FAIL，但都 0 標籤、1 component；Phase F 會先宣告接觸窗口再重跑。
  - **定案：v2_y 雙斜面、ρ = 10°**，已記入 ADR 0013。
- Phase D 完成（`5cf946f`，[證據](evidence/wp11/phaseD_home_001.json)）。
  - 新增 `wp11_workcell_v1`（+100 mm）；Isaac 可接受 compiled task；HOME reset；planner 控制器；實體 Return-to-Home 與狀態機（`episode_lifecycle.py`）。
  - Native smoke PASS：HOME 預夾、REORIENT 到 ρ = 9.99°、return home 關節誤差 0.39 mrad、刀具位姿 0.031 mm、標籤變化 0；`home_outcome` 與任務結果分開。
- **C5 live 近水平 TRIM PASS**（`5aa0bee`，[證據](evidence/wp11/phaseC5_live_trim_001.json)，[影片](media/wp11/trim_rho10/demo.mp4)）。
  - Native、`ruler_knife_v2_y`、ρ = 10°：凸起去除 97.9%，底材 0.99996，chip 持續獨立，夾持滑移 0.75 µm，Return-to-Home PASS。
  - 待改進：PRE-CUT 量測回饋振盪 ±0.2 mm，刀身最低點一度進入未切頂面 0.16 mm，間隙稽核 FAIL；chip 留在長條頂面，畫面不夠清楚。由 C5b 處理（`path_v3` 控制器＋把 chip 推離）。
- **F1 夾持壓力測試完成：v8 suite verdict FAIL（一項負對照設計錯誤），v9 修正版負對照 PASS**（`0c39b23`、`9a4cb5f`、`78e20ab`，[報告](reports/WP11_PHASE_F1_GRASP.md)，[證據](evidence/wp11/phaseF1_grasp_001.json)，Native，seed 42）。
  - 31 項中：30 s HOME／TRIM 保持、2× REORIENT、0.1–0.5 N 外力、0.01–0.05 N·m 力矩、C4 切削反力重播（含 10×）、兩項釋放全部 PASS（滑移 ≤ 4.8 µm、≤ 0.14 mrad）。
  - 唯一 FAIL：負對照 `low_friction_trim_advance_0p1n`。0.1 N −advance 的力矩（−0.009 N·m）幾乎抵銷重力在 ρ = 10° 的 +0.0083 N·m，是對照設計錯誤；門檻不改，v8 紀錄保留為 FAIL。
  - 修正版負對照 protocol v9（`136b0af` 先宣告，`3478a11` 證據，[證據](evidence/wp11/phaseF1_negctrl_v9_001.json)，cuda:1）：三項全部 PASS 且都符合預先寫下的預測。TRIM +advance 0.1 N 0.28 s 掉刀；TRIM −normal 0.1 N 轉 447 mrad；HOME −advance 0.1 N 0.16 s 掉刀。對應的 μ = 0.6 參考列都保持（≤ 0.4 µm）。
  - 弱方向是繞手指閉合軸（= 刀身法向）的力矩：0.05 N·m 保持，0.18 N·m 轉 12.5°，0.45 N·m 掉刀。切削反力在此軸只有約 0.0002 N·m。
- **F2/F3 law 8 邊界驗證完成**（`d69f1e5`，[報告](reports/WP11_PHASE_F_LAW8.md)，[證據](evidence/wp11/phaseF_material_001.json)，protocol `phaseF_v1` 先宣告；kinematic 1 mm，cuda:0，seed 42）：17 PASS、1 FAIL、6 characterization、2 ERROR。
  - 15° 閘門：14°／15° 完全不 plow，16° 每步都 plow；三者都分離凸起（93–94%）。
  - 搖擺、純旋轉、鋸切、純拉鋸、撤刀、有限刃長、平壓、側推、無接觸、再接觸、`tlmpm_v5` 三刀歷史全部 PASS。
  - **FAIL：鈍握柄負對照**（0 標籤、1 塊，但接觸平均力 0.077 N < 0.1 N）。握柄只碰到長條自由端角落，長條在摩擦支座上轉動約 1.4° 而沒有被壓縮。保留 FAIL。
  - 修正版 `phaseF_v2`（`dc64f49` 先宣告預測，`4c7aa6e` 證據，[證據](evidence/wp11/phaseF_material_v2_001.json)）**PASS**：握柄壓在右支座正上方的頂面，接觸平均力 0.855 N（預測 0.35–1.6 N），0 標籤、1 塊，刀身接觸 0 步，壓縮 1.65 mm 是剛體位移的 9.4 倍。側推未受約束長條的情形仍未驗證。
  - F3 解析度：正式解析度重現 PASS；2 mm 與 1.5 mm 下 TRIM 去除率掉到 0.82／0.58（底層粒子與刃口太近），接觸力在 dt 減半時約變 2.6 倍。**去除率與接觸力對解析度未收斂**，列為已知限制。宣告的 medium（0.15 ms）因不能整除 20 ms 控制步而 ERROR；另跑的 0.02/133 s 標為非 protocol 補充。
  - F2.11 只報告的 goal 檢查：第 2、3 刀離宣告切面 4.0／4.5 mm（> 2 mm），與先前「長條每刀滑動」現象相同。
- **Phase E 泛化第一輪（kinematic）完成**（`4b2fe08` 先宣告，`f83181d` 報告，[報告](reports/WP11_PHASE_E_ROUND1.md)，[證據](evidence/wp11/phaseE_round1_001.json)）：41 個 seed 全部預檢可行，**32／41 kinematic TASK_SUCCESS（78%）**，9 個 CUTTING_FAILURE 全部列出，沒有排除。
  - CUT 6/6、TRIM 6/6、Test A 7/8、Test B 5/5（3／4 刀得 4／5 塊）、Test C 5/5（每個凸起 0.909–1.000）、Test D 3/6、SLICE-PUSH 0/5。
  - SLICE-PUSH 五個都只差第二刀位置：第一刀後長條滑 1.2 mm，第二刀短 1.00 mm，凍結統計 4.0 mm > 2 mm。live Demo 3 用手腕深度重新定位（ADR 0009）處理這個問題，kinematic 階段沒有相機。
  - Test A s2（−20° 斜切）位置超界 0.02 mm。Test D s1701 是 `tlmpm_v5` slot 溢位（S = 8 不夠）；s1702／s1704 是目標設計錯誤（sever 之後「底材單一塊」不可能成立），依預先宣告規則仍記為 CUTTING_FAILURE。
  - 預檢沒有遇到不可行樣本，範圍要放寬才量得到可行性邊界（第二輪）。這輪是 kinematic 理想追蹤，不是 live；GRASP_FAILURE 只在 live 才可能出現。
- **Phase G policy-ready API 完成**（`a2ee588`…`184b3b7`，[API 文件](POLICY_READY_API.md)，[證據](evidence/wp11/phaseG_policy_ready_001.json)）。
  - `open_cutting_env(..., preset="cutting_research_v1")` → `PolicyReadyEnv`：實體夾持、預夾、HOME reset；動作逐步回報 ACCEPTED／CLAMPED／REJECTED_<原因>，被拒絕的動作不會執行；`cut_index` 可到 `max_cuts`（最多 15）。
  - 一般觀測（腕部 D405×2、D455 RGB-D、關節、末端位姿、夾爪、目標）與 `privileged_state`（粒子、cut code、接觸力、真實刀具位姿）分開。
  - 資料收集器 `scripts/collect_dataset.py`／`read_dataset.py`（成功與失敗 episode 都存）。
  - **Test E PASS**：kinematic 1 mm 的 TRIM 與 sever、Native TRIM，scripted planner 與只由場景數字產生的手寫動作程式都成功（Native 手寫版 98.47%，不用 privileged 輸入）。
  - **Test F**：kinematic **POLICY_READY**（7 episode，replay 完全重現）；**Native NOT POLICY_READY**：只有記憶體不合格（每次 reset host RSS +82 MiB，不收斂；GPU 記憶體持平）。Phase H 處理。
  - 可靠性規則 v1a 是在 smoke 之後、正式 Native run 之前加的（夾住的刀 reset 後 PhysX 速度雜訊 3.6 mm/s），v1 不改，兩者都計分。
  - 注意：Native run 的 manifest 是 git_dirty（當時工作樹有 C5b 未 commit 的修改），Phase H 會從乾淨 worktree 重跑。
- **Phase E 第一輪 live 階段**（`1ecfe5e` 先宣告，`fef0e1e`／`fc8c972` 證據與報告，[證據](evidence/wp11/phaseE_round1_live_001.json)，Native，實體夾持 v2 刀）：9 個推薦 seed 中 **6 個 live TASK_SUCCESS**，2 個 CUTTING_FAILURE，1 個 SCENE_INFEASIBLE；GRASP_FAILURE、IK_FAILURE 都是 0。
  - 成功：TRIM s1203（0.994）、s1201（0.962）、CUT s1101、20° 斜切 s1102、Test A s6（−x 方向削切 0.967）、**Test B s1502（四刀五塊）**。全部 Return-to-Home PASS，最大滑移 1.3 µm。
  - 失敗與 kinematic 一致：SLICE-PUSH s1301 第二刀位置 4.0 mm（長條滑 1.16 mm）；Test A s2 斜切位置 1.023 mm（> 1.00 mm）。手腕深度重新定位目前綁在 SLICE-PUSH-01，configured task 還不能用。
  - Test C s1602：live_001 預檢用單一凸包包住三個凸起，誤判碰撞而拒跑（SCENE_INFEASIBLE）；補充的 live_002 在第三次削切時因顯示用網格（`ParticleSurface`）遇到 11 顆粒子的碎片而中止（NOT_RUN）。兩筆紀錄都保留，修好後另立 live_003。
  - 只報告的路徑檢查：三次 10° 削切在削切前穩定時刀身低於未切頂面 0.19–0.24 mm（切削關閉），與 C5 相同的未解問題。
  - 影片：[Test A s6](media/wp11/test_a_s6/demo.gif)、[Test B s1502](media/wp11/test_b_s1502/demo.gif)。
- 磁碟：/mnt/data 剩 17 GB，已把 10 個 WP-11 之前的舊 artifact（約 11 GB）搬到根磁碟 `/home/ningan/RoboticCuttingSim_artifacts_rootdisk/`（rsync 後比對檔名與大小，再改 symlink；紀錄在該目錄的 `MOVE_LOG.txt`），現在 /mnt/data 剩 27 GB。
- **C5b 完成：近水平 TRIM＋把 chip 推離，live PASS**（`03dfd1d` 程式，`0b2d55b` 證據與影片，[證據](evidence/wp11/phaseC5b_live_trim_clear_001.json)，[影片](media/wp11/trim_rho10_clear/demo.mp4)）。
  - 振盪原因：path_v2 每步把（目標 − 量測）加到上一個指令，等於在有延遲的系統上做單位增益積分器；擬合出的 z 軸極點 0.92、y 軸 1.07（不穩定）。
  - `path_v3`（opt-in）：指令 = 規劃參考 + 緩慢調整的 bias（增益 0.1、20 µm 死區），參考軌跡不讀量測值；停止前幾何減速，切入點上方 1 mm 先停 0.5 s。
  - 新目標 `trim_bump_rho10_clear`：削除後用刀身在頂面上方 2 mm 近水平掃過，把 chip 推離長條（disarmed）。
  - Live attempt 002 PASS：凸起去除 97.89%、底材 0.99996、chip 被推 109 mm 掉到桌上、之後一直是 2 個 component；PRE-CUT 刀身最低點 +0.076 mm（C5 是 −0.167 mm），間隙稽核與路徑檢查都 PASS；夾持滑移 0.78 µm；Return-to-Home PASS。
  - Attempt 001 因另一個使用者的 24 GB 訓練工作佔滿 GPU 1 而 CUDA OOM（切削前），保留為 FAIL_EXECUTION。
  - 未解：SKIM 開始與結束時側向 y 偏差 −0.27／+0.14 mm（切到凸起期間 ≤ 0.024 mm）；SKIM 時刀身離頂面只有 0.013 mm。
- **Phase H 完成**（`a5b05b8`…`9e84133`，[報告](reports/WP11_PHASE_H_PROFILING.md)，[證據](evidence/wp11/phaseH_profiling_001.json)）。
  - 記憶體：每次 reset 的 host RSS 增長幾乎都來自 Kit 的 `open_stage`（Python heap 只 +0.04 MiB）。新增 opt-in `stage_reset="reload_in_place"`（保留 stage，重載 root layer），reset 結果與原本逐位相同；4 episode Test F 的 host 增長 140 → 56.5 MiB/episode（門檻 64，PASS）。
  - Native 仍 **NOT POLICY_READY**：乾淨 worktree 重跑時 GPU 記憶體 52 MiB/episode（門檻 32），只有 3 個點且先降後升，可能是 allocator 雜訊；門檻不改，需要 ≥ 8 episode 的宣告 run 才能定論。`reload_in_place` 在相機關閉時第二次 reset 會 segfault，因此要求相機開啟。
  - Profiling：kinematic 1 mm 每步 0.49–0.64 s，TLMPM 佔 99%；其中接觸 kernel 佔 GPU 時間 80%。Native TRIM 每步 0.67 s：TLMPM 58%、相機 12.6%、表面網格 11.8%。
  - 最佳化：opt-in `surface_update="on_capture"`（顯示網格只在拍照與評估前重建），表面時間 49 → 8.8 ms/步，物理逐位相同。
  - 設定檔 `runtime_profiles_v1`：`full_physics_v1`（唯一可用於驗收）與 `research_fast_v1`（每個 episode 標 `acceptance_valid: false`；kinematic TRIM 快 8×、SEVER 快 9.6× 但切面位置 FAIL；Native 快 2.3×）。
- **Test C live PASS**（live_003，`002c95e` 先宣告，`d3f94e9`／`2b4f0a7` 修正，`ac4a3e1` 證據，[證據](evidence/wp11/phaseE_round1_live_003.json)，[影片](media/wp11/test_c_s1602/demo.gif)）：三個凸起逐一削除 0.962／0.997／0.972，底材單一 component，三個 chip 各自在自己的削切中分離並離開長條；夾持滑移 ≤ 0.83 µm；Return-to-Home PASS。
  - 修正：顯示用網格遇到無法成網格的小碎片時跳過並記錄（物理、標籤、評估不變；可成網格時輸出逐位相同）；runner 預設預檢在 ≥ 2 個凸起時改用分段幾何（0／1 個凸起不變）。
  - live_001（SCENE_INFEASIBLE）與 live_002（NOT_RUN）紀錄不變。
  - 未解：PRE-CUT 時刀身低於頂面 0.23 mm（這次仍用 C5 控制器，`path_v3` 可解）；表面品質指標顯示頂面缺 0.66%、最大凹陷 22 mm（未計分，未調查）。
- **SLICE-PUSH 手腕深度重新定位（configured task）**（[ADR 0016](adr/0016-configured-perception-relocation.md)，`0e96f07` 程式，`c29f02a` 先宣告 live_004，`fa0eabb` 證據，[影片](media/wp11/slice_push_relocate/demo.gif)）。
  - opt-in `planner_options.perception = "wrist_depth_relocate_v1"`：每刀之間在場景設定決定的固定位姿用腕部 D405 深度量 +x 端面位移，平移下一刀；不用 look-at、不最佳化觀測位姿、不讀粒子／標籤／刀具真值。不要求時，計畫與 task id 完全不變。
  - Live s1301 **TASK_SUCCESS**：量到 +1.158 mm（oracle +1.149 mm），第二刀落在 +24.00 mm（live_001 是 +23.00 mm），凍結統計 1.5 mm ≤ 2 mm，三塊、推開 16.7 mm，Return-to-Home PASS。round 1 的 kinematic 與 live_001 失敗紀錄不變。
  - s1302／s1303 因 /mnt/data 降到 15 GB 下限而沒有啟動（NOT_RUN）；又搬了 22 個舊 artifact（約 12 GB）到根磁碟，現在剩 27 GB，兩個 seed 第一次啟動中。
- CI：`beb92b5` 修正 GitHub Actions 失敗（Test C 預檢測試在沒有 newton／mujoco／Franka 資產時改標 NOT_RUN）。
- 背景工作（rcs-impl，輸出在 `artifacts/wp11/`，symlink 到 `/mnt/data`）：
  - C4：1 mm 近水平削切材料測試（`artifacts/wp11/phaseC4/`）。
- 已知風險（C4）：雙斜面在 ρ < 12.5° 時下斜面低於刃口；law 8 在 < 15° 等同 law 4。
- 下一步：D 完成後跑 C5 live 近水平 TRIM（ρ = 10°）→ F1 Native 長時間夾持壓力測試 → E 泛化（Tests A–D）→ G policy API／資料收集器（Tests E/F）→ H profiling → I。

## 完成：BladeMaster 風格三個 Demo，最終組合 law 8（2026-10-29）

- **Phase 1 閘門**（計畫 §2 的整個矩陣含負對照）已由 [law 8](adr/0012-cutting-law-8-plow-gated-lead.md) 滿足：依計算式準則 M1–M9 全部 PASS（[matrix_law8_001](evidence/phase1/matrix_law8_001.json) 20:54；乾淨工作樹確認重跑 [matrix_law8_002](evidence/phase1/matrix_law8_002.json) 21:19，在 live 進行期間完成，verdict 相同）。rcs-review 判定 PASS（附條件，已處理）。
  - Law 4–7 各有一列以上 FAIL，全部保留（[0010](adr/0010-cutting-law-6-deferred-lead-commit.md)、[0011](adr/0011-cutting-law-7-eq8-coupled-flip.md) 及其證據）。
  - M8a 是預先宣告的診斷列，FAIL；M9c/d（Lab）只報告。
- **閘門通過後的 live 最終組合**（`--demo-set final_v4`，`live_20261029_law8_final`，乾淨 commit；在 matrix_law8_001 之後 13 s 開始）：
  - Demo 1a **PASS**（凍結 strict 也 PASS）；
  - Demo 1b **PASS**（swept-plane；凍結 strict 的位置與窗口兩項 FAIL）；
  - Demo 2 **PASS**（98.0%；精度 5/8 FAIL）；
  - Demo 3 **PASS**（含凍結 v3 slice-push；推開 19.7 mm；手腕深度量得位移 1.20 mm）。
  - 所有嘗試與實際時間順序見 [demos_live_004](evidence/phase3/demos_live_004.json)，[報告](reports/BLADEMASTER_DEMOS.md)。
- 完整 CPU regression（law 8 之後）：780 passed、5 xfailed（strict）、0 failed，815 s（`artifacts/wp10_claude/logs/cpu_full_final_002.log`，HEAD `b34a71d`／`cc881f7`，之後只有文件與 runner 說明文字的變更）。
- **Law 5 的 live 組合**：在閘門滿足前跑的，保留為先前證據（下節），不算驗收 run。

## 先前結果：law 5 組合（閘門滿足前，2026-10-29）

依使用者 2026-10-29 的 [計畫](BLADEMASTER_DEMO_PLAN.md)：P0 工作包 A 有條件接受（M8 FAIL，部分後續項目未完成）；B–D 完成。細節見 [完成報告](reports/BLADEMASTER_DEMOS.md)。

- 最終組合，全部是 live Native、law 5、實體夾持、`blademaster_workcell_v1`，四個都 **PASS**：
  - **Demo 1a 完整切斷**（`path_v2`）：凍結 strict 也 PASS，切口 1.5 mm。
  - **Demo 1b 20° 斜切**：依 swept-plane 準則 PASS；凍結 strict 的切口位置與切後持續時間窗口兩項 FAIL（約 5.7 s 才切斷，晚於 5.6 s 窗口）。
  - **Demo 2 削除凸起**：去除 97.9%，底材完整；精度 5/8 FAIL，另外報告。
  - **Demo 3 兩刀三塊＋刀身推開**（`path_v2_vision`）：連凍結的 v3 slice-push 報告也 PASS，推開 18.7 mm。
  - [所有嘗試](evidence/phase3/demos_live_002.json)、[媒體](media/demos/)。
- Phase 1，law 5、1 mm：
  - 依計算式準則 `adr0006a1_adr0008`，M1–M7、M9 PASS。
  - M8 在理想追蹤下 FAIL：第一刀把長條推開約 2.3 mm，第二刀宣告切面誤差 2.0000000670 mm，凍結界限 2 mm。Live Demo 3 以 [ADR 0009](adr/0009-demo3-wrist-depth-retarget.md) 的手腕深度量測修正。
  - 審查結論：rcs-review ACCEPT WITH CONDITIONS（[紀錄](evidence/phase1/matrix_law5_adjudication_001.json)）。
- 保留的失敗：
  - law 4 全矩陣（TRIM 0%）；
  - 第一輪 v1 控制器的 Demo 2（64%）與 Demo 3（第二刀與宣告切面相距 2.0000000670 mm，超過 2 mm 界限）；
  - 中斷的輸出；
  - 2 mm CPU 35° 削除（strict xfail）。
- 限制：
  - 材料參數暫定、未標定；
  - law 5 的前緣預指派改變刃口前方的動態，切後兩塊被推開 ±2.4 mm；
  - 拉切／鋸切時 swept normal 條件不佳；
  - live runtime 未逐步記錄 stroke；
  - Lab parity 與 P1／P2 未做。
- 後續項目（非阻塞，來自兩次 rcs-review；做的時候要用新版本，不改任何已記錄的 verdict）：
  - Demo 3 評估補上「兩刀之間的抬刀與移位不加標籤」檢查；
  - Demo 1 outcome 加上切口位置準則；
  - live runtime 逐步記錄 stroke 狀態；
  - 診斷 law 5 改變刃口前方動態的機制（per-field FLIP）；
  - 鋸切前加 swept normal 保護；
  - stroke 持續改為也考慮轉動；
  - capture rate 改以非展示相機量測；
  - 展示相機的影格加上 presentation-only 旗標；
  - renderer 加回 Global D455。
- 完整 CPU regression（`.venv/bin/python -m pytest -q`，law 8 之後）：780 passed、5 xfailed（strict：law 4 25°/35°、law 5–7 35° 的 2 mm 削除），0 failed，815 s（`artifacts/wp10_claude/logs/cpu_full_final_002.log`）。較早一次：689 passed、3 xfailed（strict：law 4 25°/35°、law 5 35° 的 2 mm 削除），0 failed，359 s（log `artifacts/wp10_claude/logs/cpu_full_final_001.log`）。
- 環境注意：背景子代理執行「寫入 `artifacts/`」的 Bash 指令曾卡在權限等待。後來 `/mnt/data/ningan` 已加入工作目錄；GPU／Isaac 執行改由 coordinator 直接跑。

## Paused checkpoint: Claude diagnosis of CUT/TRIM separation

Paused at the user's request for review; WP-10 is not complete and nothing is
merged. Root cause of TRIM 0% removal and the Lab CUT single component:
law-2 labels are written only where a 1 mm edge segment did cutting work in
that substep, so code-zero particles bridge the cut through the 4 mm reference
stencil (Lab's measured edge actually went deeper than Native). An opt-in
experimental law 3 in `tlmpm_v3.py` adds swept-surface sides and grid-scale
lateral/persistence/lead closures; its 9 CPU tests pass, laws 1/2 remain
bitwise equal to v2. Earlier law-3 variants still FAIL the 1 mm TRIM probe;
the current variant's GPU results and every live run are pending. New
`--cutting-law`, measured-trajectory replay and stencil-bridge tools. No new
live scene images yet. See
[the diagnosis](reports/WP-10_CLAUDE_DIAGNOSIS_20261029.md).

## User-requested checkpoint: Claude continuation

Codex work stopped at the user's request. Resume from
[the WP-10 handoff](HANDOFF_WP10_CLAUDE_20261029.md) on
`claude/lucid-mendel-vt7n82`. WP-10B is complete; C/D are partial, and E/F/G
formal validation is NOT_RUN. Final checkpoint checks: 451 quick CPU tests
PASS, 18 cpu_sim tests deselected; v3 syntax/CPU kernel compilation PASS,
v3 physical validation NOT_RUN. The latest full CPU run before subsequent
drafts remains 461 PASS. No learned policy, PPO, actual LLM-agent run or final
WP-10 robot-video matrix has been produced. No final PR or merge.
The original owned viewer is restored READY and paused in
`rcs-wrist-v5-remote-restored` on GPU 0, with fresh output and
[restoration receipt](evidence/wp10/viewer_restoration_20261029.json).

## Completed validation: WP-10B live policy interface

Both actual runtimes complete 1,200 identical fixed-tool control-fuzz steps,
with identical IK statuses and bounded deltas, legacy CUT PASS / default TRIM
FAIL regression, explicit IK rejection/hold, cut-digit checks, and complete Gym
state/vision episodes. Separate physical-grasp stress succeeds for 1,200 Native
steps but loses the tool in Lab (209 total steps); this holding limitation is
preserved. Full CPU regression is 461 PASS / 0 SKIP in 92.45 s.
[WP-10B](reports/WP-10B.md).

WP-10C is partial: Native CUT-01-R and CUT-02 strict PASS; both Lab cuts strict
FAIL with a connected component. Initial Native TRIM-01-R has 394 rejected IK
commands and strict FAIL; Lab TRIM fails pregrasp before the formal episode.
Both runtimes pass the frozen seven-case ruler rigid-grasp controls and the
complete 401-pose CUT camera/tool visual-clearance audit. A raised 35-degree
candidate passes scoped Newton geometry preflight; live execution is NOT_RUN.
An ideal-tracking 1 mm law-2 TRIM still removes 0%. [WP-10C](reports/WP-10C.md).

WP-10D GPU probes have a 0.037611 slicing/push resistance ratio and a PASS
three-part/two-digit outcome. Skim separation FAIL and the CUDA graph position
comparison FAIL (5.375594 µm versus 1 µm tolerance) remain visible. Two provisional
food profiles have finite-state dt sweeps, without force convergence or
calibration. Exact armed-withdrawal and full live angle matrices remain NOT_RUN.
[WP-10D](reports/WP-10D.md), [C/D evidence](evidence/wp10/wp10cd_handoff_v1.json).

## Historical input checkpoint: WP-10 parts A/B, CPU only

Branch `claude/lucid-mendel-vt7n82`, handed off to Codex for the GPU/live work.
Done and CPU-verified: opt-in policy interface (delta actions, step limits, IK
rejection, state profile, severed-bond reward, Gym adapter, agent tools with
PNG views), `ruler_knife_v1`, `tlmpm_v2` with opt-in cutting law 2 (slicing
work, energy-gated labels, blunt handle) whose law 1 equals frozen v1 bit for
bit, `cutting_tasks_v2` (CUT-01-R, TRIM-01-R, CUT-02; v1 thresholds), an
ideal-tracking kinematic backend, scripted knife-motion policies and
validation/probe/video scripts. Kinematic full episodes: CUT-01-R, CUT-02 and
CUT-01 PASS the frozen oracle for scripted reference, push, saw and Gym
controllers; TRIM-01-R removes the bump but fails strict surface metrics at
4 mm. Every GPU, Isaac Sim and Isaac Lab result is **NOT_RUN**; the ruler
knife is not yet authored by either live runtime. Report:
[WP-10A](reports/WP-10A.md); plan: [WP-10 specification](WP10_SPECIFICATION.md);
usage: [POLICY_CONTROL.md](POLICY_CONTROL.md); decisions:
[ADR 0003](adr/0003-ruler-knife-and-policy-interface.md).

## Current: WP-09 Evaluation v2 and hand-aligned wrist-camera v5

Evaluation v2 is implemented and validated as a read-only, shared CUT/TRIM
interface with separate recorded-state integrity, cutting outcome and surface
quality. Diagnostic/Research/Precision profiles retain the exact original
strict benchmark and report continuous metrics, full missing masks, independent
analytic/represented targets, nonflat/mesh/implicit geometry and predeclared
regions. Eight actual historical/new episodes match every original recorded
strict status/check/metric. Fixed TRIM still removes 98.5294% and fails precision;
the seed42 three-bump recording reports 96.2791% overall and per-bump values.
The new research outcome label does not assert flatness or full physical
fidelity. Full CPU regression: 414 PASS. See [Evaluation v2](EVALUATION_V2.md)
and [compact evidence](evidence/evaluation_v2_validation.json).

Franka 1 now stays at its original safe park in both runtimes. Both v5 D405s use
axis-aligned constant hand-local mounts without observation pose or wrist aiming.
V4's actual Native/Lab full CUT, Lab CUT/TRIM smokes and explicit secondary
motion remain preserved. New v5 Native CUT/TRIM each run 400 steps and Lab's
separate two-step smoke passes RGB-D/FK/clock contracts. Both camera mounts
clear all 401 sampled robot poses of the new Native CUT; v4's primary
bracket/link7 intersection remains a preserved historical failure. The original
full-wrist-footprint visibility screen still fails and remains separate from
cutting. README media and the live remote viewer use the new rig.
The README now shows separate CUT-01 and TRIM-01 v5 scene images from the
actual synchronized t=4s captures, with provenance and episode videos for both.
See [camera v5](WRIST_CAMERA_V5.md) and
[the recovered interruption checkpoint](RESUMPTION_20261029.md).

The earlier clean-bootstrap and randomized/capsule jobs have completed:
physical smokes and both capsule CUT task scores PASS; seed43 randomized CUT
executes/splits but retains its strict cut-location FAIL. Lab's Cartesian
transfer control remains failed/incomplete. No old evidence was overwritten
and full WP-08 physics acceptance remains false.

## WP-08 in progress: independent tool and physical finger grasp

Baseline `1500ed7` is frozen on branch `wp08/physical-tool-grasp`, with the
original CUT/TRIM criteria and all WP-07 evidence retained. Baseline CPU277 and
both actual legacy runtime smokes passed before implementation. `legacy_fixed_tool`
remains explicit and the default until end-to-end physical grasp is validated.

The shared compound tool now has a continuous handle/shaft/blade, derived
finite mass/inertia, nominal grasp frames and a separate measured edge. Native
PhysX pregrasp002 established actual bilateral loaded handle contacts for a2s
pre-episode settling phase. Its subsequent runtime falsely returned PASS_SMOKE
after camera edits invalidated a rigid tensor view; the first full attempt was
therefore aborted and rejected. Camera assets now precede physics views, and
smoke003 completed two formal steps with fresh contact data, all three RGB-D
streams and no invalid-view errors. Neither smoke proves physical cutting.
All API/initialization failures and raw traces are preserved. Full candidate-v2
CUT/TRIM episodes now execute in both actual runtimes with three RGB-D streams
and reset checks. Both CUT strict task scores pass; both2mm TRIM episodes
separate1.608g chips and retain strict surface FAIL. Native material ON/OFF
matched runs pass the frozen causal-response engineering screen, with actual
arm/tool/contact differences; full coupling-energy acceptance remains separate.

Native v2 also passes all seven frozen hold/load/torque/stress/weak-grip/
low-friction/release controls. A contact-event geometry timing bug was corrected
using explicitly recorded pre-step poses without changing contact tolerances;
old strict failures remain intact. Opening fingers gives an independently
measured unsupported gravity interval. Newton's initial point-contact model
retains actual clearance-transfer failures despite passing the full cutting
paths. A finite, geometry-derived finger-patch torsion model is being evaluated
as a separate candidate, without changing TLMPM or task scoring.

The shared API and same-instance GUI now distinguish actual tool, commanded
edge and measured hand, expose physical gripper commands, and pause on grasp
loss until reset. Local Chrome over the Tailscale URL passed11 real streamed
GUI transactions, including release/drop/automatic loss termination, reset,
CUT/TRIM switching, camera selection and typed joint control. External Mac
connectivity remains NOT_RUN. New actual physical-grasp RGB-D videos are linked
from README. Randomized/capsule regression and a fresh isolated bootstrap are
running; optional rack skill attempts and their failures are preserved.

See [physical tool use](PHYSICAL_TOOL.md), [frozen controls](PHYSICAL_TOOL_CONTROLS.md)
and [latest physical-grasp media](media/wp08_physical_grasp_v2/manifest.json).

See [the supplied specification](WP08_SPECIFICATION.md). No driver, unrelated
project, solver cutting law or frozen scoring criterion was changed.

## Latest: target-centred Global, interactive viewing and portable deployment

The current v3 scene keeps both D405 cameras mounted on actual articulated
hands and places Global D455 at `(0.28,-0.42,0.46)` m looking toward
`(0,0,0.02)` m, about0.67 m from the target. Two new400-step native CUT/TRIM
episodes retain their physical outcomes and Task statuses. Actual Global
material-pixel minima improve8.13×/8.92× over v2; full-workcell coverage is
intentionally superseded by the user's close-cutting-view request. Both Lab
two-step camera smokes pass separately. All earlier evidence is preserved.
See [the new close-view report](GLOBAL_CLOSE_VIEW.md),
[compact receipt](evidence/global_close_view.json) and latest [README](../README.md).

The Remote Interactive Viewer reuses the existing Isaac Sim environment and
native runner, adding one full Kit GUI/WebRTC bootstrap and a queued control
panel. Actual local Chrome connected through the Tailscale URL has decoded
1280×800 video and acknowledged streamed camera, step/reset/play/pause and
Task-selection clicks. GPU startup/idle tests prove rendering does not advance
robot/material state. Keyboard and external Mac-client results are recorded
separately in [the interactive viewer report](REMOTE_INTERACTIVE_VIEWER.md).
No independent simulator/container or alternative cutting engine is used.

WP-07 Portable Deployment & Reproducibility completed clean-environment
verification on the original host. The third public clone installed new managed
Python and all three isolated environments in 516.067 s, then ran both actual
runtime smokes and four 400-step CUT/TRIM episodes at frozen source `f2bf240`.
All runtime, RGB-D and physical-separation contracts passed; strict CUT metrics
passed and strict TRIM surface metrics remain FAIL. The scoped result is
`PASS_DEPLOYMENT_WITH_EXPECTED_TRIM_SURFACE_FAILURE`, with
`full_benchmark_ready=false`. The frozen run passed 276 CPU tests; later local
and CI checks passed 277 separately. A one-CUDA-visible-GPU smoke passed both
runtimes on the same dual-card host. Independent cross-host reproduction is
NOT_RUN because all three checked Linux peers rejected SSH authentication.
The first download-timeout failure and second native stage-loading failure are
preserved, together with exact native camera-reset and other known limitations.
See [deployment workflow](PORTABLE_DEPLOYMENT.md),
[actual validation report](PORTABLE_DEPLOYMENT_VALIDATION.md) and
[machine-readable receipt](evidence/portable_deployment_validation.json).
No drivers, RSI, other projects or policies were changed.

Final local streamed-input proof passed14GUI transactions, including typed
Cartesian/joint actions and coupled material steps. The active same-instance
Kit session is paused on Global, with the browser client slot released for
external Mac verification. The local proof does not substitute for that Mac
client test. See [browser receipt](evidence/remote_web_viewer.json).

## Preserved v2: moving wrist D405 prototypes and workcell overview

The new `wrist_d405_global_d455` layout has completed CUT-01 and TRIM-01 in
both actual Isaac runtimes: four full 400-step / 8-second episodes, 492 captured
camera frames and 1,604 measured hand/camera samples. No solver, material or
frozen Task success criterion was changed for the camera update. The preserved
fixed-camera recordings and all unsuccessful pilots remain available.

| Camera / validation | Measured result | Status and boundary |
| --- | --- | --- |
| `wrist_d405_0` / `wrist_d405_1` | True children of each actual `panda_hand`; fixed hand-to-camera transforms, actual articulated poses checked every 20 ms; synchronized 848×480 RGB-D | FK / inverse / timestamps / runtime optics **PASS** in all four episodes; both sensors explicitly **`ideal_d405_prototype`** |
| Closer `global_d455` | Official D455 USD, World-fixed at `(1.10,-1.20,1.05)` m; native 640×480 RGB-D, whole workcell in view | Minimum valid-depth fraction 15.8337% / 15.8490% in native CUT / TRIM versus 6.3249% in the preserved far CUT; engineering visibility **PASS**, native sensor calibration **NOT_VALIDATED** |
| Actual secondary-arm motion probe | Native PhysX joint excursion 0.081217 rad; hand translation 21.74 mm; optical translation 28.11 mm; fixed Global drift zero | Separate 2-second / 100-step motion proof **PASS**; cutting goal **NOT_RUN** |
| CUT-01 physical regression | Both runtimes retain 96,000 particles / 144 g, actual support graph 1→2, two 72 g parts and 2.4 s sampled separation persistence | Independent Task metrics **PASS**; delayed interface-work limits still **FAIL** |
| TRIM-01 physical regression | Explicit 2 mm particles, 1.608 g chip + 144.024 g base, 98.5294% target-material removal; no history rewriting | Permanent separation **PASS**; strict surface accuracy and interface-work limits **FAIL**, same as the earlier coarse baseline |
| Visibility and mount limits | All three cameras see the knife throughout active TRIM; secondary wrist is palm-occluded at 6.4–8.0 s after withdrawal. All housings and other robot pairs clear; primary bracket overlaps its own link7 flange at the fixed anchor | Frozen all-frame secondary-knife and strict bracket clearance screens **FAIL**; visual camera mass/colliders are not modelled |
| API / recording / regression | New named streams; optional Global policy input; explicit Cartesian/joint actions in both live runtimes; synchronized videos, per-step transform bundles, independent source/hash checks | Full Task execution, API smoke, prototype optics and hardware-model fidelity are reported independently; 235 CPU tests pass |

Native depth tails and invalid pixels are retained, including a rare Global
valid-depth error of approximately 0.849 m. Programmable views and geometric
synchronization do not establish calibrated sensor accuracy. D405 working-range
controls at 6 / 8 / 49 / 51 cm and near-foreground occlusion were actually run
on the GPU; these validate ideal first-hit optics, not stereo/noise/calibration.
The final profile is 848×480, effective 87×56.485° FOV, logical range 0.07–0.50 m.

Code and commands: [WRIST_CAMERAS](WRIST_CAMERAS.md),
[native integration](WRIST_CAMERA_NATIVE_PIPELINE.md),
[independent full validation](WRIST_CAMERA_VALIDATION.md),
[compact receipt](evidence/wrist_camera_audit.json),
[actual motion proof](evidence/native_wrist_motion_proof.json),
[shared API](SHARED_TASK_API.md). Current three-camera videos and full measured
transform bundles are linked from [README](../README.md). Repository text,
configurations, evidence labels and documentation are English.

Final raw episodes: `artifacts/isaac_lab_wrist_cut01_002`,
`isaacsim_wrist_cut01_002`, `isaac_lab_wrist_trim01_001`,
`isaacsim_wrist_trim01_001`. Motion proof: `isaacsim_wrist_motion_probe_001`.
Earlier far-Global wrist runs, source-race trial, rejected geometry, VGA optics
trial and all original D0–D3 / fixed-camera evidence are preserved separately.
No training, real-robot operation, RSI changes, driver changes or reboot occurred.

The first public CI run for this camera delivery failed during collection:
the new CPU reference-ray test imports Warp, absent from the workflow's old
install list. The workflow now explicitly installs the existing pinned
`warp-lang==1.17.0`; no assertion is skipped or relaxed. An isolated environment
without Isaac/Newton, with GPUs hidden, passes all235 tests in1.49 s. The
[initial failed run, logs, versions and clean recheck](evidence/wrist_ci_validation.json)
are retained independently of the physical episodes.

## Preserved: fixed-camera cutting task delivery

Fixed CUT-01 and TRIM-01 have each run in **Isaac Lab 3.0 GA and Isaac Sim 6.2**,
completing 400 control steps / 8 seconds. An actual Franka holds the fixed blade, the measured edge contacts material, and TLMPM produces
persistent separation with material reaction fed back to the arm. The second arm remains parked. Both tasks share the
`reset/step/observe/render/evaluate/close` interface and independent Task evaluator;
no models were trained and no RSI, other environments or drivers were modified.

| Delivery this round | Actual result | Status and boundaries |
| --- | --- | --- |
| Fixed CUT-01 | One Lab/Sim episode each; 144g→72g+72g, 1→2 actual MPM components, cut-location reference-support bound 1.5mm, persistent separation for 2.4s after withdrawal | **PASS_TASK_METRICS**; complete camera/coupling acceptance is separate |
| Fixed TRIM-01 | Explicit 2mm initial particle sampling; one Lab/Sim episode each; unchanged 145.632g, 1.608g chip, 98.5294% target-material removal | **Physical separation PASS**; original full-surface accuracy **FAIL_TASK_METRICS** |
| Randomization | Frozen distribution: variable lengths, cut positions, XY/yaw and 1–3 continuous dome/Gaussian/ridge/irregular bumps; both native seed42 tasks fully executed | CUT produces 66.6/59.4g parts and passes Task; TRIM produces three chips with 96.2791% removal, geometry still FAIL; no success-rate claim |
| Native cameras | Per episode: 41 frames of dual 640×480 RGB-D + 1280×960 Global; current surfaces and actual robot poses enter the native Kit pipeline | Geometry/time/CAM-03 PASS; CAM-04, chip pixels and strict reset still include FAIL; Lab explicitly uses ideal Warp |
| Public videos | Both fixed tasks provide Global, dual RGB, dual depth, synchronized views, command/actual paths and source hashes | Encoded from actual captures; raw retained; no USD replay presented as live |
| Regression | 205 CPU tests; 8 golden hashes for the old D0–D3 sampler unchanged; additional actual 5-step Lab D0 prefix comparison | Code/reference/mass agree, maximum position difference 52.15nm; not a complete D0–D3 dynamics rerun |

Both original 1mm live TRIM runs still FAIL: two code0 particles bridge the material, preventing actual disconnection. Extended-path,
second actual-cut and 2mm-grid negative results are preserved; only the later 2mm-particle configuration produced separation in both live runtimes.
This is an explicit resolution-dependent engineering result, without changing defaults, deleting particles or claiming convergence.

Fixed coarse-particle TRIM has maximum positive residual approximately 0.106–0.109mm, but full-footprint p95 approximately 1.67mm
and overcut approximately 2%, still failing the original criteria. Independent no-bump target controls and three-dimensional surface-distance measurements are complete:
represented-column metrics still FAIL; bidirectional surface-distance p95 approximately 0.29mm is descriptive only, with no new thresholds fitted.
Fixed-task partitioned coupling-work residuals remain over the limit, and material/fracture parameters are not calibrated against real samples. Randomized TRIM separates,
but retains tool contact after reference withdrawal; not all three chips can be claimed blade-clear or in free flight.

Main new evidence: `artifacts/isaac_lab_cut01_001`, `isaacsim_cut01_001`,
`isaac_lab_trim01_coarse_001`, `isaacsim_trim01_coarse_001`,
`isaac_native_randomized_benchmark_001`; compact sources/hashes/measurements and FAIL results are in public docs.
Complete tables and commands: [CUTTING_BENCHMARK](CUTTING_BENCHMARK.md),
[fixed experiments](reports/CUTTING_TASK_EPISODES.md),
[randomized experiments](reports/RANDOMIZED_TASK_EPISODES.md),
[native cameras](TASK_CAMERA_VALIDATION.md),
[shared API](SHARED_TASK_API.md), [surface controls](REPRESENTED_TARGET_VALIDATION.md).
Videos are linked from [README](../README.md). Local raw data is retained; large assets are not committed to Git.

## Preserved: previous Isaac runtime integration and acceptance

This round actually executed Isaac Lab 3.0 GA and
Isaac Sim 6.2 / Kit110.4 on the original machine and driver 570.211.01, using officially supported isolated Torch2.11/cu128 environments.
Measurements supersede the earlier blanket CUDA13 BLOCKED classification for the complete runtime.
No driver upgrade, reboot, RSI/other-project environment modification or model training occurred.

| Actual path | Completed work | Acceptance boundary |
|---|---|---|
| Live Lab | Actual DirectRLEnv/InteractiveScene/NewtonManager; 400steps, 8s; actual dual Frankas, fixed blade, TLMPM and reaction feedback; 1.656g chip and two mechanical components | Kit-less cameras are explicitly labelled full-scene ideal Warp RGB-D, not native D455 |
| Live Kit | Sim6.2 native PhysX; 400 control / 4000 rigid steps, 8s; dual native SingleView D455 + 1280×960 Global, persistent separation, actual camera data/videos | Episode execution is separate from all 18 acceptance gates; no full-integration PASS claim |
| Trajectory correction | Optional joint-velocity feedforward with identical path/material/gains; Lab active max0.507mm, Kit0.480mm, passing the 1mm limit | Original-controller 1.24–1.26mm failures preserved; improved tracking does not automatically pass coupling energy or flattening |
| Native depth | OFF/ON and development/held-out controls frozen in advance; actual OVRTX and actual Kit sampling correspondence pass separately | Complete RAW Z retained; not physical D455 calibration; original H0 failure, noise and invalid pixels retained |
| Geometry controls | Known closed boxes, rotations and analytic-volume tests distinguish the 16mm equivalent base from 2mm sampling and surface expansion at density0.25 | Shrinking the rendered surface is not physical flattening; new profiles still need deformation/task validation |

Explicit remaining failures: D0 reconstructed-surface maximum residual approximately 1.9–2.0mm; FF's 20ms delayed coupling interface-work
relative residual exceeds 5%; some FF episodes do not meet postwithdraw cross-component recontact thresholds; strict Kit camera reset and
per-particle repeatability also fail. These remain FAIL / NOT_RUN, without relaxing prefrozen thresholds.
Lab initial-reset q/material/ideal-depth differences are 0; full same-source repetition still has per-particle differences,
so ENV-01 FAIL. New native rigid-penetration audit: maximum 32.494µm / tool-to-rigid 0; ENV-02 passes.

Main evidence: `artifacts/isaac_lab_d0_001`, `isaac_lab_d0_feedback_off_001`,
`isaac_lab_d0_feedforward_001`, `isaacsim_live_native_d0_001`,
`isaacsim_live_native_d0_ff_001`, `single_view_kit_audit_003`.
Details: [ISAAC_LAB_INTEGRATION](ISAAC_LAB_INTEGRATION.md),
[ISAAC_SIM_VALIDATION](ISAAC_SIM_VALIDATION.md), [CAMERA_VALIDATION](CAMERA_VALIDATION.md).

The user's next stage is CUT-01 complete bar severing → TRIM-01 single-bump bar flattening →
randomization → regression testing. After preserving current integration evidence and unresolved issues, continue with shared Task/
Geometry/Fixture definitions, workpiece-local targets and a unified API; retain the original D0–D3 baselines.

### Earlier CUT-01 / 1mm TRIM-01 implementation and negative results

Geometry, material, fixtures, fixed blade mounts and independent scoring criteria for both tasks are frozen; see
[CUTTING_TASKS](CUTTING_TASKS.md). Added an optional workpiece-frame material interface and finite fixture contacts;
old D0–D3 contact/surface defaults remain. New tasks use complete 1mm particle-volume sampling:
CUT-01 has 96,000 particles / 144g, and TRIM-01 has 97,044 particles / 145.566g.

Both Lab and Kit completed fixed CUT-01 in actual 400-step / 8-second episodes and passed independent Task scoring:
144g splits into two 72g main parts, actual MPM support graph 1→2, separation persisting for 2.4 seconds after withdrawal, and cut-location
midpoint bound 1.5mm. Both fixed TRIM-01 episodes also complete 8 seconds but are FAIL_TASK_METRICS:
two displaced code0 particles still bridge the base and chip; 97% of particles having side labels
cannot substitute for actual separation. Prefrozen extended-path / second actual-cut diagnostics are being investigated, without promotion yet.
The four complete experiments and original failures are in [Task episodes](reports/CUTTING_TASK_EPISODES.md).
Initial JSON Boolean-serialization failures are preserved separately; corrected runs use fresh output directories.
New tasks use the current-volume surface profile that passed independent closed affine-geometry controls; this validation
does not establish task flattening and does not alter cutting physics or the old D0 renderer defaults.

The shared [actual environment API](SHARED_TASK_API.md) passed actual Cartesian and joint
command checks over 2 steps each in Lab/CUT and Kit/TRIM; official observations contain only dual RGB-D and proprioception. Independent [camera audits](TASK_CAMERA_VALIDATION.md)
show native depth coverage approximately 97.5–99.1% in material regions; full-frame CAM-04 holes/outliers still FAIL.
Trajectory, full-geometry and timestamp audits pass, without promotion to complete camera or cutting acceptance.

## Previous standalone baselines and historical record

This round actually executed and integrated Physical Cutting → Franka Control → RGB-D → Simulation Benchmark,
delivering a standalone environment for continued research. The user authorized public
GitHub, ongoing commits/pushes, agents and reasonable adjustments to the original report's stage gates. Numerical results and
capability statuses are not rewritten when gates change; no RSI, other projects or drivers were modified, and no Policy was trained.

| Package | Delivered/measured | Remaining validation |
|---|---|---|
| WP-00 | Independent repo, environment/versions/official source pins, official standalone Newton smoke, CPU CI; this round adds actual live Lab/Kit D0 | Four historical source-only failures preserved; distinguish cu130 compatibility from this round's cu128 |
| WP-01 | Continuous base+bump, shapes/seeds, Newton MPM settling/indentation/scraping, fixtures, traces/videos | Material uncalibrated; complete convergence outstanding |
| WP-02 | Seven CRESSim source corrections, 16 actual-header controls and GPU stability; independent Warp TLMPM online persistent cuts | CRESSim lacks persistent separation; project TLMPM is an uncalibrated prototype, not official BladeMaster code |
| WP-03 | Current-mechanical-component ParticleSurface and live full-scene ideal dual RGB-D; native Kit pipeline and held-out sampling correspondence added this round | Hardware-noise calibration, strict reset and full dynamic-depth agreement remain separate acceptance items |
| WP-04 | Dual-Panda dynamics, IK/joint/Cartesian control, collisions, fixed matching-geometry wedge blade and actual material-reaction feedback | 20 ms explicit delay; no monolithic/coupling-energy convergence; second arm parked |
| WP-05 | ResearchCuttingEnv reset/step/observe/render/close, complete 400-step test; four D0–D3 physics+image benchmarks | Fixed multi-bump paths leave residuals; some D3 parameters not connected; no learning or task-success-rate claim |
| WP-06 | Code, commands, compact hash evidence, numerical traces, video paths, negative results, CI and this document | Large raw artifacts retained locally; full scientific benchmark acceptance incomplete |

Available capabilities: `persistent_cutting_D0=PASS_EXPERIMENTAL`, `articulated_control=PASS`,
`coupled_cutting_D0=PASS_EXPERIMENTAL`, `live_ideal_rgbd=PASS`, `standalone_rtx_rgbd=PASS`,
`benchmark_executable=true`. The original complete scientific flags `PHYSICS_PROVEN / VISION_PROVEN /
WORKCELL_PROVEN / BENCHMARK_READY` are not all claimed passed; this does not make the current environment unusable.

## Measured highlights and formal boundaries

- Frozen TLMPM matrix: 11 completed physics cases plus preserved setup failure; engineering evidence for single cuts, withdrawal, independent chip motion and recontact. 19,340 particles; single-cut chip 140 particles / 1.680 g; seven actual GPU mathematics/local-impulse controls pass.
- Same-seed repetition: maximum position difference 10.59 µm and velocity difference 0.00452 m/s, within prefrozen engineering tolerances; not bitwise deterministic, with one particle's label timing differing in an intermediate frame. Final labels/components agree.
- Frozen second cut was a finite local incision and did not create another separated part; T6 FAIL is preserved. A later full-cross-section crossing path with flat rigid support, removed side walls produces 1→2→3 components over 4.5 s, minimum detF0.9027 and unchanged existing digits. Two failed free-specimen controls remain; this is not two-direction Franka validation.
- Complete coupled D0: 8 s, actual 1.656 g chip, maximum material reaction 1.041 N; ON/OFF feedback control changes tool motion by 0.345 mm, confirming reaction affects the robot. Delay 20 ms; interface-work residual 0.326 mJ is diagnostic, not total-energy convergence.
- Live API: 400steps / 8 s, 164.55 s wall, dual 640×480 RGB-D and measured proprioception every step; initial reset q/depth error 0, truncation passes, no force/oracle/global in observations.
- Native RTX coupled002: 41 current mechanical surfaces and 401 measured robot poses; chip visible in both depth streams; mesh-depth QA passes. Native D455 postprocessing and ideal RGB-D are recorded separately; fallback is not presented as sensor validation.
- Native SingleView fresh tilted-plane noise OFF/ON: registration helper preserving actual RAW depth implemented; p95 1.309/1.344 mm exceeds prefrozen 1 mm, with noise mainly changing the valid mask. `simulated_d455` remains BLOCKED/prototype; see [SINGLE_VIEW_REGISTRATION](SINGLE_VIEW_REGISTRATION.md).
- D0–D3 seed42: all four 8 s physics + 5-frame full-scene ideal RGB-D cases PASS_EXECUTION; D2/D3 residual-height proxies 21.7/16.0 mm do not establish task success.
- 4/3/2 mm and half-dt probes show chip mass still changes, without convergence; coarse recontact overlap, J2 parameters and sharp-blade model limitations are explicit.
- A clean isolated CPU-only environment without Newton/Warp/CUDA passes 93 tests; actual GPU/RTX/control evidence and CPU tests are recorded separately.

## Main entry points and local evidence

```bash
.venv/bin/python -m robotic_cutting_sim.cli research-demo --output-dir artifacts/fresh_live
.venv/bin/python -m robotic_cutting_sim.cli benchmark --output-dir artifacts/fresh_benchmark
.venv/bin/python -m pytest -q
```

| Evidence | Local artifact |
|---|---|
| Frozen physics/repeat/refinement | `artifacts/TLMPM_validation_001`, `TLMPM_validation_two_cuts_001` |
| Full-cross-section second cut | `artifacts/tlmpm_supported_cross_cuts_001` |
| Actual robot-reaction coupling and control | `artifacts/coupled_robot_cut_002` |
| Live API | `artifacts/research_env_full_001`,`research_env_demo.mp4` |
| Actual cutting RTX | `artifacts/ovrtx_coupled_002`, `ovrtx_coupled_closeup_002` and `coupled_rgbd_qa_002` |
| D0–D3 benchmark | `artifacts/simulation_benchmark_001` |
| CRESSim before/after corrections | `artifacts/CRESSim_candidate`, `cressim_scrape_promoted_001`; historical FAIL results retained |

Detailed commands/hashes/negative results: [TLMPM_VALIDATION](reports/TLMPM_VALIDATION.md),
[COUPLED_ROBOT_VALIDATION](COUPLED_ROBOT_VALIDATION.md), [WP-03-cutting](reports/WP-03-cutting.md),
[RESEARCH_ENVIRONMENT](RESEARCH_ENVIRONMENT.md), [BENCHMARK_REPORT](BENCHMARK_REPORT.md).
Public repo: https://github.com/AnsonNing/RoboticCuttingSim . Large data and official assets are not committed to Git.
