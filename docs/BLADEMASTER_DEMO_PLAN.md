# BladeMaster 風格三個 Demo 與切割研究場景 — 執行計畫（2026-10-29）

使用者決定（2026-10-29）：目標場景、刀具與展示方式參考 BladeMaster
（專案頁 <https://jango6324.github.io/blademaster/>，論文 Section III-C、Eq. (4)–(6)、Fig. 1/5/8）。
不再擴張新的 Agent 或訓練框架；先完成一個真正可操作的 Robotic Cutting Workcell 與三個 Demo。
> **狀態（2026-10-29 晚）**：A 有條件接受（M8 在理想追蹤下 FAIL，部分後續項目未完成）；B–D 完成。四個最終 live Native episode 都 PASS（Demo 1a、1b、2、3），完整結果、失敗紀錄與限制見 [完成報告](reports/BLADEMASTER_DEMOS.md)。第 1 節是開工前的暫停點，保留作歷史紀錄。P1／P2（E–H）未開始。

本文件取代 [WP-10 規格](WP10_SPECIFICATION.md) 第 3 節的執行順序；WP-10 的 E/F/G 降為後續（見第 6 節）。
工作分支：`claude/lucid-mendel-vt7n82`。交接診斷：[WP-10 Claude 診斷](reports/WP-10_CLAUDE_DIAGNOSIS_20261029.md)。

## 0. 第一階段完成的定義

- **Demo 1 — Complete Cutting**：Franka 0 以兩指物理夾持長直刃（`ruler_knife_v1`），把連續軟材料完整切成兩塊，撤刀後持續分離。
- **Demo 2 — Surface Trimming**：Franka 以適當刃角沿表面削過，把凸起真正分離成 chip；底材仍為完整主體。
- **Demo 3 — Sequential Slicing & Manipulation**：同一把刀連續切兩刀（`cut_index` 0、1）成三個機械獨立部分，再以刀身鈍面、`cut_enabled=False` 用真實接觸力把其中一塊推開。

三者都要有：真實刀具與材料物理、6DoF 動作、穩定物理夾持、RGB-D、Reset、Recording、Evaluation。
驗收影片必須由 **Live Isaac Sim（Native PhysX/Kit/RTX）** 產生；kinematic、replay 或示意圖不能取代。
Franka 1 保持正常停放，只保留手腕 D405（rig v5，不瞄準、不做觀測姿態最佳化）。

## 1. 現況（暫停點）

- Native CUT-01-R／CUT-02 strict PASS（law 2）。Lab 兩個 CUT 為 1 component。TRIM-01-R 全部失敗（含理想追蹤 1 mm：0% 去除）。
- 根因：law 2 只在「該 substep、該 1 mm 刃段」有切削功時寫 side label；未標記（code 0）粒子透過 4 mm reference grid 的共用節點橋接兩側。
- 實驗性 law 3（`tlmpm_v3.py`，opt-in）：掃掠面分側＋橫向／持續／前緣閉合。9 個 CPU 測試通過，law 1/2 與 v2 逐位相同。
  1 mm 結果：TRIM 25°／35° 仍 0%；Lab CUT-01-R／CUT-02 量測軌跡重播已分成兩塊（約 72/72 g、179/181 g），但切口位置界面 4.0 mm，超過凍結門檻。
- 工具：`scripts/replay_measured_edge.py`（量測軌跡材料重播）、`scripts/analyze_cut_bridges.py`（橋接診斷；**目前用全 1 權重，需改成實際 stencil weights／stress gradients，與 `mechanical_components()` 一致**）、`--cutting-law` CLI。
- 環境：根磁碟約 99%；新輸出一律放 `artifacts/wp10_claude/`（→ `/mnt/data`），`TMPDIR=/mnt/data/ningan/tmp`。主機曾重啟，舊 viewer tmux 已不存在。

## 2. Phase 1（P0）— 先修切割物理，不啟動機器人

只用 kinematic backend／量測軌跡重播，正式解析度（1 mm 粒子、0.1 ms、4 mm grid），GPU。

1. 依 BladeMaster Section III-C 分離兩種狀態：
   - **Cutting progress \(D_p\)**：由刀具接觸與方向性切削功累積，只控制切削阻力與刃口何時克服材料。
   - **Persistent side code \(c_p\)**：由**有限刀刃在時間區間內實際掃過的幾何區域**（含刃口前後、相對 reference grid 尺寸的切割帶寬度）指派，不要求每顆粒子當下已累積足夠切削功，但必須限制在實際有限刀刃掃掠區內。
   - 要和「普通工具碰撞」分開：只有「刃口實際沿刃面方向穿過材料、且有接觸」的掃掠才建立 side code。側推、刀面平壓、停止、後撤、未接觸材料都不建立分離。
2. 以新的 law 版本實作（例如 `cutting_law=4` 或新模組 `tlmpm_v4.py`）；**保留 law 1/2/3 與所有失敗結果**，不改 `tlmpm.py`、`tlmpm_v2.py`、v1/v2 任務門檻與紀錄。
3. 驗證不能只看 component 數：量測每個 side label 到實際 swept surface 的距離分佈，避免「刀沒碰到卻分離」；切口位置用凍結 oracle 的 `cut_at_workpiece_local_target_plane`。
4. 1 mm 最低測試矩陣（全部 PASS 才進 Phase 3）：

| 測試 | 預期 |
| --- | --- |
| 垂直完整下切 | 兩塊，切口位置符合凍結門檻 |
| 約 20° 傾斜下切 | 沿實際傾斜路徑形成切面 |
| 水平／斜刃削除凸起（TRIM-01-R） | 凸起 ≥ 90% 機械分離，底材為完整主體 |
| 切到一半拔刀 | 下方仍連通 |
| 刀面側推、平壓 | 有接觸力，不切斷 |
| 靜止、後撤 | 不增加分離 |
| 完整切斷後撤刀 | 分離持續 |
| 同物體連續兩刀 | 三個機械獨立部分 |

5. 用量測軌跡重播確認 Native 已成功的 CUT-01-R／CUT-02 在新 law 下仍 PASS；Lab 重播也檢查。

## 3. Phase 2（P0）— 統一 Workcell

同一個 Workcell，只換材料幾何、支撐與 Task Goal：Franka 0（wrist D405＋物理夾持尺狀刀）、Franka 1（park＋wrist D405）、
Global D455（現有 close view，可加純展示 overview camera）、**有實際 collider 與摩擦的切割板**（視覺幾何與物理 collider 對齊，不能有不一致的隱形 fixture）、TLMPM 軟材料、桌面。
材料：Demo 1 用 CUT-01-R（160×30×20 mm）或 CUT-02；Demo 2 用 TRIM-01-R（同底材＋8 mm 連續凸起；`ruler_workcell_v2` 只是候選，35° 不穩就找其他經實測的安全姿態）；
Demo 3 用軟長條或 capsule footprint（只用一套暫定軟材料）。新增設定一律版本化（新 scenario／任務 config），舊的不改。

## 4. Phase 3（P0）— Native 三個 Live Scripted Demo

Live Isaac Sim／PhysX＋physical grasp＋scripted controller（先不做 BC／PPO／LLM）。

- Demo 1：Reset → Approach → Cut（先 CUT-01-R 標準，再加一個約 20° 斜切）→ Withdraw → Evaluate（兩部分可獨立運動、質量保留、切口位置合理、沒掉刀、撤刀後不重新連通）。
- Demo 2：Reset → Approach（保持手指間隙）→ Skim → Withdraw → Evaluate。首要成功條件：凸起 ≥ 90% 機械分離、底材完整、無非預期大範圍切除、chip 可碰撞滑動但不重新黏合。原 0.5 mm 表面精度保留為**獨立的 precision score**，不作唯一判定（不修改凍結門檻，只是分開報告）。
- Demo 3（新任務，例如 `SLICE-PUSH-01`，允許獨立 duration，不改凍結 v1 benchmark）：第一刀 `cut_index=0` → 抬刀移位不新增標記 → 第二刀 `cut_index=1` → ≥ 3 個機械獨立部分 → 刀身鈍面 `cut_enabled=False` 推開其中一塊（關閉切削不等於關閉接觸；不得直接改粒子速度或 component pose）→ 驗證被推那塊移動、其他塊沒被隱藏連通性拖走。需要擴充 multi-cut evaluation 與切後操控評估。

## 5. Phase 4／5 — 視覺與可重複 Demo 環境

- 每個 Demo 產出：Global view 影片（Franka、刀、切割板、材料）、particle debug view（依分離狀態上色）、reconstructed surface view、wrist D405 RGB-D、物理診斷（切削力、手指接觸、刀具 pose、分離歷史）。影片標示控制方式、時間、切削開關與評估結果，放 `docs/media/demos/`。**完成後把三個 Demo 的影片／GIF 與新的場景影像放進 GitHub main 的 README**（使用者要求），並註明來源 episode、runtime 與評估結果。
- 若現有 component ParticleSurface 足以看出分離就沿用；Demo 3 若出現切面填補或視覺黏合，再參考 BladeMaster 的 linked-volume／local dual contouring 做 cut-aware surface。
- 介面：`open_cutting_env(task_id="SLICE-PUSH-01", runtime="isaac_sim", tool_attachment_mode="physical_grasp", tool_init_mode="pregrasped")`、`reset`／`step`／`evaluate(profile="research")`；每集輸出 action、50 Hz robot state、同步影像（照實記錄實際影像頻率）、刀具狀態、物理診斷與 success metrics。一個 demo runner 指令可重跑三個 Demo。

## 6. 優先級

| 工作包 | 優先級 | 完成條件 |
| --- | --- | --- |
| A. Cutting law & separation | P0 | 1 mm 矩陣（第 2 節）全部通過，負對照通過 |
| B. Shared workcell | P0 | 物理切割板、尺狀刀、夾持與相機一致 |
| C. Native three demos | P0 | 三個 live scripted demo 完成，有原始紀錄與影片 |
| D. Dataset & demo runner | P0 | Reset／Action／Record／Evaluate 可反覆執行 |
| E. Isaac Lab parity | P1 | Lab 夾持（TRIM pregrasp 角速度）與切割差異 |
| F. Cut-aware surface | P1 | 需要時改善切面視覺 |
| G. BC／Diffusion／PPO／LLM／RSI | P2 | 場景穩定後再做（原 WP-10 E/F/G） |
| H. 多材料校準、大規模 RL 加速 | P2 | 之後研究 |

只用一把尺狀刀、一套軟材料、一台主動 Franka 完成三個 Demo。

## 7. 規則

- `AGENTS.md` 全部適用：不改 v1 物理／門檻／紀錄；新東西版本化且 opt-in；PASS／FAIL／BLOCKED／NOT_RUN 分開；保留失敗；不覆寫輸出；記錄指令、commit、seed、版本、wall time、artifact 路徑；不改驅動、RSI 或其他專案。
- 使用者已明確允許：在本分支隨時 commit 並 push；全部完成並審查後合併 main 並 push；整理 GitHub（README、PROGRESS、報告、場景影像）。
- GPU：先 `nvidia-smi`；不要停止別人的行程；兩張 5090 都可用，優先 GPU 1，避開正在跑的他人工作。
- Phase 1 沒通過，不開始完整 live robot demo 的反覆嘗試（單次 smoke 用來取得夾持／IK 診斷可以）。
- 不為了展示成功而修改物理結果或評分。
