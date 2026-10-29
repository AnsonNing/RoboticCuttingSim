# WP-10 Claude 接手診斷與進度（2026-10-29，暫停點）

狀態：**未完成，依使用者要求暫停**。分支 `claude/lucid-mendel-vt7n82`，未合併 main。
這份文件記錄 Claude 從 [Codex 交接](../HANDOFF_WP10_CLAUDE_20261029.md) 之後的診斷、修改與實跑結果，供使用者綜合查看後決定下一步。
新的原始輸出放在 `artifacts/wp10_claude/`（symlink 到 `/mnt/data/ningan/RoboticCuttingSim_artifacts/wp10_claude`，因為根磁碟只剩約 10 GB；不進 Git）。

## 1. 核心結論：切不開的原因是「標籤橋接」，不是機器人沒切到

任務目標是 Franka 夾刀把物體切斷（CUT）或切掉凸起（TRIM）。交接時：

- Native CUT-01-R／CUT-02 strict PASS；Lab 兩個 CUT strict FAIL（仍是一塊）。
- TRIM-01-R：Native 有 394 次 IK 拒絕；**連刀具完全照指令走的 kinematic 1 mm 探測也是 0% 去除**。

用新的 [`scripts/analyze_cut_bridges.py`](../../scripts/analyze_cut_bridges.py)（以實際 8 節點 MPM stencil 重算連通圖）查出：

1. TLMPM 的「切開」是 reference grid（4 mm）上的相容性：只要有一個**未標籤（code 0）**的粒子同時碰到切口兩側的節點，整塊就仍然相連。1 mm 粒子對 4 mm 格子，切口兩側各約 4 mm 內的粒子都必須被標籤。
2. **法則 2 的標籤只在「這個 substep、這個 1 mm 刃段」真的做了切削功時才寫**，在時間與空間上都是斷斷續續的：
   - TRIM：凸起側邊之外的底材頂層（切面下方）沒有被標籤，和凸起底層共用格子 → 橋接（`c_trim_kinematic_formal_001`：615 個 code-0 橋接粒子，z 17.5–19.5 mm）。
   - Lab CUT-01-R：刀刃量到的軌跡與 Native 幾乎相同（最低點 Lab −4.6 mm、Native −3.3 mm，刃面法向誤差 < 0.007），**Lab 其實切得更深**；失敗是刀刃離開底面那一刻已沒有材料可做功，最底層粒子進入標籤窗口時刃段沒有被「通電」，留下 38 個 code-0 橋接粒子（底層 z = 0.5 mm）。Native 通過屬於同一機制下的運氣差異。
3. 斜刀（25°/35°）時，法則 2 以刀面分上下側，會把切面上方一層粒子分到下側；凸起被刀面往前推（plowing）時，少數在刃口前方的粒子永遠進不了標籤窗口。

## 2. 修改（全部 opt-in；v1／法則 1／法則 2 行為不變）

| 檔案 | 內容 |
| --- | --- |
| `src/robotic_cutting_sim/tlmpm_v3.py` | 法則 3（實驗性）：以刃口掃過的曲面分側；三種格子尺度閉合：橫向（±2 格內有通電刃段時兩側都可補標）、持續（做過切削功的刃段在刀刃持續前進／拖曳時保持切削；停止、後撤、法向推、解除即清除）、前緣（刃口前方 ≤ 2 格、且自己的 stencil 已有同側標籤的 code-0 粒子）。法則 1/2 不啟動任何新 kernel。交接草稿只做負側閉合（sha256 `69bc6ff5…`），已實測不足（見第 3 節）。 |
| `src/robotic_cutting_sim/isaac_sim_runner.py`、`isaac_lab_env.py`、`scripts/run_wp10_live.py` | 新增 `cutting_law`（預設 None = 任務預設，v2 任務仍是法則 2）與 CLI `--cutting-law {1,2,3}`；只允許 v2 任務。**尚未在 live runtime 實跑。** |
| `scripts/probe_wp10_task_policy.py` | 新增 `--cutting-law`、`--steps`（預設不變）。 |
| `scripts/replay_measured_edge.py`（新） | 把 live episode **量到的**刀刃姿態與切削開關重新驅動材料，用凍結 oracle 評分。無 IK／夾持／相機／手指墊，只用來隔離切削法則；結果標 `measured_trajectory_material_replay`，不算 live episode。 |
| `scripts/analyze_cut_bridges.py`（新） | 上述橋接診斷。 |
| `src/robotic_cutting_sim/blade_probe.py` | `solver_version` 參數（用於 v3 法則 1/2 與 v2 逐位比對）。 |
| `tests/test_cutting_law_v3.py`（新） | 9 項 CPU 測試：v3 法則 1/2 與 v2 逐位相同；法則 3 對齊推切切開、側推／平壓／刀柄不切、帶電後撤不增加分離、拉切阻力 < 推切 25%、兩個 digit 切三塊、25°/35° 斜削凸起以 stencil 圖判定分離。**9 PASS（64.5 s）**。 |

## 3. 實跑紀錄（seed 42，1 mm 粒子，0.1 ms，GPU；路徑在 `artifacts/wp10_claude/`）

| 輸出 | 設定 | 結果 |
| --- | --- | --- |
| `k_trim_law3_25deg_001` | kinematic TRIM-01-R，交接草稿法則 3，25° | FAIL，去除 0%；底材側已完整標籤，剩 2 個 chip 內 code-0 + 36 個被分到下側的 chip 底層粒子構成橋 |
| `k_trim_law3_v2_001` | 同上，`ruler_workcell_v2`（35°） | FAIL，去除 0%；9 個直接橋接粒子 |
| `k_trim_law3sym_25deg_001`、`k_trim_law3sym_v2_001` | 兩側橫向閉合版 | 仍 FAIL 0%；同兩個粒子 (4.5, ±2.5, 24.5) mm 在刃口前方被推著走，從未進入窗口 |
| `r_lab_cut01r_law2_001` | Lab CUT-01-R 量測軌跡重播，法則 2 | FAIL，1 component（重現 live 失敗，證明重播可用） |
| `r_lab_cut01r_law3_001` | 同上，兩側橫向閉合版法則 3 | FAIL，1 component（底層橋接仍在） |
| `k_trim_law3b_25deg_001`、`k_trim_law3b_v2_001`、`r_lab_cut01r_law3b_001`、`r_lab_cut02_law3b_001` | 目前版法則 3（加持續與前緣閉合） | 暫停時見下方「暫停時仍在跑」 |

所有失敗輸出保留，未覆寫任何舊實驗。`tlmpm.py`、`tlmpm_v2.py`、`cutting_tasks_v1/v2.json` 未修改。

## 4. 尚未完成（依 WP-10 規格）

- C：法則 3 的 1 mm 驗證通過後，在 Native／Lab 實跑 `ruler_workcell_v2` TRIM（含 Lab pregrasp 角速度失敗診斷：舊收據沒有 settling trace，需重跑取得）、Lab CUT 重跑；README 換上新的 ruler 場景影像（使用者要求）。
- D：法則 3 GPU 負對照矩陣（含帶電後撤）、法則 1 矩陣、刀柄力量級說明。
- E：完整機器人動作／角度矩陣與可達性地圖。F：RL 速度表。G：腳本／BC／PPO／實際 LLM agent 控制、每種控制方式的實際機器人 RGB-D 影片、RSI `run(env, ctx)`。
- 最後整理 GitHub（README、PROGRESS、報告連結），合併 main 並 push（使用者已明確要求）。

## 5. 環境注意

- 根磁碟 99%（約 10 GB 可用）；新輸出一律寫到 `artifacts/wp10_claude/`，長任務用 `TMPDIR=/mnt/data/ningan/tmp`。
- 交接時恢復的 v5 viewer（tmux `rcs-wrist-v5-remote-restored`）在本次 session 中斷後 GPU 0 只剩約 1.8 GB 使用、0% 利用率；是否仍 READY 尚未確認，未主動停止或重啟。
