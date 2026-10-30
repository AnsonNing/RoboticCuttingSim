# BladeMaster 風格三個 Demo：完成報告（2026-10-29）

依使用者 2026-10-29 的 [計畫](../BLADEMASTER_DEMO_PLAN.md) 完成 P0 工作包 A–D：1 mm 切割法則、統一 workcell、Live Isaac Sim（Native）三個 Demo，以及錄影、評估與可重跑的 demo runner。分支 `claude/lucid-mendel-vt7n82`。

## 閘門重做與最終組合（law 8）

計畫 §2 要求 1 mm 矩陣（含負對照）**全部** PASS 才能進入 live。

- **Law 5 沒有滿足閘門**：依計算式準則 M8 FAIL，卻仍在審查的「有條件接受」下先跑了 live。那批結果保留在下方，作為先前證據，不算通過閘門的驗收 run。
- **為了通過閘門，依序宣告並測試了 law 6、7、8**：

  | Law | ADR | 結果 |
  | --- | --- | --- |
  | 6 | [0010](../adr/0010-cutting-law-6-deferred-lead-commit.md)，延後寫入 | TRIM 35° FAIL |
  | 7 | [0011](../adr/0011-cutting-law-7-eq8-coupled-flip.md)，Eq. (8) 耦合 pⁿ | M8b、M3 FAIL |
  | 8 | [0012](../adr/0012-cutting-law-8-plow-gated-lead.md)，只在偏離刃面 ≥ 15° 的犁切刃段使用 law 5 的前緣規則 | M1–M9 全部 PASS |

  - law 4、6、7 的失敗都保留。
  - Law 8 的計算式結果：[matrix_law8_001](../evidence/phase1/matrix_law8_001.json) 中 M1–M9 全部 PASS。
  - M8a 是預先宣告的診斷列，FAIL 照報。
  - M2 只在 ADR 0006 Amendment 1 下 PASS，原本的 2 mm 門檻是 FAIL。
  - M9c/d 是 Lab，只報告；Lab 切口位置 4 mm，FAIL。
- **實際順序**：
  1. 20:54:04 commit `3e5055c`：matrix_law8_001 全部 PASS，滿足 ADR 0012「任何 law 8 live episode 計入前，每一列都必須 PASS」的規則。
  2. 20:54:06 開始 live 最終組合：`--demo-set final_v4`（law 8），run `live_20261029_law8_final`，每次都在乾淨的 commit 上執行。
  3. 審查（rcs-review PASS，附條件）在 1a、1b 跑的期間完成，review note 21:08 commit。
  4. 依審查條件，M1a、M3a、M3b、M8b、M7 在乾淨工作樹重跑（21:07–21:18）。結果 [matrix_law8_002](../evidence/phase1/matrix_law8_002.json) 21:19 commit，此時 1a、1b、2 已跑完。

  matrix_law8_002 確認了 matrix_law8_001 的每一列 verdict，沒有任何 demo verdict 因此改變。結果如下：

| Demo | 結果 | 主要數據 | 影片 |
| --- | --- | --- | --- |
| 1a 完整切斷（`path_v2`） | **PASS**，凍結 strict 也 PASS | 兩塊 50/50%；切口位置 1.5 mm；撤刀後持續分離 3.0 s；刃口 x 誤差 ≤ 0.046 mm | [GIF](../media/demos/demo1a/live_20261029_law8_final_demo1a.gif)、[MP4](../media/demos/demo1a/live_20261029_law8_final_demo1a.mp4) |
| 1b 20° 斜切（`path_v2`） | **PASS**（swept-plane） | 50.5/49.5%；界面統計 4.78 mm，允許值 ≤ 4.44 + 1 mm；凍結 strict 的切口位置與持續時間窗口兩項 FAIL（理由同下） | [GIF](../media/demos/demo1b/live_20261029_law8_final_demo1b.gif)、[MP4](../media/demos/demo1b/live_20261029_law8_final_demo1b.mp4) |
| 2 削除凸起（`path_v2`） | **PASS**（分離） | 去除 98.0%；底材 99.99%；底材脫落 0.008%；刃口 z 誤差 ≤ 0.026 mm；精度 5/8 FAIL（p95 0.58 mm、overcut 1.66%、保守 gouge 規則） | [GIF](../media/demos/demo2/live_20261029_law8_final_demo2.gif)、[MP4](../media/demos/demo2/live_20261029_law8_final_demo2.mp4) |
| 3 兩刀三塊＋推開（`path_v2_vision`） | **PASS**（含凍結 v3 slice-push） | 三塊 37.5/31.25/31.25%；兩刀宣告切面誤差 0 層；推開 19.7 mm；手腕 D405 量得長條位移 1.20 mm | [GIF](../media/demos/demo3/live_20261029_law8_final_demo3.gif)、[MP4](../media/demos/demo3/live_20261029_law8_final_demo3.mp4) |

這組的共同條件：

- **執行**：四個 episode 全部 400／800 步完整執行，IK 拒絕 0 次，夾持全程 HOLDING。
- **切削**：未啟用切削的階段標籤變化為 0。
- **時間**：physics wall time 分別為 350、440、352、528 s。
- **證據**：所有嘗試見 [demos_live_004.json](../evidence/phase3/demos_live_004.json)，每個報告數字的來源都在 repo 內，並附上上述時間順序。

## 先前組合：law 5（閘門滿足前，保留作證據）

所有 episode 的共同條件：

- **執行環境**：Live Native Isaac Sim（PhysX/Kit/RTX），Franka 0 以兩指實體夾持 `ruler_knife_v1`，Franka 1 在一般停放姿勢。
- **相機與場景**：手腕 D405 rig v5，scenario `blademaster_workcell_v1`。
- **切割與材料**：cutting law 5（`tlmpm_v4`），1 mm 粒子，0.1 ms，4 mm grid，seed 42。（最終組合改用 law 8，見上節。）
- **紀錄**：動作與機器人狀態 50 Hz；影像實際錄製 10 Hz（RGB-D 每 5 步一張）。

| Demo | 任務 | 控制器 | 結果 | 主要數據 | 影片／證據 |
| --- | --- | --- | --- | --- | --- |
| 1a 完整切斷 | CUT-01-R | scripted `path_v2` | **PASS**（凍結 strict 也 PASS） | 兩塊 50/50%；撤刀後持續分離 3.0 s；切口位置 1.5 mm（門檻 2 mm）；量測刃口 x 誤差 ≤ 0.046 mm | [GIF](../media/demos/demo1a/live_20261029_pathv2_d_demo1a.gif)、[MP4](../media/demos/demo1a/live_20261029_pathv2_d_demo1a.mp4)、[摘要](../media/demos/demo1a/live_20261029_pathv2_d_demo1a.json) |
| 1b 20° 斜切 | CUT-01-R `--motion oblique --angle 20` | scripted `path_v2` | **PASS**（swept-plane 準則） | 兩塊 50.5/49.5%；界面統計 4.91 mm，允許值 ≤ 理想 4.44 + 1 mm；標籤與幾何不一致 ≤ 1.07 mm。凍結 strict 有兩項 FAIL：切口位置（垂直面檢查不適用於斜切），以及切後持續時間窗口（path_v2 較慢，約 5.7 s 才切斷，晚於凍結 v1 主動切削窗口的結束時間 5.6 s；撤刀後持續分離 1.82 s，未重新連通） | [GIF](../media/demos/demo1b/live_20261029_pathv2_d_demo1b.gif)、[MP4](../media/demos/demo1b/live_20261029_pathv2_d_demo1b.mp4) |
| 2 削除凸起 | TRIM-01-R，35° 前傾 | scripted `path_v2` | **PASS**（分離） | 凸起去除 97.9%，底材 99.99% 為主體，底材脫落 0.008%；削切中刃口 z 誤差 ≤ 0.026 mm。精度另計 5/8（FAIL） | [GIF](../media/demos/demo2/live_20261029_pathv2_c_demo2.gif)、[MP4](../media/demos/demo2/live_20261029_pathv2_c_demo2.mp4) |
| 3 兩刀三塊＋推開 | SLICE-PUSH-01（v3） | scripted `path_v2_vision` | **PASS**（含凍結 v3 slice-push 報告） | 三塊 37.5/31.25/31.25%；兩刀宣告切面誤差 0 層；刀身鈍面（切削關閉）推開右塊 18.7 mm，其餘不動，推動中無新標籤、未合併；手腕 D405 量得長條位移 2.17 mm 並據此修正第二刀 | [GIF](../media/demos/demo3/live_20261029_vision_e_demo3.gif)、[MP4](../media/demos/demo3/live_20261029_vision_e_demo3.mp4) |

這四個 law 5 episode 的所有嘗試（含失敗）見 [demos_live_002.json](../evidence/phase3/demos_live_002.json)，新增執行當時的 git 狀態；001 保留未改。GIF 為 2 倍速，MP4 以實際錄製頻率即時播放；原始 episode 在 `artifacts/wp10_claude/demos/<run_id>/`（不進 Git）。這四個 law 5 episode 都是 400／800 步完整執行，IK 拒絕 0 次，夾持全程 HOLDING，最大滑移 < 1 µm。Physics wall time：1a 355 s、1b 401 s、2 348 s、3 552 s。

## Phase 1：切割法則（工作包 A；本節是 law 4／5 的紀錄，最終的 law 6–8 見上方「閘門重做」）

- **Law 4**（[ADR 0006](../adr/0006-swept-band-cutting-law-4.md)）：依 BladeMaster Section III-C 拆成兩個狀態。
  - Cutting progress \(D_p\) 只影響方向性接觸阻力。
  - Side code 由「整把刀的切削行程」閘控：必須有方向性切削功，且刃口在刃面 45° 內前進或拉切；無功時最多再持續 2Δx 行程。
  - 行程內依 Eq. (4)/(6) 的有限掃掠帶指派 side code，以 swept surface 分側。
  - 側推、平壓、停止、後撤、刀柄、未接觸材料都不會建立 side code。
- **Law 4 的失敗**：1 mm CUT-01-R 通過，但 TRIM-01-R 仍 0%。原因是兩顆被前傾刃推到前方的 chip 粒子維持 code 0，經 34 顆被壓到切面下的凸起底層粒子形成橋接。
- **Law 5**（[ADR 0008](../adr/0008-cutting-law-5-lead-preassignment.md)）：在刃口前方 2Δx 內、離 swept surface 超過一格的材料，提前依其所在側指派；核心帶 |s| ≤ Δx 仍要等刃口通過才標籤。
- **1 mm 矩陣**（[runner 原始結果](../evidence/phase1/matrix_law5_001.json)、[計算式準則 `adr0006a1_adr0008`](../evidence/phase1/matrix_law5_002.json)、[審查與裁定](../evidence/phase1/matrix_law5_adjudication_001.json)）：
  - M1 CUT-01-R／CUT-02 strict PASS。
  - M2 20° 斜切 PASS（依 ADR 0006 Amendment 1；原 2 mm 門檻連理想幾何標籤都只能做到 4.44 mm）。
  - M3 TRIM 25°／35°／切割板全部 PASS（≥ 90% 去除、底材 ≥ 99%）。
  - M4／M6 切到一半拔刀、停住：仍連通，核心帶 100% code 0。
  - M5 側推、平壓、刀在切縫內側移：有接觸力，0 個新標籤。
  - M7 切斷後撤刀：持續分離。
  - M9 Native 量測軌跡重播 PASS。
  - **M8 FAIL**：理想追蹤下 SLICE-PUSH-01 的第二刀宣告切面誤差 2.0000000670 mm，凍結界限 2 mm。原因是第一刀把剩下的長條推開約 2.3 mm（law 4 約 1.2 mm）。M8 的拓撲條件（三塊、移位不加標籤）成立。Live Demo 3 改用 ADR 0009 的手腕深度量測修正第二刀，凍結檢查因此通過。
- **審查（law 5，已被取代）**：rcs-review（xhigh）判定 ACCEPT WITH CONDITIONS，認為 Demo 1／2 可進行，Demo 3 需要預先宣告新的設計（ADR 0009）。之後確認計畫 §2 的閘門並未滿足，改由 law 8 完成閘門，見上方。
- **保留下來的失敗**：
  - Law 2／3 的歷史結果。
  - Law 4 的全部矩陣列（[matrix_law4_001](../evidence/phase1/matrix_law4_001.json)）。
  - 中斷的輸出目錄。
  - 2 mm CPU 35° 削除（strict xfail）。

## Phase 2：統一 workcell（工作包 B）

- **切割板**：`blademaster_workcell_v1`（[ADR 0007](../adr/0007-blademaster-workcell-and-demo-tasks.md)）沿用 `ruler_workcell_v2` 的 +50 mm 擺放與 35° TRIM 前傾。每個支撐改成 10 mm 木紋切割板加灰色 riser。
  - 同一組 box 同時用於 PhysX `CollisionAPI`＋`MaterialAPI`（靜／動摩擦 0.3、恢復係數 0）和 TLMPM fixture contacts。0.3 也是 TLMPM 材料的摩擦係數。
  - 桌面使用相同材質。
  - CUT 保留凍結的 40 mm relief 槽；SLICE-PUSH-01 用三段板與兩道 8 mm 槽。
- **新任務**：`SLICE-PUSH-01` 定義在新版 `cutting_tasks_v3.json`，有自己的 16 s duration 與 max_cuts 2，門檻在任何 episode 之前凍結。v1／v2 任務、場景與 plan 不變，並以 hash 釘住。
- **相機**：手腕 rig v5、Global D455 不變。另有兩個只作展示的相機：overview（65°，含 Franka 0）與 close-up（24°，約 0.24 mm/px）。它們不屬於任何觀測或 camera contract。
- **Native smoke**（[證據](../evidence/phase2/native_smokes_001.json)）：CUT-01-R、SLICE-PUSH-01、TRIM-01-R 三個場景都 PASS_EXECUTION，0 次 IK 拒絕，夾持 HOLDING。

## Phase 3：Native 三個 Demo（工作包 C）

第一輪（v1 腳本控制器，`live_20261029_a/b`）：

- **Demo 1a、1b**：PASS。
- **Demo 2**：FAIL，去除 64.2%。原因在控制器，不在物理：
  - 機械臂以 100 mm/s 下降，在 start 點過衝，回彈上升途中就被判定「到達」。
  - 削切階段朝 84 mm 外的終點直線前進，每步 z 修正只有約 0.01 mm。
  - 結果刃口整段偏高約 1.2 mm。
- **Demo 3**：三塊與推開都成立，但第二刀與宣告切面相距 2.0000000670 mm，超過 2 mm 界限，因此 FAIL。原因與 M8 相同：第一刀推動了長條。

修正後：

- **`path_v2` 控制器**（`7ba30c9`，opt-in）：
  - 切削時沿規劃路徑做 cross-track 修正（增益 0.5）。
  - 切前最後 10 mm 以 20 mm/s 接近，並停留 0.3 s 讓機械臂穩定。
  - 位置容差 0.3 mm。
  - 用它重跑 Demo 1a、1b、2，全部 PASS。
- **`path_v2_vision`**（[ADR 0009](../adr/0009-demo3-wrist-depth-retarget.md) 與 Amendment 1，都在計入任何 run 之前宣告）：
  - 第一刀前與第二刀前各取一張手腕 D405 深度影像，量測中段＋右段的 +x 端面。
  - 用兩次量測的差估計長條位移，再平移第二刀與之後的所有路點。
  - 只用 rendered depth、K、由機器人運動學得到的相機外參，以及模擬器量到的刀具 pose，不使用粒子、標籤或 component 狀態。
  - 注意：rendered 表面是依每個機械 component 重建的，所以深度影像並非完全獨立於切割拓撲；被量測的 +x 端面離切口很遠，不受影響。
  - 這個 profile 另外調整了幾個不切削的動作：切縫內抬刀 0.04 m/s、升刀 0.02 m/s，並在 retarget 後停留。
  - 離線驗證（v1 episode）誤差 −0.012 mm。law 5 那次 live 量得 2.17 mm，law 8 最終組合量得 1.20 mm，兩次 Demo 3 都 PASS。

審查條件 2 在 law 5 組合與 law 8 最終組合都已檢查：量測刃口在目標面 ±0.5 mm 內（兩組都是 1a ≤ 0.046 mm、2 ≤ 0.026 mm），且切削關閉階段的標籤變化為 0。

## Phase 4／5：錄影、評估與 demo runner（工作包 D）

- **錄影**：`scripts/render_demo_episode.py` 從 live episode 產生 MP4、GIF 與 poster。畫面包含 overview、wrist D405 RGB 與深度、close-up，以及依機械 component 上色的粒子 debug view。這個 debug view 標示為 oracle，不是觀測。下方是診斷曲線：刀上受力、手指接觸力、刃口高度、標籤數與 component 數。每格標示控制方式、時間、切削開關、cut_index 與最終評估。
- **Demo runner**：`scripts/run_blademaster_demos.py` 跑 run → evaluate → render → publish，不會覆寫舊輸出。重做某個 stage 時會寫成新的編號副本。
- **評估**：每個 demo 的 `demo_summary.json` 分成 execution（integrity）、outcome、quality 三層，只有前兩層決定 verdict。
- **重跑方式**：

  ```bash
  # 最終組合（1a/1b/2 用 path_v2，3 用 path_v2_vision；law 8、seed 42）
  .venv/bin/python scripts/run_blademaster_demos.py --demo-set final_v4 --gpu 1 --run-id <new_id>
  # law 5 的先前組合：--demo-set final_v1
  # 加 --publish 把媒體複製到 docs/media/demos/<demo>/（不覆寫）
  ```

  Runner 的 `--controller-profile` 預設是 v1，也就是第一輪的設定。最終組合請用 `--demo-set final_v4`，也就是 `live_20261029_law8_final` 所用的指令；`final_v1` 是 law 5 的先前組合。
  （law 5 組合）Demo 1b 那次 run 是在未 commit 的 `policies.py`（path_v2 開發中）下執行。審查把紀錄的輸入重新餵給已 commit 的控制器，所有動作都完全重現。law 8 最終組合的四個 run 都在乾淨的 commit 上執行。

## 限制與未完成（照實列出）

- **材料與標定**：材料與刀具參數都是暫定、未標定（provisional_uncalibrated）。Law 4／5／8 是 process-zone 近似，不是收斂的斷裂能模型。
- **Law 8 的驗證範圍**：
  - 只驗證了刃面內切削（0°；live 接觸中的雜訊 < 9°），以及 25°／35° 定前傾削切。
  - 15° 門檻本身沒有被測到：15–25° 之間沒有任何測試動作，0–25° 之間的任何門檻都會得到相同的 verdict；法則在 15° 處是不連續切換。
  - GPU 上的等價只成立在 GPU 非確定性範圍內（CPU 上逐位相同）。
  - Live 沒有記錄犁切閘門是否觸發；審查是用重建的軌跡估計的，垂直切削接觸中觸發 0 次。
- **Law 5 的動態影響**：前緣預指派會改變刃口前方的動態。審查指出，同時帶 digit 與 code 0 的節點，其 per-field FLIP 歷程不同；因此切後兩塊被推開約 ±2.4 mm，law 4 約 ±1.2 mm。機制尚未完全診斷。Law 8 在刃面內切削時不啟用前緣規則，所以最終 Demo 3 量得的位移是 1.20 mm；在犁切的 Demo 2，law 5 的動態仍然存在。
- **Swept normal 在拉切主導時條件不佳**：任何鋸切或拉切動作上線前，要先以新版本加保護。目前的 demo 都不含鋸切。
- **Stroke 持續只看刃口中心速度**：繞中心純轉動會讓行程一直不結束。目前的 demo 都沒有這種動作。
- **Live runtime 未逐步記錄 stroke 狀態**：kinematic 有，live 沒有。
- **Demo 2 的精度分數 5/8 FAIL**：
  - p95 平面誤差 0.58 mm，門檻 0.5 mm。
  - 最大 gouge 20 mm：保守評分器把 0.62% 缺失的表面格子當成高度 0。
  - Overcut 1.66%，門檻 1%。
  - 這些項目依計畫獨立報告，不影響分離判定。
- **Demo 1b 的凍結切口位置對斜面不適用**：照實列為 FAIL。
- **Lab 端**：Isaac Lab parity（P1）未做。Lab 量測軌跡重播會分成兩塊，但實測刃口偏 −0.54 mm，所以切口位置是 4 mm、FAIL。
- **其他未做項目**：cut-aware surface（P1）、BC／PPO／LLM／agentic control（P2）、多材料標定（P2）。

## 完整性聲明

- 凍結的 `tlmpm.py`、`tlmpm_v2.py`、`tlmpm_v3.py`，以及 v1／v2 任務門檻與紀錄都沒有修改。新的法則、任務、場景與控制器都有版本並且 opt-in。
- 沒有幾何刪除、預先分割、renderer mask，也沒有直接修改粒子速度或 component pose。
- 驗收影片全部來自 live Native。
- 所有失敗輸出都保留，沒有覆寫任何輸出。
- 每次修改準則或法則都在相應的 run 之前宣告：ADR 0006 Amendment 1、ADR 0008、ADR 0009 與 Amendment 1／2／3、ADR 0010、0011、0012。原始 runner 的 verdict 一併保留。
