# WP-09 — Generalized Cutting & Surface Evaluation Framework

Repository: https://github.com/AnsonNing/RoboticCuttingSim

## 0. 核心目標

請檢查 RoboticCuttingSim 最新 main，改善現有 Surface Evaluation 與 Task Evaluation，建立可支援不同物體、不同切割方式及不同研究目標的通用評估系統。

這次的重點不是單純調整 TRIM-01 的成功門檻，而是改善整體評估架構。

**RoboticCuttingSim 的主要目的，是建立一個能讓機器人切割不同物體、模擬物理反應、收集 Demonstrations、執行 Policy Inference，以及進行自主研究的模擬環境。**

Surface Evaluation 應該是研究工具，而非限制場景使用的硬性條件。

### 核心要求

1. 保留真正的物理切削與材料分離驗證。
2. 不要把任務成功與否全部綁定在 0.5 mm 的平面整平誤差。
3. 不要讓表面重建誤差被誤判為刀具真的切壞材料。
4. 支援不同形狀、尺寸、位置、方向的物體，以及不同切割目標。
5. 評估系統必須能提供連續的品質指標，而不只是 PASS/FAIL。
6. 使用者與 Policy 應該可以自由執行切割，即使結果不符合某個精度標準。
7. 保留既有嚴格 Benchmark，確保歷史研究結果可重現。
8. 不要為了讓舊測試通過，而偷偷修改物理求解器、切割結果或原始門檻。
9. 避免過度工程化；優先建立實際可用、可擴充的最小架構。

## 1. 先分析現有實作與問題

重點閱讀：

- `src/robotic_cutting_sim/tasks.py`
- `src/robotic_cutting_sim/scene_validation.py`
- `src/robotic_cutting_sim/cutting_surface.py`
- `src/robotic_cutting_sim/live_env.py`
- `src/robotic_cutting_sim/benchmark.py`
- `configs/cutting_tasks_v1.json`
- `configs/represented_flatbar_target_v1.json`
- `docs/REPRESENTED_TARGET_VALIDATION.md`
- `docs/INITIAL_FLATBAR_GEOMETRY_CONTROL.md`
- `docs/SURFACE_GEOMETRY_VALIDATION.md`
- `docs/reports/CUTTING_TASK_EPISODES.md`

目前已知的問題：

### A. 平坦物體也可能無法通過評估

已驗證的 2 mm 粒子、未切削平坦材料，其重建表面相對理想平面的 P95 誤差約 1.526 mm。

這代表即使沒有進行切割，Surface Reconstruction 自身也可能讓 0.5 mm 的嚴格條件失敗。

請分析如何區分：

- 真正的材料變形與切削誤差。
- 粒子離散化誤差。
- Surface Reconstruction 誤差。
- 邊界和幾何交點缺失。
- 刀具過度切削造成的真實材料損失。

### B. Missing Surface 不應直接等於 Gouging

現有 `mesh_heightfield_metrics()` 會將沒有表面交點的格點視為高度零。

這可能讓單一 Missing Ray 產生 20 mm 的最大凹陷評分。

請重新定義：

- Actual measured gouge。
- Missing surface / missing intersection。
- Boundary reconstruction artifact。
- Unsupported or unknown geometry region。
- True material deficit。

不可以單純刪除缺失格點、裁切異常值，或把錯誤改成零來取得較好的分數。

應保留全部原始資料，提供分開的數值、比例、位置與原因。

如果某項指標無法可靠測量，回傳 `NOT_MEASURABLE` 或相同語義的明確狀態，而不是假裝通過。

### C. 評估目標過度依賴平坦長條材料

目前 CUT-01 的目標是完整切斷，而 TRIM-01 的目標是將凸起表面整平至固定 Z 高度。

這兩種任務可以保留，但不能讓未來所有物體都必須使用 Heightfield 或固定平面。

請找出目前評估程式中與特定物體形狀、理想平面、固定尺寸或單一切割目標耦合的地方。

## 2. 將 Evaluation 分成三個獨立層次

### Layer A — Physical Integrity

評估模擬的物理結果是否可信。

例如：

- 模擬有沒有正常完成。
- Particle Mass 是否守恆。
- 是否發生真實的材料分離。
- 分離後的 Component 是否持續存在。
- 是否有不合法的 Particle State。
- 是否出現 NaN、Inf 或明顯數值不穩定。
- 切割結果是否由實際物理狀態產生，而非單純刪除 Render Mesh。
- Contact、Deformation、Cutting History 是否具有可追溯來源。

這一層屬於物理研究及資料品質驗證。

不要把「必須至少切成兩塊」當成所有任務共用的強制條件，因為某些操作可能只需要壓入、刻痕、局部刮削，或者本來就是失敗示範資料。

請將物理有效性、操作是否達到目標、資料能否用於研究分開報告。

### Layer B — Cutting Outcome

評估機器人對物體實際造成的切割效果。

根據任務目標選擇適用指標，例如：

**Complete Cutting**

- 是否形成真正分離的物體。
- 分離部分數量。
- 每個部分的質量。
- 切割位置誤差。
- 分離後的持續性。
- 是否切到不應切的位置。

**Surface Trimming**

- 目標材料移除比例。
- 剩餘凸起高度。
- 多凸起物各自的移除比例。
- 過度切削體積。
- 最終表面形狀與目標形狀的差異。

**General Cutting**

- 目標區域的實際移除量。
- 材料形狀改變程度。
- 切割區域與目標區域的差異。
- 切割後物體的幾何與連通狀態。
- 不必要的材料損失。
- 未完成切削的區域。

這些都應該提供連續 Metrics。

不是每個 Task 都必須計算全部指標，也不是每個切割任務都要求物體完全分離。

### Layer C — Surface / Geometric Quality

獨立評估切割後的幾何品質。

根據物體與目標幾何，使用適合的方法：

- Heightfield Difference。
- Surface-to-Surface Distance。
- Symmetric 3D Surface Distance。
- Area-weighted P95 / RMS Distance。
- Volume Difference。
- Remaining Material / Overcut Volume。
- Target Region Coverage。
- Missing Geometry Coverage。

對於平面整平，可以繼續使用高度誤差。

但對於曲面、不規則物體或非平面目標，不應強制要求使用 Z-heightfield。

請使幾何評估與目標表示方式解耦，例如可以評估：

- Target Plane。
- Target Heightfield。
- Target Mesh。
- Target SDF / Implicit Surface。

不要求一次實作所有格式的完整物理切削功能，但新的評估架構必須不依賴固定平面。

## 3. 設計可設定的 Task Goal 與 Evaluation Profile

不要將所有判斷寫死在 `TRIM-01` 或 `CUT-01` 裡。

建議用輕量、可配置的 Goal Specification，指定：

- Task Goal Type。
- Target Geometry。
- Evaluation Region。
- Applicable Metrics。
- Optional Thresholds。
- Metric Units。
- Target Coordinate Frame。
- Evaluation Protocol Version。

Goal Type 可以涵蓋：

- `sever`
- `trim_surface`
- `remove_region`
- `shape_to_target`
- `diagnostic_only`

這是建議的語義，請依現有 Task API 做最小合理整合，不必為了這幾個名稱建立大型新框架。

### Evaluation Profiles

建議支援三種不同用途：

**Diagnostic**

只輸出實際測量結果與品質資訊，不強制將不符合某個精度目標的 Episode 判定為無法使用。

適合：

- 建立新切割場景。
- 測試不同物體。
- 收集 Demonstrations。
- Debugging。
- 測試新 Policy。

**Research**

根據該任務選擇適用的完成條件與品質指標。

適合：

- 比較不同切削策略。
- Policy Evaluation。
- RSI 自主研究。
- 大量隨機化場景測試。

門檻可以透過設定指定，不必只有唯一的硬性精度標準。

**Precision**

保留高精度目標。

例如 TRIM-01 原始：

- P95 ≤0.5 mm。
- RMS ≤0.5 mm。
- 正向高度殘留 ≤1 mm。
- Gouge ≤1 mm。
- Overcut ≤1%。

這些可以保留，但不應是所有任務的唯一成功定義。

### Status 必須清楚

請明確區分：

- Episode 是否完成。
- 物理執行是否有效。
- 任務目標是否達成。
- Surface Quality 是否符合所選 Profile。
- 指標是否可測量。

不能因為 `episode_completed=true` 就宣稱切割成功，也不能因為 Precision 失敗就宣稱整個模擬場景不可用。

## 4. 改善 Surface Evaluation 的參考基準

目前 TRIM-01 使用理想幾何平面作為目標，但實際表面由粒子重建。

請考慮同時提供兩類比較：

### A. Analytic Target

對理想幾何進行比較。

這適合表達最終工程精度要求，必須保留。

### B. Represented Target

使用相同粒子解析度、Surface Reconstruction 及座標系統，建立獨立且未經切割的目標表面。

這適合辨別切削誤差與表示誤差。

要求：

- Target 不能根據切削後結果進行擬合。
- 不能使用最終結果反向修正 Target。
- 不能隱藏 Missing Regions。
- 不能只報告比較好看的指標。
- 兩類比較必須有不同的名稱、來源與版本。

另外，可以保留真正的 3D Surface Distance 作為參考，避免在不適合使用 Heightfield 的物體上誤用高度誤差。

## 5. Evaluation Region 不應固定為整塊矩形

對不同的切割任務，應能定義：

- Entire Object。
- Cutting Region。
- Target Removal Region。
- Surface of Interest。
- Protected Region。
- Boundary Region。

例如 TRIM-01 可以同時提供：

- 整塊材料的整體表面品質。
- 凸起物所在區域的切削品質。
- 未切削區域的意外變形。
- 邊界重建誤差。

但不要為了提高分數，事後才把較差區域排除。

ROI 應來自 Task Geometry / Goal Definition，並在評估前明確定義。

所有 ROI 結果應保留面積、覆蓋率與原始資料來源。

## 6. 保持與物理模擬器解耦

Surface Evaluation 不應決定：

- 哪個物體可以加入場景。
- TLMPM 是否允許切割。
- 刀具能不能接觸材料。
- Policy 是否可以繼續執行動作。
- 是否允許儲存失敗示範。

評估器應該消費模擬的記錄狀態，輸出 Metrics 與 Results。

不應為了評估方便而修改：

- Particle Positions。
- Particle Mass。
- Cutting History。
- Contact Forces。
- Robot Actions。
- Dynamic Surface Mesh。
- Physical Separation。

也不要依賴模擬器未來時間的 Component Labels 去推論較早影格的物理分離。

## 7. 支援未來任意 Cuttable Object

這次主要工作是 Evaluation，不要求重新實作所有材料的切削物理。

但未來我希望 RoboticCuttingSim 能建立更多場景，例如：

- 不同尺寸的長條材料。
- 多個不規則凸起。
- 自由形狀的軟材料。
- 有曲面的材料。
- 需要局部切除的物體。
- 不同材料特性的物體。
- 具有不同切割目標的場景。

請讓評估架構可以接收不同 Workpiece Geometry 與 Goal Representation，而非僅支援一個固定的 160 × 30 mm 平面。

若目前只有 Heightfield 類別的幾何能真正執行 TLMPM Cutting，請保留這項物理限制並明確記錄。

Evaluation 的可擴充性不代表目前已經能真實切割所有材料。

## 8. 對 Policy 與 Dataset Collection 的支援

未來每個 Episode 應能保存：

- Task / Scene Configuration。
- Random Seed。
- Initial Workpiece Geometry。
- Target Goal。
- Actual Cutting Trajectory。
- Observation / Action。
- Simulation Time。
- Physical Integrity Metrics。
- Cutting Outcome Metrics。
- Surface Quality Metrics。
- Evaluation Profile / Version。
- Completion / Success / Failure Reason。

保留原始數值與單位，不要只有成功率。

這樣未來可以分析：

- 哪個 Policy 移除最多目標材料。
- 哪個 Policy 過切最少。
- 哪個 Policy 切割路徑較有效。
- 哪些物體形狀最難切。
- 哪種材料或切割角度造成更多失敗。
- 某一版 Policy 是否優於 Baseline。

失敗 Episode 也應可保存與匯出。

這次不必建立完整的 Policy Training Pipeline，但請保留與現有 `reset/step/observe/evaluate` API 的相容性。

## 9. Validation 與回歸測試

請至少驗證以下情況：

### A. 未切削的平坦物體

確認評估器可以分辨：

- Analytic Geometry Error。
- Reconstruction Error。
- Missing Ray。
- Actual Gouge。

不能將無交點直接錯誤描述成真實刀具造成的 20 mm 凹陷。

### B. 原始 TRIM-01

使用既有的固定 2 mm 切削錄製資料重新評估。

保留原本約 98.53% 的材料移除結果。

對比原始 Strict Score、新 Diagnostic Metrics、新 Research Profile。

### C. Randomized Multi-Bump TRIM-01

使用現有 Seed 42 三凸起實驗，確認可以：

- 量測每個凸起的切除效果。
- 量測整體移除比例。
- 量測最終 Surface Quality。
- 保留實際材料分離的物理證據。

### D. CUT-01

確認 CUT-01 不會被不適用的平面整平條件判定失敗。

保留原本的材料分離、質量與切割位置評估。

### E. 不同形狀的 Synthetic Geometry

至少建立一個非平坦目標或曲面，以及對應的幾何評估測試。

確保新架構不要求每個物體都具有固定 Z 平面。

### F. Negative Controls

至少涵蓋：

- 完全未切除。
- 局部切除。
- 明顯過切。
- Surface Mesh 缺失。
- 邊界缺口。
- Target 與實際結果座標不一致。
- 物理資料來源不完整。

避免任何評估方式因缺少資料而錯誤回傳 PASS。

## 10. 研究與工程執行原則

請自主分析、實作、測試並迭代，不需要每個細節都等待我決定。

優先追求：

1. 評估結果具有物理意義。
2. 各種錯誤來源可以分辨。
3. 不限制機器人探索不同切割任務。
4. 能支援新物體與新目標。
5. 能被 Policy / Agent 用來比較研究結果。
6. 保持程式架構簡單、可維護。

不要為了增加功能而建立過度抽象的框架。

也不要將初步候選門檻直接寫成全專案唯一的成功標準。

若需要新的 Research Threshold，請先提供理由、測試數據與選擇依據，再獨立版本化。

### Baseline Preservation

- 不修改原始 `cutting_tasks_v1.json` 的歷史門檻。
- 不覆蓋原始驗證結果。
- 不將新的分數偽裝成舊版 Benchmark PASS。
- 不為了通過評估而修改物理結果。
- 所有新增結果提供 Protocol Version、Source Hash 和可重現命令。

如有必要，建立 `Evaluation v2` 或其他清楚命名的新協定。

## 11. 最終交付

請完成：

1. 現有 Surface Evaluation 問題診斷。
2. 可運作的通用 Evaluation Core。
3. 改善後的 TRIM Surface Evaluation。
4. CUT/TRIM 共用的 Evaluation Interface。
5. 可選擇的 Diagnostic / Research / Precision Profile。
6. 不同幾何 Target 的擴充能力。
7. 連續 Metrics 與結構化輸出。
8. 回歸測試、Negative Controls 與實際錄製資料重新評估。
9. README、API 和 Evaluation 文件。
10. GitHub Commit、Push 及完成報告。

如果 GPU 環境可用，請在保留原始實驗的前提下，執行必要的實際 Isaac Sim / Isaac Lab 端到端測試。

不需要為了這次工作重新訓練 Policy，也不需要改動 RSI。

## 最終成功定義

這次工作成功，不是因為 TRIM-01 從 FAIL 變成 PASS。

真正的成功定義是：

**RoboticCuttingSim 可以讓研究人員建立不同的切割場景，自由執行切割任務，並獨立、合理地量測材料切除、幾何變化、表面品質和物理有效性，而不受單一平面整平精度標準限制。**

對現有 CUT-01 / TRIM-01，評估結果要能明確指出：

- 物理模擬是否有效。
- 切割造成了什麼變化。
- 目標完成到什麼程度。
- 哪些誤差是真實物理結果。
- 哪些誤差來自 Surface Reconstruction。
- 哪些項目仍無法可靠測量。

這套系統應該成為未來不同物體、不同刀具、不同 Policy，以及 RSI Autonomous Research 共用的 Evaluation Foundation。
