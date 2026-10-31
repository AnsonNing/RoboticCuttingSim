# WP-12 — Physically Consistent Cut Surfaces, BladeMaster Workcell & Generalized Cutting

**Repository:** https://github.com/AnsonNing/RoboticCuttingSim

**Starting baseline:** `main` at `fbe99bd0`（開始實作前重新確認 HEAD）

**Primary references:**
- BladeMaster: https://jango6324.github.io/blademaster/
- BladeMaster paper: https://arxiv.org/abs/2609.27342
- Dual Contouring of Hermite Data: https://people.eecs.berkeley.edu/~jrs/meshpapers/JuLosassoSchaeferWarren.pdf
- Dynamic Cutting with Elastic Snapping: https://doi.org/10.1111/cgf.70005
- NVIDIA Isaac Lab Implicit MPM: https://docs.nvidia.com/isaaclab/3.0/concepts/implicit-mpm

## 0. Project Goal

請在 RoboticCuttingSim 最新 `main` 的基礎上，完成下一階段的模擬環境升級。

這次工作的核心目標是將現有系統從「能在部分場景完成切割任務」，提升為具有**可信切割幾何、物理一致性、自然工作台配置、廣泛任務支援與可量化泛化能力**的研究環境。

未來這套環境將提供給 agentic control、Imitation Learning、Diffusion Policy、VLA、RL 等研究系統使用。

但本次工作不需要接入 agentic control，不需要訓練 Policy，也不需要開發 Agent。

請聚焦環境本身的品質。

### 最終希望具備的能力

1. Franka Panda 透過真正的物理夾持持有長直刃刀具。
2. 刀具能以不同位置、角度、方向與路徑進行真實物理切割。
3. 切割過程中的材料變形、分離與接觸具有一致的物理行為。
4. 切面跟隨實際刀刃路徑及材料運動生成，而非單純對分離粒子重新建立光滑表面。
5. 切面清楚、連續、幾何品質良好，同時保留物理產生的壓痕、局部變形、切屑和接觸結果。
6. 工作場景接近 BladeMaster：完整木製切割板直接放在桌面，而非多個灰色墊高支柱。
7. 支援不同工件、尺寸、位置、姿態、凸起形狀、數量及材料參數。
8. 支援單刀、多刀、斜切、削切、連續削除、分段切割及混合任務。
9. 可評估成功率、切割品質、切面品質、物理穩定性與失敗原因。
10. 所有研究結果可重現、可追溯，並透過穩定的 Environment API 對外使用。

**本次最重要的要求：不要為了讓影片看起來漂亮而犧牲物理正確性。**

切面應與模擬中的材料狀態一致，而不是將材料強制裁切成預期形狀。

請先自主研究、分析既有技術與失敗證據，再設計和實作。不要直接重寫整個物理引擎。

---

# Engineering Principles

整個 WP-12 必須遵守以下原則。

### A. Physics First, Geometry Consistent

切割結果必須由實際材料模擬決定。

- 不允許使用預先知道的目標切面來直接刪除材料。
- 不允許用 Mesh Mask 隱藏尚未分離的材料。
- 不允許僅為視覺效果改寫 Particle Position。
- 不允許在沒有有效切割事件時，根據刀具路徑直接生成已分離的切面。
- 不允許透過修改 evaluator 或放寬門檻製造成功。
- 不允許直接套用目標平面來強制平整切面。
- 不允許使用未記錄的特例、seed-specific patch 或 task-specific hack。

需特別區分：

1. **Physical discontinuity**：材料在局部已產生力學不連續。
2. **Global separation**：材料已分成獨立 Mechanical Components。
3. **Surface geometry**：這些物理狀態對應的可見表面。

局部切口可能已經存在，但材料底部仍然連接，因此不能要求所有可見切面都必須對應兩個完全獨立的 global components。

反過來，也不能讓顯示系統憑空產生尚未被物理求解器切開的區域。

### B. Reuse Existing Architecture

優先擴充既有架構：

- `rcs_scene_v1`
- `rcs_goal_v1`
- `ConfiguredTaskSpec`
- `ToolProfile`
- `tlmpm_v5`
- `cut_topology`
- `CuttingSurface`
- `PolicyReadyEnv`
- `goal_eval_v2a`
- Existing Native Isaac Sim runtime

保留 WP-11 已完成的能力。

不要重新打造 Scene System、Goal System、Policy API 或整個 TLMPM。

### C. Versioned Changes

新物理法則、Surface Reconstruction、Workcell Profile 與 Evaluation Protocol 應使用明確版本。

保留舊版本作為可比較 baseline。

尤其不得靜默修改：

- `final_v4`
- WP-11 frozen results
- Existing law-8 acceptance records
- Existing task identity
- Historical evaluation evidence

必要時建立新 profile、schema 或新 solver variant。

### D. Research Before Implementation

對不確定的物理問題，先提出假設並建立最小實驗。

每次重要修改都需要回答：

1. 目前失敗原因是什麼？
2. 有哪些候選方法？
3. 為什麼選擇這個方法？
4. 是否改變物理結果？
5. 相較 baseline 的改善是多少？
6. 是否產生新的 failure mode？

不要只靠影片主觀判斷。

---

# Phase 0 — Audit, Research and Frozen Baseline

## 0.1 Repository Audit

首先閱讀並確認以下檔案：

**Documentation**
- `README.md`
- `docs/PROGRESS.md`
- `docs/WP11_PLAN.md`
- `docs/SURFACE_GEOMETRY_VALIDATION.md`
- `docs/POLICY_READY_API.md`
- `docs/reports/WP11_PHASE_F_LAW8.md`
- `docs/reports/WP11_PHASE_E_ROUND1.md`
- `docs/reports/WP11_PHASE_E_ROUND2.md`
- `docs/reports/WP11_PHASE_H_PROFILING.md`
- `docs/adr/0012-cutting-law-8-plow-gated-lead.md`
- `docs/adr/0013-ruler-knife-v2-and-trim-orientation-metrics.md`
- `docs/adr/0014-scalable-cut-history-sparse-logical-fields.md`
- `docs/adr/0015-scene-goal-controller-evaluation-separation.md`

**Source**
- `src/robotic_cutting_sim/cutting_surface.py`
- `src/robotic_cutting_sim/usd_surface_bridge.py`
- `src/robotic_cutting_sim/cut_topology.py`
- `src/robotic_cutting_sim/tlmpm_v4.py`
- `src/robotic_cutting_sim/tlmpm_v5.py`
- `src/robotic_cutting_sim/scene_config.py`
- `src/robotic_cutting_sim/native_fixture.py`
- `src/robotic_cutting_sim/fixture_contacts.py`
- `src/robotic_cutting_sim/isaac_sim_runner.py`
- `src/robotic_cutting_sim/live_env.py`
- Existing Workcell, Controller, Recording and Evaluation implementations

另外查看相關 `configs/`、`scripts/`、`tests/` 與 frozen evidence。

若檔名或實作位置已有改變，以最新程式碼為準，並記錄差異。

## 0.2 Confirm Existing Capabilities

目前已知 WP-11 完成：

- `ruler_knife_v2` 物理夾持。
- TLMPM v5 packed multi-cut history。
- Law 8 cutting。
- Near-horizontal TRIM。
- Multiple slices。
- Multiple protrusion removal。
- Scene / Goal / Controller / Evaluation separation。
- Configured perception relocation。
- Policy-ready API。
- Dataset collection。
- Native Isaac Sim reliability validation。

請確認上述能力仍可運作。

不得把已存在的功能當成新功能重新實作。

## 0.3 Freeze Baseline

建立新的工作分支，例如：

`claude/wp12-physical-cut-surfaces`

保存：

- 完整 Commit SHA
- Working tree status
- Python / CUDA / Warp / Newton / Isaac Sim 版本
- GPU 與 Runtime 配置
- CI Status
- Existing test results
- Current geometry metrics
- Cutting metrics
- Runtime performance
- Existing failure cases

對關鍵場景保留 baseline：

1. Single through-cut。
2. Horizontal / near-horizontal TRIM。
3. Four-slice Test B。
4. Three-protrusion Test C。
5. SLICE-PUSH with relocation。
6. Failed oblique cuts。
7. Failed high-rake cuts。
8. Mixed-operation sequence。

每個 baseline 應保存：

- Initial and final particle states
- Cut codes
- Mechanical components
- Surface meshes
- Blade poses and swept path
- Contact diagnostics
- Screenshots / video
- Evaluation result
- Execution time
- Configuration and random seed

## 0.4 Research Review

深入研究 BladeMaster Section III-F。

尤其分析：

- Reference-grid Linked-volume Representation
- Surface and blade intersection Hermite Data
- Link disconnection
- Local connected material regions
- Local Dual Contouring
- Duplicated cut-face representatives
- Cut-side-aware surface embedding
- Motion field interpolation
- Dynamic surface update

比較以下技術方向：

**Approach A — Improve ParticleSurface**

保留目前每個 Mechanical Component 個別重建表面的方式，研究 kernel、resolution、deformation-aware extraction 與 mesh processing。

優點是整合成本較低。

缺點是缺少直接來自切割路徑的幾何資訊。

**Approach B — Cut-aware Reference-grid Reconstruction**

參考 BladeMaster，建立追蹤實際刀刃切割事件的幾何表示。

將切割產生的局部不連續與切面交點保存，再進行局部表面重建。

這應列為主要研究候選。

**Approach C — Hybrid Reconstruction**

原始外表面使用 ParticleSurface 或其他適合的方法，切割產生的內部表面使用 Cut-aware Reconstruction。

研究如何維持邊界連續、法向一致與正確拓撲。

**Approach D — Alternative Mesh Optimization**

研究 Elastic Snapping、其他 contouring 或 mesh refinement 方法。

但必須區分幾何重建與物理頂點移動，不能直接將另一種物理離散方法的操作搬到 TLMPM。

請製作技術比較表，包含：

- Expected geometric fidelity
- Topological consistency
- Physical consistency
- Implementation complexity
- GPU memory cost
- Runtime cost
- Dynamic multi-cut support
- Integration risk
- Validation difficulty

### Phase 0 Deliverables

建立：

`docs/WP12_PLAN.md`

`docs/reports/WP12_BASELINE_AUDIT.md`

`docs/reports/WP12_SURFACE_TECH_REVIEW.md`

先確認研究結論、架構與量化評估方法，再進入後續開發。

---

# Phase 1 — Diagnose Cut Surface Geometry

這是整個 WP-12 最優先的技術診斷。

## 1.1 Separate Physical Error from Rendering Error

目前 `cutting_surface.py` 依 Mechanical Component 使用 Newton `ParticleSurface` 重建表面。

因此切口不平整可能來自：

1. Particle distribution。
2. Physical deformation。
3. Uncut or wrongly labelled particles。
4. Component fragmentation。
5. Surface extraction resolution。
6. Smoothing kernels。
7. Surface-to-physics synchronization。
8. Actual cutting trajectory。

請建立能逐一辨別這些因素的診斷工具。

不要將所有不平整都歸因於 Surface Reconstruction。

## 1.2 Build a Multi-layer Debug Viewer

對相同切割結果產生：

- Original particles
- Current particles
- Persistent side labels
- Mechanical connectivity
- Blade swept volume
- Actual contact region
- Existing ParticleSurface
- Proposed Cut-aware Surface
- Final rendered surface

允許切換顯示。

至少支援：

- Arbitrary cross-sections
- Top view
- Side view
- Blade-aligned view
- Cut-face close-up

Debug visualization 可以使用 privileged state，但不得出現在 deployable policy observation。

## 1.3 Establish Independent Geometry Tests

使用不依賴任務成功與否的控制實驗：

**G1 — Uncut box**

沒有切割時確認幾何體積、邊界、平面、角落與法向。

**G2 — Known static cut**

使用已知幾何交點與理想切面測試 reconstruction algorithm。

此測試僅作為幾何單元測試，不得把 oracle cut 注入正式物理模擬。

**G3 — Full physical cut**

使用實際 Law 8 產生的切割結果驗證物理與表面的一致性。

**G4 — Partial cut**

刀具只切入一部分材料。

不得因為可見裂縫，就錯誤宣告整個物件已完全分離。

**G5 — Thin shaving**

削除小型凸起，產生薄切屑。

確認微小碎片、薄層及新生成表面。

**G6 — Multiple cuts**

同一物體依序執行多刀。

檢查新切面與既有切面的交會。

**G7 — Deformation and recontact**

切完後壓縮、搬動、碰撞、重新接觸。

確認不同側切面仍正確跟隨材料運動。

**G8 — No-cut negative controls**

刀具靠近、接觸、壓住、滑過但未滿足物理切割條件時，不得產生假切口。

## 1.4 Quantitative Metrics

請建立以下指標：

| Metric | Purpose |
|---|---|
| Cut Surface Distance P50/P95/Max | 與有效切割幾何的偏差 |
| Signed Surface Error | 切面是否偏向刀具某側 |
| Normal Angular Error | 切面法向與局部切割方向的誤差 |
| Surface Roughness | 局部起伏程度 |
| Mesh Validity | Degenerate / invalid / self-intersecting faces |
| Boundary Continuity | 原始表面與新切面的接合 |
| Topological Consistency | 與物理切割連接狀態一致 |
| Detached Mass | 實際分離質量 |
| Remaining Bridges | 未切斷的材料連接 |
| Volume Error | 重建體積與材料狀態的差異 |
| Temporal Stability | 表面是否隨時間不合理跳動 |

特別注意：

**不能直接用最終 Mesh 到「預期理想切割平面」的距離來判斷真實物理切割品質。**

因為材料受到刀具作用後可能發生變形。

應建立 Material-reference-space 與 Current-space 的對應，將原始切割幾何和後續材料變形分開測量。

### Phase 1 Gate

必須能回答：

- 現有切面不平整，有多少來自粒子物理？
- 有多少來自 Surface Reconstruction？
- 有多少來自刀具實際路徑？
- 哪些缺陷可由 Cut-aware Geometry 修正？
- 哪些必須修改 Cutting Law 或 Contact Model？

沒有完成這項診斷，不應直接以 smoothing 或 mesh editing 作為解決方案。

---

# Phase 2 — BladeMaster-style Flat Cutting Board Workcell

## 2.1 Target Scene

請將預設工作場景調整得更接近 BladeMaster：

- 一張正常的工作桌。
- 一塊完整木製切割板。
- 切割板直接放在桌面上。
- 工件直接放在切割板上。
- Franka 使用物理夾持的長直刃刀具進行切割。
- 第二支 Franka 維持停放，不主動協助切割。
- 兩個 wrist-mounted D405 RGB-D cameras。
- 一個 global D455 RGB-D camera。

不需要目前的多個灰色 Riser / Pedestal。

場景的自然程度、機器人工作空間與攝影機觀察品質都需要改善。

## 2.2 Preserve Scene Config Architecture

目前 `rcs_scene_v1` 已支援：

- `solid_board`
- `slotted_board`
- Explicit fixture boxes
- Workcell configuration

請優先擴充這套架構。

建議新增版本化 workcell profile，例如：

`blade_master_flat_board_v1`

要求：

- Default：continuous solid board。
- Optional：slotted board。
- Workcell configuration 允許尺寸、材質、位置和摩擦參數變化。
- 木板與桌面的幾何不依賴某一個 Demo。
- 工作台可供不同物體、不同 goal 共用。

原有 `wp11_workcell_v1` 保留。

## 2.3 Geometry and Collision Consistency

需確保：

1. 木板底部與桌面接觸。
2. 工件底部與木板頂面接觸。
3. 沒有可見懸浮、初始穿透或錯位。
4. 視覺 Mesh 和 PhysX Collider 對齊。
5. TLMPM 使用的 Fixture Contact Geometry 與實際板面一致。
6. 工件受到重力後能穩定放置。
7. 刀具不會直接穿過實體木板。
8. Material Contact 和 Rigid Contact 不會重複計算或互相矛盾。
9. 木板移動、厚度或位置改變後，Contact Plane、IK、Collision Preflight 會同步更新。
10. 不得在畫面中移除 Riser，卻仍讓物理求解器使用不可見的舊支撐體。

## 2.4 Critical Problem — Through-cut Versus Solid Board

這是本階段重要的物理衝突。

當刀刃要完整切斷放在實體木板上的材料時，可能需要到達材料的最底層。

但刀具不能不受限制地穿透木板。

因此請研究：

**Option A — Cut to Board Surface**

刀刃恰好到達材料底面，利用有效切削幾何完成分離。

確認最底層粒子確實被切開。

**Option B — Physical Board Contact**

刀刃與木板產生真實接觸，運動受到阻擋。

如果這使最底層材料無法完全切斷，就必須報告限制。

**Option C — Sacrificial / Compliant Board**

若材料與模擬能力允許，建立可受刀具壓入或產生局部痕跡的切割板模型。

這需要額外物理建模，不能僅用允許穿透的碰撞設定代替。

**Option D — Explicit Slotted Support**

對需要刀刃越過工件底面的特殊切割，保留可選擇的 Slotted Board。

它不能成為所有場景的隱藏預設。

請透過實驗選擇主要方案。

不要為了同時達到「完整切斷」與「完整木板外觀」，讓刀具偷偷穿過實際為剛體的木板。

## 2.5 Scene Layout and Camera Validation

調整：

- Table dimensions。
- Board dimensions。
- Workpiece location。
- Robot reach。
- HOME pose。
- Global camera position。
- Wrist camera attachment and calibration。
- Lighting and material appearance。

畫面應清楚呈現刀具、工件及切面。

但 Presentation Camera 不得替代正式 Policy Observation。

確保 wrist cameras 在手臂運動時正確跟隨，RGB 與 Depth 都能使用。

## 2.6 Workcell Acceptance

至少執行：

- Static material-on-board stability。
- Gravity settling。
- Board collision。
- Friction and sliding。
- Knife-board contact。
- Partial cut。
- Complete through-cut。
- TRIM。
- Multiple slices。
- Multiple protrusions。
- Arm return-home。

比較新舊 Workcell 的：

- Penetration depth
- Contact forces
- Object displacement
- Cutting success
- Remaining material bridges
- Tool grasp stability
- IK / preflight failures
- Performance

### Phase 2 Deliverables

建立：

`docs/reports/WP12_WORKCELL.md`

包含：

- Scene geometry diagram
- Contact architecture
- Before / after screenshots
- Physical validation
- Failure cases
- Default workcell recommendation

---

# Phase 3 — Cut-aware Surface Reconstruction

這是 WP-12 最重要的核心技術工作。

## 3.1 Objective

將切面重建從單純依 Particle Components 生成表面，提升為能使用：

- Actual blade motion
- Physical cutting events
- Persistent side labels
- Reference-space cut topology
- Material deformation

建立切割產生的新表面。

優先研究 BladeMaster Section III-F 的方法，但需適配現有 TLMPM v5。

不要求完全逐行重現論文。

要求物理一致、可維護、可驗證。

## 3.2 Proposed Architecture

建議研究以下分層架構：

**Layer 1 — Physical Material State**

由 TLMPM v5 保持：

- Particle positions
- Velocities
- Deformation gradients
- Persistent side codes
- Mechanical connectivity
- Contact state

此層仍是材料物理的唯一真實來源。

**Layer 2 — Cut Event Tracking**

建立物理允許的切割事件資料：

- Cut ID
- Blade edge pose
- Edge segment geometry
- Swept path
- Contact and engagement
- Cutting activation
- Material side labels
- Affected reference regions
- Physical cut progression

需要與 Law 8 的切割條件一致。

不得讓尚未發生有效物理切割的 knife sweep 生成虛假裂縫。

**Layer 3 — Reference-grid Cut Topology**

研究 Linked-volume Representation。

維護：

- Reference-grid cells
- Neighbor links
- Original material boundary
- Cut intersections
- Local material regions
- Persistent disconnected links

在有效切割事件發生時更新局部連接關係。

重要：這套 visual topology 應與物理 side labels 及 mechanical coupling 對應。

不能成為第二個與 TLMPM 無關的切割真相。

**Layer 4 — Local Surface Reconstruction**

使用 Hermite intersections：

- Intersection position
- Surface normal
- Material-side identity
- Cut identity

研究 Local Dual Contouring。

在適當的材料 region 生成新表面 representatives。

需要特別處理：

- Partial cuts
- Open crack fronts
- Cut intersection
- Thin chips
- Multiple cut sides
- Near-coincident cut faces
- Original boundary intersections

允許切面兩側擁有獨立頂點。

不應重新 weld 已被物理切開的區域。

**Layer 5 — Cut-side-aware Embedding**

將 Cut Surface Vertices 綁定到對應的材料運動場。

研究：

- Reference positions
- Trilinear interpolation
- Compatible TLMPM stencils
- Persistent side codes
- Local deformation
- Material-region ownership

切割後表面需要自然地：

- Move
- Stretch
- Compress
- Rotate
- Recontact
- Slide

不可以一直停留在原始切割平面。

**Layer 6 — USD / Rendering**

把更新後的幾何傳給現有 USD Surface Bridge。

確保：

- Temporal consistency
- Correct normals
- Correct face orientation
- No false welding
- No disconnected phantom surfaces
- No severe flickering

## 3.3 Geometry Versus Physics

Cut-aware Reconstruction 主要是一種幾何表示。

如果採用單向 TLMPM embedding，表面 Mesh 不應再獨立施加材料動力學。

若發現顯示 Mesh 與物理接觸邊界之間有顯著落差，請明確記錄，並評估是否需要後續雙向耦合。

不能默認 Render Mesh 已經代表真實碰撞表面。

## 3.4 Reconstruction Options

至少實作可比較的兩種 profile：

- `particle_surface_v1`：現有 baseline。
- `cut_aware_surface_v1`：新方法。

如有必要，可加入：

- `hybrid_surface_v1`

Profile 選擇必須明確記錄於 episode metadata。

不得修改 frozen baseline 的輸出。

## 3.5 Core Tests

**S1 — Flat cut**

真實切開一個方塊。

新切面應在合理解析度下接近實際切割路徑。

**S2 — Deformed cut**

先壓縮材料，再切割。

檢查切面跟隨已變形材料。

**S3 — Curved trajectory**

刀具執行具有方向變化的有效切割路徑。

切面不能被強制改成單一平面。

**S4 — Partial cut**

切入一半後停止。

保留正確的未切斷材料連接。

**S5 — Intersecting cuts**

兩刀交錯。

保留正確的局部幾何和切割歷史。

**S6 — Multiple parallel slices**

連續四刀。

生成五個獨立部分及對應表面。

**S7 — Thin-chip trimming**

削下薄切屑。

確認切屑厚度、質量和表面沒有非物理遺失。

**S8 — Recontact and deformation**

切開後重新推回兩側材料。

表面可接觸，但不能重新焊接。

**S9 — Negative controls**

未啟用切割或未達成切削條件時，不能生成新的切面。

## 3.6 Surface Quality Objectives

預先使用 Phase 1 的幾何控制實驗設定門檻。

建議以 particle spacing 為尺度，研究：

- P95 surface error
- Max geometric error
- Mesh validity
- Surface normal error
- Discontinuity fidelity
- Conservation of represented material volume
- Temporal stability

對解析幾何控制案例，可以先將 P95 幾何誤差不超過約 1–2 個 particle spacing 作為候選研究目標。

但這不是可直接宣告通過的既定標準。

正式門檻必須在評估前凍結，並說明：

- 實際解析度
- Reference geometry
- Deformation handling
- Expected discretization error
- Measurement uncertainty

**不能用平滑度單獨判斷成功。**

真實壓痕與變形必須保留。

## 3.7 Performance

量測：

- Surface extraction time
- Cut event processing time
- Local topology update
- Embedding update
- USD upload
- GPU memory
- Mesh size
- Total episode runtime

比較 full recomputation 與 local update。

優先考慮 incremental updates，但不要為了加速犧牲幾何一致性。

### Phase 3 Gate

新方法必須在獨立幾何測試中優於或至少不劣於原方法，並且：

- 不新增虛假物理分離。
- 不改寫粒子運動。
- 不破壞既有切割歷史。
- 不遺失有物理意義的材料。
- 不出現嚴重破面或拓撲錯誤。
- 在多刀、薄切屑與變形案例中保持有效。

若 Cut-aware Reconstruction 過於昂貴，應比較 Hybrid 方法，而不是直接放棄物理一致性。

---

# Phase 4 — Cutting Physics and Generalization Improvements

## 4.1 Improve Existing Law 8 Failure Regions

WP-11 已知：

1. 約 ±30–35° 斜切時，底部數層粒子可能沒有正確被標記，導致材料仍連接。
2. 45° rake 的削切可能因 movement alignment gate 而無法啟動。
3. 一些連續切割在內部留下少量未分離粒子。
4. 部分薄切屑和底材產生不合理的殘留或局部壓陷。
5. TRIM 與 contact force 尚未證明解析度收斂。
6. 不同工作台支撐條件可能改變切割結果。

請不要單純放寬全部 Cut Assignment Thresholds。

需要針對每類失敗研究根因。

## 4.2 Cutting Diagnostics

增加每個有效刀刃 segment 的診斷：

- Blade motion direction
- Edge tangent
- Blade normal
- Contact status
- Contact impulse / work
- Cutting progress
- Stroke validity
- Plowing state
- Assigned side codes
- Unassigned particles
- Remaining connected regions

對失敗案例，應能追溯是哪一個物理條件阻止材料分離。

尤其需要分辨：

- Real insufficient cutting depth
- Unsupported cutting orientation
- Missing side assignment
- Numerical resolution problem
- Contact instability
- Tool trajectory error
- Support collision

## 4.3 Investigate Alternative Cutting Laws

若 Law 8 確有機制限制，研究新的 versioned cutting law，例如 `law9`。

但在採用新法則之前：

1. 建立對照測試。
2. 保留原本 Law 8。
3. 明確定義新的物理假設。
4. 研究方向性切削與壓切的差異。
5. 保留 blunt-contact negative controls。
6. 驗證材料不會因單純碰撞而自動被切斷。
7. 檢查物理能量、接觸反力和動量。
8. 驗證現有 in-plane CUT 與 TRIM 不退步。
9. 完整重跑原 Law 8 acceptance matrix。

任何新法則都要和既有系統公平比較。

## 4.4 Generalized Cutting Orientation

目前已定義：

- Edge tilt τ
- Edge skew ψ
- Blade rake ρ
- Motion elevation μ

請研究不同姿態的實際可行範圍。

至少涵蓋：

- Vertical cuts
- Oblique cuts
- Near-horizontal trimming
- Different approach directions
- Different yaw orientations
- Different stroke directions
- Multi-step reorientation
- Sawing or repeated strokes where physically valid

不要將 ρ、τ、ψ、μ 混為同一角度。

`ruler_knife_v2` 具有實際厚度與 bevel。

例如 rake 0° 可能因刀具幾何造成下表面刮入材料，不能直接假設所有角度都可以安全切割。

最終應提供：

**Supported Cutting Orientation Envelope**

明確標示：

- Physically feasible
- Kinematically reachable
- Validated cutting
- Partially validated
- Unsupported
- Known failure

目標是擴大有效切割範圍，而非聲稱無限制支援所有角度。

## 4.5 Workpiece Generalization

目前主要是 box / bar 加 protrusions。

請評估如何擴充幾何生成器。

優先研究：

- Different bar lengths / widths / heights
- Rectangular blocks
- Thin slabs
- Different aspect ratios
- Multiple protrusions
- Gaussian bumps
- Dome-shaped bumps
- Asymmetric bumps
- Overlapping bumps
- Irregular local surface features
- Different workpiece yaw and translation

若既有 HeightField Representation 無法表達某些形狀，請研究新的 Geometry Provider。

不要在尚未支援任意 Mesh Sampling 的情況下宣稱完全支援任意物體。

需要提供明確的 geometry capability matrix。

## 4.6 Material Generalization

研究 Material Profile 是否能讓環境模擬不同材料特性。

例如：

- Soft elastic material
- More plastic / clay-like material
- Stiffer deformable material
- Different friction
- Different cutting resistance
- Different density

不同 Material Profile 必須是可解釋的參數組合。

不得僅靠更改顏色宣稱是不同材料。

材料數值尚未校準時，必須標示為 provisional。

## 4.7 Resolution and Numerical Stability

建立系統性解析度研究。

以現有正式設定作為比較基準：

- 1 mm particles
- 4 mm grid
- 0.1 ms material timestep

研究更粗與更細的配置。

確認：

- Separation result
- Remaining bridges
- Force convergence
- Deformation
- Minimum det(F)
- Contact penetration
- Surface fidelity
- Runtime
- Memory

比較時需要控制其他變數，並使用有效的 timestep / grid ratio。

不得只因某一組設定跑得成功就宣稱數值收斂。

### Phase 4 Deliverables

建立：

`docs/reports/WP12_CUTTING_PHYSICS.md`

`docs/reports/WP12_RESOLUTION_STUDY.md`

提供：

- Failure root-cause classification
- Candidate law comparison
- Supported orientation envelope
- Numerical convergence evidence
- Remaining limitations

---

# Phase 5 — Systematic Generalization Benchmark

本階段希望建立的不是更多手工寫死的 Demo，而是能測量環境泛化能力的 Benchmark。

## 5.1 Define Task Families

至少包含：

| Test | Task |
|---|---|
| A | Single straight cut |
| B | Oblique and direction-varied cuts |
| C | Near-horizontal trimming |
| D | Multiple sequential slices |
| E | Multiple protrusion removal |
| F | Mixed sever + trim |
| G | Cut + push / relocation |
| H | Shape / pose / support variation |

對每個任務都應使用 Scene + Goal Configuration 生成，而不是在求解器中加入對應 task-specific branch。

## 5.2 Randomization Dimensions

支援獨立、可組合的 randomization：

**Scene**
- Workpiece position
- Workpiece yaw
- Workpiece size
- Workpiece shape
- Protrusion count
- Protrusion location
- Protrusion geometry
- Board dimensions
- Board friction

**Cutting**
- Target location
- Cut count
- Cut order
- Cut depth
- Knife orientation
- Motion direction
- Approach position
- Tool speed

**Physics**
- Material profile
- Friction coefficients
- Contact parameters
- Mass / density
- Cutting resistance

不同 randomization 必須可透過 seed 還原。

## 5.3 Avoid Impossible Scene Distributions

將樣本分類為：

1. Valid and feasible。
2. Scene geometrically infeasible。
3. IK unreachable。
4. Collision constrained。
5. Physically unsupported。
6. Execution failure。

不可把所有不可能的任務直接算成控制器失敗。

但也不能把失敗的任務從整體報告中直接刪除。

必須分別報告：

- Feasibility rate
- Success rate among feasible cases
- Overall success rate
- Failure class distribution

## 5.4 ID / OOD Evaluation

建立兩組預先宣告的分布。

**In-distribution**

位於已確認有效的幾何、姿態及材料參數區間。

**Out-of-distribution**

超出既有 Demo 的較大位置、角度、尺寸或形狀範圍，但仍需將物理可行性和任務成功分開評估。

不應將預先知道不可能完成的場景當作泛化失敗的唯一依據。

## 5.5 Predeclared Test Protocol

建議最少：

- 6 個主要 task strata。
- 每個 stratum 至少 10 個預先固定的 randomized configurations。
- 總計至少 60 個 kinematic feasibility / physics evaluation candidates。
- 從各 stratum 分層抽樣 Native Isaac Sim 測試，目標至少 3 個，每個實際執行數量及未執行原因皆需記錄。
- 額外保留 held-out seeds。
- 必要時建立具獨立分布的 OOD set。

正式測試前保存：

- Seed list
- Parameter ranges
- Test definitions
- Metric definitions
- Evaluation thresholds
- Execution budget

不能因結果不好而事後調整測試集合。

可以設立 development set 供研究迭代，但最終 held-out set 不得反覆用於調參。

## 5.6 Success Metrics

對每個 episode 評估：

**Task outcome**
- Target removal fraction
- Number of pieces
- Sever completeness
- Piece displacement
- Cutting location error
- Goal completion

**Physical quality**
- Substrate retention
- Remaining unwanted bridges
- Contact stability
- Knife grasp stability
- Material loss
- Penetration
- Physical cutting validity

**Geometry quality**
- Cut surface error
- Mesh topology
- Reconstruction validity
- Volume consistency
- Deformation fidelity

**Runtime**
- Episode wall time
- Simulation throughput
- Surface reconstruction cost
- GPU memory
- Reset overhead

**Reliability**
- Reset success
- Multi-episode stability
- Recorded output validity
- Reproducibility

## 5.7 Generalization Goals

對預先確認可行且屬於支援範圍的 ID 任務，可將下列數字作為工程改善目標：

- Kinematic task success：≥ 90%。
- Native task success：≥ 80%。
- No silent solver failures。
- No false positive task success。
- No hidden sample exclusion。

這些是建議的目標，不代表目前已經達到，也不能代替統計驗證。

必須報告實際分母、樣本數與不確定性。

對 OOD 任務不應假設必須達到相同成功率，而是量測有效範圍與失敗邊界。

若未達目標，應記錄為未達標，而非更改定義。

## 5.8 Mixed-operation Tasks

特別測試：

1. Cut → Cut。
2. Cut → Push。
3. Trim → Trim。
4. Trim → Clear chip。
5. Sever → Trim。
6. Trim → Sever。
7. Multiple bumps removed sequentially。
8. Cut → Reposition → Cut。

確認：

- Cut history persists。
- Cut indices correctly assigned。
- Earlier cuts remain valid。
- No accidental reconnection。
- Evaluator correctly scopes each operation。
- Robot and tool remain physically valid。

### Phase 5 Deliverables

建立：

`docs/reports/WP12_GENERALIZATION.md`

附完整成功率矩陣、失敗案例與 OOD 邊界分析。

---

# Phase 6 — Performance, API Integrity & Agentic-control Readiness

這階段不需要真正使用 agentic control。

但需要確保未來接入 agentic control 時不必再重構整個模擬器。

## 6.1 Environment API

保留：

```python
from robotic_cutting_sim.live_env import open_cutting_env

with open_cutting_env(
    runtime="isaac_sim",
    preset="cutting_research_v1",
    scene="...",
    goal="...",
    runtime_profile="full_physics_v1",
    seed=42,
) as env:
    obs, info = env.reset(seed=42)

    obs, reward, terminated, truncated, info = env.step(action)

    result = env.evaluate()
```

新場景、Surface Profile 和 Material Profile 應透過版本化 config 或相容的 API 擴充。

不要讓 Policy 必須直接存取 TLMPM 內部結構才能執行。

## 6.2 Observations

確認保留：

- Wrist D405 RGB-D ×2
- Global D455 RGB-D
- Robot joint states
- End-effector pose
- Gripper state
- Goal description
- Action result
- Episode state

物理粒子、cut codes、真實刀具位姿和 contact oracle 應仍屬 privileged state。

一般 Policy Observation 不能無意洩露答案。

## 6.3 Action Interface

必須支援：

- Cartesian motion
- Orientation control
- Tool reorientation
- Cut enable / disable
- Cut index
- Sequential operations
- Valid action rejection
- Safe reset
- Episode termination

驗證不同 Controller 可以執行同一 Scene + Goal。

包括 scripted planner 與經公開 API 輸入的獨立 action sequence。

## 6.4 Recording

每次 episode 應能保存：

- Full configuration
- Runtime profile
- Action sequence
- Camera observations
- Physical state
- Cut events
- Contact diagnostics
- Surface reconstruction metadata
- Evaluation
- Failure classification
- Randomization seed
- Code commit
- Timing information

對新版 Surface Reconstruction，請將必要的 reference-grid topology / Hermite / embedding provenance 納入可追溯記錄。

可以依 storage budget 使用摘要、checkpoint 或可重建事件記錄，但不得丟失驗證所必需的資料。

## 6.5 Repeated Episodes

至少執行：

- Consecutive resets
- Repeated cuts
- Mixed tasks
- Failure followed by reset
- Failed action followed by valid action
- Native multi-episode run

確認：

- No state leakage
- No stale cut labels
- No stale surface topology
- No stale contacts
- No camera attachment drift
- No accidental tool detachment
- No unbounded resource growth

目前 Native 已知存在 GPU float32 atomics 造成的非逐位確定性，因此不得把 GPU bitwise repeatability 當成已具備的能力。

應區分：

- CPU deterministic replay
- Native verdict-level repeatability
- Numerical trajectory variation

## 6.6 Profiling

分析新增 Cut-aware Reconstruction 後：

- TLMPM physics time
- Cut event update
- Surface extraction
- USD upload
- Contact computation
- Camera rendering
- Total wall time

保留：

`full_physics_v1`

作為有效物理驗收 profile。

其他加速 profile 可用於開發，但若物理設定不同，不能使用其結果冒充正式物理驗收。

## 6.7 agentic-control-ready Requirements

最後確認：

- 任務可以由配置產生。
- 任務完成可以客觀評估。
- Agent 可以反覆 Reset / Execute / Evaluate。
- Agent 可以取得失敗原因。
- 不同研究方法可以共用環境。
- Dataset 可以收集成功與失敗案例。
- 不需要修改物理引擎才能替換 Policy。
- 不需要讓 Policy 讀取 privileged state 才能執行正式任務。

這裡的 agentic-control-ready 只代表環境具有研究接口與驗證能力。

不代表已訓練出自主切割 Policy，也不代表材料模型已經過真實世界校準。

---

# Phase 7 — Final Validation and Delivery

## 7.1 Regression Tests

完整重跑：

- Existing CPU test suite
- Original frozen cutting controls
- Law 8 acceptance tests
- WP-11 multi-cut tests
- WP-11 grasp tests
- WP-11 Policy-ready tests
- New WP-12 geometry tests
- New workcell collision tests
- New generalization tests

保留既有 `final_v4` 的結果身份與評估規則。

新版可以另行比較，但不得覆寫歷史證據。

## 7.2 Required Native Demonstrations

請產出以下真實 Native Isaac Sim 影片。

**Demo 1 — Clean Straight Cut**

完整木板上的工件，以物理刀具切斷。

顯示切割過程、新切面及分離後的運動。

**Demo 2 — Oblique Cut**

以非零角度切割，展示與直切不同的幾何結果。

**Demo 3 — Near-horizontal Trim**

削除一個凸起，保留真實材料變形。

**Demo 4 — Multiple Slices**

同一物件連續切四刀，生成五個獨立部分。

**Demo 5 — Multiple Bumps**

同一物件具有至少三個凸起，依序移除。

**Demo 6 — Mixed Operation**

例如：

Cut → Reposition → Trim

或：

Trim → Trim → Clear Chip

**Demo 7 — Generalized Scene**

使用未曾針對該 seed 手工設計軌跡的隨機場景。

每個 Demo 都需要：

- Start frame
- Cutting progress
- Final frame
- Full video
- Result JSON
- Physical metrics
- Geometry metrics
- Runtime metrics
- Exact source configuration

影片應使用真正的 Native runtime，不得用 kinematic replay 冒充。

## 7.3 Visual Comparison

至少提供以下比較：

- WP-11 Workcell vs WP-12 Workcell
- ParticleSurface vs Cut-aware Surface
- Original cut vs improved cut
- Original multi-cut vs improved multi-cut
- Geometric reconstruction vs actual particle state

對切口提供 close-up。

不能只提供遠距離影片，讓重要缺陷難以觀察。

## 7.4 Acceptance Matrix

最終以以下矩陣驗收：

| Area | Acceptance |
|---|---|
| Workcell | 完整木板直接位於桌面，無隱藏支撐 |
| Board physics | 真實接觸，無非物理穿透 |
| Grasp | Franka 物理夾持保持有效 |
| Straight cut | 物理分離與幾何一致 |
| Oblique cut | 宣告有效角度範圍及結果 |
| TRIM | 真實移除凸起，保留底材 |
| Multi-cut | 多刀歷史持續且正確分離 |
| Surface | 與切割事件及材料運動一致 |
| Deformation | 不強制平整真實變形 |
| Negative controls | 無切割條件時不產生假切口 |
| Generalization | 預先宣告的測試與失敗分布 |
| Evaluation | 獨立、客觀且不可事後調整 |
| API | Reset / Step / Evaluate 正確 |
| Recording | 完整 evidence 和 provenance |
| Reliability | 多 episode 無狀態汙染 |
| Performance | 有完整 profiling，成本可量化 |
| Regression | 舊功能不被破壞 |
| Reproducibility | Clean checkout 可以還原測試 |

每一項都要明確標示：

`PASS / FAIL / PARTIAL / NOT_RUN / NOT_APPLICABLE`

不能把 PARTIAL 或 NOT_RUN 寫成 PASS。

---

# Required Documentation and Deliverables

最終至少包含：

**Planning**
- `docs/WP12_PLAN.md`

**Research**
- `docs/reports/WP12_BASELINE_AUDIT.md`
- `docs/reports/WP12_SURFACE_TECH_REVIEW.md`

**Implementation and physics**
- `docs/reports/WP12_WORKCELL.md`
- `docs/reports/WP12_SURFACE_RECONSTRUCTION.md`
- `docs/reports/WP12_CUTTING_PHYSICS.md`
- `docs/reports/WP12_RESOLUTION_STUDY.md`

**Validation**
- `docs/reports/WP12_GENERALIZATION.md`
- `docs/reports/WP12_PERFORMANCE.md`
- `docs/reports/WP12_FINAL_VALIDATION.md`

如有重要架構選擇，新增 ADR。

所有新增配置、測試與執行程式都要納入 repository。

更新 README 與 PROGRESS，清楚區分已完成、部分完成與已知限制。

## Implementation Workflow

請採用以下流程：

1. 從乾淨的最新 main 建立 WP-12 branch。
2. 完成 audit、research review 與 baseline freeze。
3. 為每個 Phase 定義測試和 acceptance protocol。
4. 先建立失敗重現與 negative controls。
5. 實作最小可驗證修改。
6. 執行 unit tests 與 kinematic physics tests。
7. 對重要修改執行 Native Isaac Sim。
8. 比較 baseline 與新方法。
9. 記錄所有失敗與原因。
10. 依證據迭代改善。
11. 每個 Phase 完成後 commit。
12. 最終進行獨立 review、full regression 和 CI。
13. Push branch，並提供可審查的 PR / commits。

允許平行處理獨立工作，例如 Workcell 和 Geometry Diagnostic。

但 Physical Cut Events、Surface Reconstruction 和 Cutting Law 的修改必須明確控制資料依賴及整合順序。

每個 Phase 必須有自己的驗證結果。

## Important Constraints

- 不得重寫整個 TLMPM，除非有明確證據證明必要。
- 不得犧牲原有 multi-cut 能力。
- 不得改成直接將刀具固定在手上。
- 不得讓刀具和木板發生無效穿透。
- 不得將純視覺幾何改動稱為 Cutting Physics Improvement。
- 不得用理想切割平面修正實際材料結果。
- 不得隱藏失敗案例。
- 不得靜默放寬評估。
- 不得用固定 Demo 路徑代表泛化能力。
- 不得宣稱未驗證的材料、角度或速度範圍已受支援。
- 不得使用 `research_fast_v1` 的結果冒充 full-physics validation。
- 不得覆蓋舊 WP 的 frozen evidence。

## Completion Criteria

只有在以下條件成立時，才能宣告 WP-12 完成：

1. 新工作台場景具有正確物理接觸。
2. Cut-aware Surface Reconstruction 有完整研究、實作與測試證據。
3. 新切面的品質改善可以量化。
4. 切割物理和切面幾何沒有重大矛盾。
5. 現有核心 Demo 不退步。
6. 已知 Law 8 失敗區域經研究、改善或明確界定。
7. 泛化測試涵蓋新的幾何、姿態與任務組合。
8. 評估結果包含成功和失敗案例。
9. Native Isaac Sim 有足夠的實際驗證。
10. API、Reset、Recording、Evaluation 維持穩定。
11. CI 和必要 regression tests 通過。
12. 所有程式碼與研究成果完成 commit / push。

若某項研究問題無法在合理工程成本內解決，應保留可運作的改善成果、完整記錄失敗原因及後續方案，不得為了宣告完成而降低物理標準。

## Final Report Format

完成後請提供：

### 1. Executive Summary
本次完成哪些核心能力。

### 2. Architecture Changes
包含主要模組、資料流與設計選擇。

### 3. Physics Improvements
解決哪些物理問題，哪些尚未解決。

### 4. Surface Reconstruction Results
原方法和新方法的定量比較。

### 5. Workcell Results
工作台和實際碰撞品質。

### 6. Generalization
各任務成功率、有效範圍及失敗分布。

### 7. Performance
新增功能的計算成本與改善。

### 8. Validation
測試、CI、Native run 與證據路徑。

### 9. Remaining Limitations
尚未校準、尚未收斂及未支援的部分。

### 10. Final Deliverables
Commit SHA、branch、PR、影片、文件及可執行指令。

---

# Final Instruction

請自主完成從研究、規劃、實作、測試到驗證的完整工作流程。

不要只停留在提出方案，也不要只完成幾個示範影片。

面對失敗時，應系統性分析物理、幾何、接觸、數值解析度、控制路徑及評估機制，選擇有研究依據的方法持續改善。

但必須保持工程可追溯性，避免不受控制的重寫。

**最終要交付的是一個更可信、更通用、能被後續自主研究 Agent 使用的 Robotic Cutting Simulation Environment，而不是一組針對固定場景調整成功的 Demo。**

---

## 附錄：使用者的背景說明（2026-10-30）

RoboticCuttingSim — 場景、切面品質與泛化能力升級規劃
我同意先把模擬環境完善，再交給 agentic control 使用。這一階段應該把 RoboticCuttingSim 從「已能完成指定切割 Demo」，提升成具有可信切割幾何、自然工作台配置，以及更廣泛切割能力的研究環境。

我核對了最新 main（fbe99bd0）與 BladeMaster 的技術後，認為有三個值得優先處理的問題。
方向	現況	目標
Cut Surface	依分離後的粒子群重建表面，切口容易呈現起伏	切面幾何更準確、連續，並保留真實變形
Workcell	木板分段，下方有灰色墊高支柱	像附圖一樣，完整切割板直接放在桌面
Generalization	有多刀、多凸起及場景隨機化，但仍存在明確失敗區域	不同物體、角度、切割順序均能合理執行與評估

目前最重要的技術發現是：

RoboticCuttingSim 的材料分離和表面生成是兩套不同機制。 現在 cutting_surface.py 使用 Newton ParticleSurface，依照 Mechanical Component 分別重建表面；它並沒有直接從刀刃實際掃過的位置建立專用的 Cutting Surface。

BladeMaster 的 Section III-F 則使用 Reference-grid Linked-volume Representation、Hermite Intersection Data 和 Local Dual Contouring 建立切面，再讓切面隨對應的 TLMPM 運動場變形。這是值得優先研究的直接技術路線。

不過，更平整的切口不等於把材料強制切成理想平面。我們仍須保留刀具造成的真實壓縮、塑性變形、切屑厚度、材料位移和接觸結果。

另外，現有 rcs_scene_v1 已有 solid_board、slotted_board，因此不必重新建立場景系統；需要處理的是目前 wp11_workcell_v1 會自動產生 Board + Riser 的實作，以及刀具切穿材料時與實體木板的接觸。

我建議下一個工作包命名為：

WP-12 — Physically Consistent Cut Surfaces, BladeMaster Workcell & Generalized Cutting

實作順序應是：先診斷切面 → 建立真正的平面木板場景 → 完成 Cut-aware Reconstruction → 系統性改善泛化 → 最後做 agentic-control-ready 驗收。
1. 建議參考的研究技術
我會讓 Claude 先比較幾條有研究依據的方案，而不是直接重寫 TLMPM。
技術	值得借鑑的部分	優先級
BladeMaster (2026)	Reference-grid Cut Surface、Persistent Discontinuity、Cut-side-aware Embedding	最高
Dual Contouring of Hermite Data	使用切面交點與法向重建更清晰的幾何邊界	高
Dynamic Cutting with Elastic Snapping (2025)	研究切割路徑與 Mesh Quality 的關係，提供可比較的替代方案	中
NVIDIA Implicit MPM	粒子表面重建、接觸及材料求解器的實作參考	中，暫不替換 TLMPM

BladeMaster 最值得研究的是：刀刃切開材料時，同步更新幾何連接，保存切面交點與法向，並讓新生成的兩側表面各自跟隨其材料運動。它不只是對已經分離的粒子群重新產生 Mesh。

Dual Contouring 使用 Hermite Data（交點與法向）進行表面重建，適合研究如何保留清楚的幾何特徵；Elastic Snapping 則提供另一個思路，但不能未經驗證就拿來修改我們的物理粒子位置。

切面幾何：目前視覺表面主要由粒子重建，尚未採用 BladeMaster 的切割感知幾何重建機制。

物理切斷：Law 8 在約 30–35° 斜切、45° rake，以及部分連續切割案例仍存在未分離區域。

物理精度：切割結果尚未證明解析度收斂，材料參數也尚未經過實驗校準。

工作台配置：需要改成自然的木板與桌面接觸，同時處理刀刃、木板、材料之間的碰撞。

泛化能力：要從固定 Demo 和有限種子測試，提升為明確定義任務分布、失敗邊界與樣本外評估的研究環境。

參考影像（本機，未納入 repo）：`/home/ningan/a.png`（刀具外觀，ADR 0013）、`/home/ningan/b.png`（目標場景：完整切割板直接放在桌面、工件在板上、Franka 夾長直刃刀）。
