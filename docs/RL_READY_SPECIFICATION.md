# RL-Ready Simulator & Rubber Physics — specification (faithful copy)

This file is the working specification for the RL-ready upgrade and the rubber
material work, handed over by the user on 2026-11-10. It reproduces the user's two
documents — *Implementation Plan v2.0* and *Execution Brief v2* — verbatim except
for one mechanical change required by this repository's naming rule
(`tests/test_repo_naming.py`): the user's separate research/optimizer project is
called **the external agentic optimizer** (外部 agentic optimizer), and its tools are
described generically (its research tree, its autonomous environment-engineering
tools, its external-simulator evaluation with a frozen checkout). The plan §7 status word that names that project is spelled
`MERGED_RCS_PENDING_OPTIMIZER_SYNC` in this repository (the external repository
uses the exact original word); the other status words are unchanged. Nothing else
was edited, reordered or summarised. The verbatim originals are kept outside the
repository (runner directory of the unattended session); their SHA256 are
recorded below so that the copy can be checked against them.

| Original | SHA256 |
| --- | --- |
| Implementation Plan v2.0 | `2816c835300681a3e484ea75bbea1649a6c035b8fd583a74f35d84cb146caac3` |
| Execution Brief v2 | `4b859ded64f350de0c57126b17d81d78323105d896d127508a22708080a88768` |

User instructions at hand-over (2026-11-10), not part of the two documents:
implement both documents in an unattended Claude Code CLI session; complex
decisions, reviews and analysis at **xhigh** effort, implementation and testing at
**high** effort; commit and push continuously. Work packages, acceptance per work
package, measurement boundaries and decisions are in
[RL_READY_PLAN.md](RL_READY_PLAN.md); progress in
[RL_READY_PROGRESS.md](RL_READY_PROGRESS.md).

---

# Part A — Implementation Plan v2.0

RoboticCuttingSim — RL-Ready Simulator & Rubber Physics
Implementation Plan v2.0｜取代先前三階段規劃
日期：2026-11-10
狀態：實作規劃；尚未執行程式碼變更或 GPU 實驗
主要 Repository：RoboticCuttingSim
研究／自主演化 Repository：外部 agentic optimizer repository（the external agentic optimizer repository）
本版最重要的修正：不是新增獨立的 RL 專用 toy simulator，也不是只讓 外部 agentic optimizer 自己有一條快速路徑。整個 RoboticCuttingSim 要升級成 RL-ready 的可擴展模擬平台；其上既有 CUT/TRIM、機器人、感測、物理與未來橡膠任務應共用同一套核心架構。第二個工作主軸是在此平台內建構可信的橡膠物理。此輪不開發刀切／剪刀新任務、不訓練 cutting policy。

1. 目標與非目標
1.1 最終交付定義
1. RCS 是 RL-ready 模擬器：支持 GPU batched simulation、多個互相隔離的環境、vectorized reset/step、可重現的 seed、GPU-native observation/action、reward/termination、任務與材質隨機化、長期無洩漏 rollout。
2. RCS 仍然是完整的 Cutting Simulator：原始 Native Isaac Sim、Franka、TLMPM／物理接觸、CUT/TRIM、相機、切割歷史、正式 evaluator 都保留。RL-ready 是 RCS 核心能力升級，不是把原始功能另存一份或換成不能切的假物理。
3. RCS 內建 Rubber Material Physics：hyperelasticity、viscoelasticity、dynamic deformation、contact/friction、材料校準、精度／收斂驗證；可以直接在原有單環境與向量化多環境內使用。
4. 外部 agentic optimizer 是可選的自主優化層：外部 agentic optimizer 可分析性能瓶頸、修改 RCS 的候選分支、執行測試與提出更新。所有正式 simulator 改動應回寫 RCS，之後 外部 agentic optimizer pin 到 RCS 的新 commit；不維護隱形 fork。
5. RL-grade ≠ 已完成 RL policy 訓練：本輪只建置可被 PPO／SAC／VLA／模仿學習等使用的環境與吞吐量；可做隨機 policy、短 benchmark 或 minimal pipeline smoke，但不要求訓練成果。
1.2 目前明確不做
- 不新增橡膠刀切、剪刀、裂紋／材料分離任務；原本 CUT/TRIM 的回歸必須保留。
- 不實作高速移動條材剪切策略、不訓練 agentic control Cutting Agent／RL cutting policy。
- 不另外建立一個與 RCS 分離的專用 RL simulator repository。
- 不為了宣稱 RL-ready 而默默跳過接觸、材質力學、切割判定或降低解析度；簡化路線必須獨立標記與測試。
- 不建立新的 agent framework、資料庫或 dashboard；沿用 RCS benchmark 與 外部 agentic optimizer 的研究工具。
2. 目前程式碼狀態與需要彌補的能力差距
閱讀 RCS 主分支後，已存在：
現有能力	目前情況	需要升級
src/robotic_cutting_sim/gym_env.py	CuttingGymEnv 已提供單環境 Gymnasium wrapper、state/vision observation	現有 wrapper 並不等於 batched GPU RL 環境；加入 vectorized training adapter
src/robotic_cutting_sim/policy_env.py	PolicyReadyEnv 有 action validation、50 Hz step、reset、deployable/privileged 分流	向量化 action／observation、每環境 termination、局部 reset、無逐環境 CPU round-trip
src/robotic_cutting_sim/isaac_lab_env.py	真正繼承 Isaac Lab DirectRLEnv，但建立的 scene num_envs=1；material 仍由外部 TLMPM 驅動	實際擴充多個獨立場景與 material instances；不能只把 num_envs 數字改大
src/robotic_cutting_sim/live_material.py	TLMPM solver 與 rigid robot 的力回饋整合	需可索引每個 environment 的 material state、reaction wrench、history
src/robotic_cutting_sim/profiling.py	已有 stage/kernel profiling	建立 vectorized throughput、reset、memory、GPU occupancy/scaling suite
Native Isaac Sim 6.2	原有完整機器人／相機／剪切顯示與數據輸出	保留 reference/full-validation path，對比新的 vectorized headless path
外部 agentic optimizer 的 external-simulator 評估／自主環境工程工具	可把 RCS 當 external sim 運行正式評估	為 RCS 優化建立隔離 candidate → frozen tests → RCS PR → 外部 agentic optimizer pin 同步


重要判斷：現在已有一些「可供 policy 使用的 API」，但不足以宣稱達到高吞吐量、多環境並行的 RL-ready。關鍵升級在 material/robot batching、局部 reset、GPU memory/layout、觀測和正式評估契約，而不是再加一層 API。
2.1 目前速度基線
已知 WP17 測得兩個完整 Native 24s episode wall：seed 512 約 375.653 秒、seed 15 約 390.306 秒；兩場合計 765.959 秒。配置為每集 1200 個 50 Hz 控制步、TLMPM 1 mm particles / 4 mm grid / 0.1 ms dt。這是包含 Native 啟動、相機與影片輸出的 single-env baseline，不可直接拿去和純 headless batched step latency 宣稱 apples-to-apples speedup。

WP16/WP17 先前已改善 contact atomics、device diagnostic、active frontier。WP17 的 seed 512 最終 cut-code 與 WP16 存在差異；物理等價仍有未解問題。因此所有新的性能報告必須同時呈現物理差異與 workload 差異。

參考：
- WP17
- WP16
- Policy-ready API
- Isaac Lab integration
3. 系統架構：一個 RCS，兩種執行模式
                         RoboticCuttingSim repository
              ┌────────────────────────────────────────┐
              │ Scene / Robot / Material / Task / Goal   │
              │ Physics Backend + Contact + Evaluation   │
              │ Stable action/obs/episode contracts      │
              └────────────────────────────────────────┘
                          │                    │
          ┌───────────────┴──────┐  ┌──────────┴────────────┐
          │ RL_GPU_HEADLESS      │  │ FULL_NATIVE_REFERENCE │
          │ Many envs / 1 GPU    │  │ Full sensors / physics│
          │ Batched kernels      │  │ Visualization / audit │
          │ Partial reset        │  │ High-fidelity checks  │
          │ Torch action / obs   │  │ Original CUT/TRIM      │
          └───────────────┬──────┘  └──────────┬────────────┘
                          │                    │
                RL rollouts/benchmarks    physical cross-check
                          └──────────┬─────────┘
                                 physics gates
                                       │
                          agentic optimizer research
                                       │
                     validated change → RCS PR → merge
                                       │
                         optimizer env descriptor pin update
兩個 runtime profiles 同屬 RCS repo，重用相同的 scene/task/material/evaluation 定義。不要求 Native RTX/GUI 把數百個場景都即時渲染；高保真 Native 是完整視覺和物理參考，GPU headless 是可擴展訓練執行路徑。兩者必須以可測的 parity tests 保持可用的物理契約；無法一致的能力要列明限制。

不能預設「headless ＝低保真」：headless 表示不需要 GUI／presentation render；只有確實改了物理模型、解算精度或觀測才標 reduced_fidelity。
3.1 最小 API 契約
RCS 應提供 GPU-first VectorEnv（API 語義示意，實際名稱可依專案現況調整）：
vec = open_vector_env(
    num_envs=16,
    runtime="rl_gpu_headless",
    task="existing_cut_or_trim",
    material="existing_profile",
    observation_profile="state",
    device="cuda",
)
obs, info = vec.reset(seeds=seed_tensor)                # 各環境 seed 獨立
obs, reward, terminated, truncated, info = vec.step(     # 一次提交 N 個 action
    action_tensor  # tensor [num_envs, action_dim]
)
obs, info = vec.reset(env_ids=done_ids)                  # 僅重設完成的環境
硬性 API 要求：
- action、proprio／state observation、reward、done mask 以 batch-first 形式提供；儘量留在 GPU／Torch 或明確支持的零拷貝資料路徑。
- 每個 env 有自己的 material state、robot state、contact history、seed、時計、cut history、goal、reward 和 termination。
- reset(env_ids) 只重設指定索引；其他 env 的物理狀態／時計不受影響。
- info 使用結構化、可批次回傳的輕量 metrics；除錯與高成本 oracle 僅在正式需要時讀取。
- state policy 不應被迫初始化 RTX cameras；vision policy 要有具體 camera source、timestamp、render frequency 和 latency。
- 不依賴每個 env 一個 Isaac Sim / Python process；原則為單一 GPU 進程內的多 env batched simulation。
- Gymnasium single-env interface 保留；提供 Isaac Lab compatible 的 batched runner，必要時再提供 TensorDict/RL-library adapter。
- 載入不支援向量化的場景/solver 時，必須明確報 UNSUPPORTED，不能偷偷循環呼叫 N 個慢速 Native 實例卻稱為 GPU-vectorized。
3.2 原有功能覆蓋要求
RCS 原有功能仍必須受測：CUT、TRIM、multi-cut/切割歷史、雙 Franka/physical grasp、contact + rigid coupling、scene randomization、camera/depth、原有 evaluation。第一版可先用精簡場景驗證 batched 架構，但正式宣稱 RCS RL-ready 前，現有代表性 CUT/TRIM 任務需在 vectorized 路徑實際運行。功能未完成的部分需做 feature-parity matrix，不能只靠一個無工具的橡膠方塊 pass 就宣稱整個 RCS RL-ready。
Phase 1 — Whole RoboticCuttingSim RL-Ready Upgrade
4. Phase 1 驗收指標
重點是 aggregate simulator throughput，而非只追求單次 Native episode 加速。
指標	定義與評量方法
Aggregate environment steps/s	已完成的所有 env control steps / wall seconds；單位明確，warm-stepping 不計 cold init
Physics simulated time / wall time	N × per-env physical seconds / total wall seconds；與單 env RTF 分開
Scaling N	比較 N={1,4,16,32,64,128} 等可行配置；不支持時記錄 VRAM/solver 限制，不捏造達標
Reset throughput	full reset 與 partial reset(env_ids) 的平均、P95，是否干擾其他 env
Memory	per-env VRAM、host RAM、peak、長時間 rollout 的成長率、OOM/exit-137
Training interface	PyTorch tensors、固定 shape/metadata、action/reward/done、無不必要 CPU→GPU→CPU copying
Physical parity	單 env vs batched 相同 seed/設定的物理差異、CUT/TRIM 分離、contact 和切割歷史
End-to-end Native	原有完整執行 wall 與畫面/紀錄，不和 headless 純 step 混成一個 speedup 數字


階段性工程目標（非既定能力）：
- M0：`N=1` 的新 RL runtime 可完整 step/reset，物理與 reference 設有明確 parity 範圍。
- M1：`N=4` 同 GPU 並行；env 獨立與局部 reset PASS。
- M2：至少 N=8–16 可連續 rollout，並有實測相對 N=1 的吞吐提升；這是功能性 RL-ready 的最低批次證據。
- M3：爭取 N≥32；64/128 為硬體和材料負載許可的擴展研究目標。100/1000+ steps/s 只能寫成可測的長期 performance aspiration，不能在高精度 MPM 未測前保證。
- 所有 N 都需報粒子數、material dt、substeps、robot DOF、接觸、相機、recording、batch shape；不允許以簡化工作負載取代正式配置而不註記。
P1-0 — 現況診斷與正式效能基準（先做）
RCS：
1. 核對 Git HEAD、Python/Warp/Newton/Isaac Sim/Isaac Lab、GPU 驅動和可用記憶體。
2. 複用 profiling.py 和現有 benchmark。分測 material solver、frontier/contacts、robot coupling、CUDA Graph、CPU sync、render/surface、reset。
3. 建 configs/rl_benchmark/（建議）標準 workloads：A physics-only material；B robot+material+contact；C 原有 CUT；D 原有 TRIM；E dynamic rubber（Phase 2 加入）。
4. 建 single_env_native, single_env_headless, batched_headless 三種不可混淆的測量邊界，輸出 JSON+MD 概覽。
5. 建與現有 v5/v7/v7_2 CUT/TRIM 物理結果相比的 frozen regression cases；先保留 WP17 已知 cut-code 差異為未解問題。
交付：可重現 benchmark runner、log/artifact、performance+physics baseline，不新增重複 profiler。
P1-1 — GPU batched physics：核心架構升級
這是最大的技術工作包，不可跳過。
1. 選擇現有 Isaac Lab 3.0 DirectRLEnv/Newton 路線作為最小侵入的首選，但要讀實際 backend 原始碼並確認 deformable/MPM 的多 env 支援。若內建不支持，明確設計 batched Warp TLMPM adapter；不預設上游會免費替 RCS batch。
2. 將 TLMPM 的 particles、velocities、deformation、material params、grid、cut codes/history、tool pose、reaction wrench 轉為per-env indexed batch，使用 [N,P,...] 或扁平 buffer + offsets／mask 的合理布局。
3. 每個 env 的 hash/contact/grid、cut history、material sample、random generator 必須正確隔離；GPU kernel 跨 env 並行，不能污染鄰近環境。
4. scene 中已有兩個 Franka、physical grasp 與力回饋。從最小單臂測試開始，但最後要回到代表性的雙臂／現有 CUT/TRIM 配置；單臂 toy smoke 不算 whole-RCS ready。
5. 強制遵守穩定的 control / material / rigid dt 契約，保留目前雙向或延遲力耦合的時間語義。對跨 env robot/material 力回饋做測試。
6. 對不同幾何／粒子數使用已驗證的 packing / buckets，避免 max_particles × N 的誇張空間浪費；保留相同物理模型下的合理精度。
7. 既有單環境 API 可以當 reference wrapper；不要為了 batched 的 convenience 改掉已驗證 frozen profile。
交付：實際 batched MPM/robot/material step；不靠 Python for-loop 逐環境執行單一 RCS episode；可於 profiler 看見真實 GPU batch gains。
P1-2 — Vectorized Environment API + Fast/Partial Reset
- 實作 batched reset, step, observe, evaluate，per-env terminated/truncated。
- reset(env_ids) 恢復該 env 的 material particles、grid、cut history、tool、robot、contacts、goal、time、random state；未選 env bitwise/數值差異符合單獨 step 的 reference。
- 支援 state observation 且 GPU-native；vision observation 視需要採 batched GPU rendering，保留相機外參和 time stamps。
- policy 不應拿到 oracle material particles、cut codes、真實 contact truth（如果只有 simulated force sensor 真正可部署才允許）。
- 與現有 Gymnasium adapter 及 Isaac Lab DirectRLEnv lifecycle 對齊；清楚區分單 env reset() 與 batched partial reset。
- reward/termination 在 batch 上計算，與原有 frozen goal evaluator 核對；若昂貴 evaluator 不適合每個 step，可做經驗證的緊湊增量 metrics，最後一樣用獨立完整 evaluator。
交付：可直接接 RL library 的 batched observation/action API、可單獨 reset done env，沒有 shape/dtype/device 混亂。
P1-3 — Headless Runtime／Kernel／Data Movement 最佳化
- 本階段的 headless 屬於 RCS 的正式執行模式，不是旁邊的外部實驗 simulator。
- 保留現有 CUDA Graph 的優勢，在 tensor shapes/scene groups 固定時研究 graph replay；避免每 step 重新編譯。
- 減少 CPU–GPU synchronization、per-env Python callbacks、頻繁 geometry upload／particle snapshots。
- kernel 優化候選：spatial hash、particle neighbor/contact、active frontier、conservative broad-phase、GPU reductions、preallocated memory；候選需真實 hotspot profiling。
- camera frequency/recording frequency 獨立於 physics frequency；state-only 可以不產生影像；vision profile 要實測 rendering。
- 不降低正式 physics fidelity；任何 coarse/fused numerical approximation 需獨立 profile 與偏差報告。
交付：各 runtime 的 stage breakdown、性能改善、完整物理驗證及 rollback。
P1-4 — 記憶體、穩定性與長時間 RL Rollout
- GPU buffers 應重用；不同 env reset 不應反覆建/dealloc 大量資料。
- 多次 episode 與 partial reset 下維持 bounded VRAM/RSS；保留 Kit reload_in_place 的歷史已知限制，尤其 Native camera-off crash 的先前失敗紀錄。
- 單一進程支援長串 rollout；加入可靠的 counter、GPU memory telemetry、異常 episode 單獨回報與 recorder streaming。
- 將 OOM、Kernel error、runtime crash 分成 infrastructure failure，不算 RL task failure。
- 以多個 seeds、重置次數、幾何變體測試 buffer/graph state 的安全性。
交付：長時間 vectorized rollout smoke、8+ serial full episodes、reset memory slope、失敗恢復／完整資料收據。
P1-5 — Whole-RCS Feature Parity & RL Integration Acceptance
能力	RL_GPU_HEADLESS	FULL_NATIVE_REFERENCE	驗收
原有 scene/goal/material schema	必須	保留	相同 task 可編譯
既有 CUT / TRIM representative tasks	必須能完成核心物理任務	保留	independent physical evaluator
Robot articulation/contact	必須真實執行	保留	grasp/coupling/contacts
CUT history / persistent severing	必須與 solver 契約一致	保留	bridge/components/history
Camera RGB-D（有要求時）	視覺 profile 支援並標明來源	Native 正式視角	實際時間戳及 camera calibration
Vectorized reset/step	核心必需	不要求 GUI 數百場景並行	N=1/4/16 實測
GPU-native action/obs/reward	核心必需	不要求	torch device／shape
Seeded randomization	必須	保留	獨立/重現
性能 profiling	必須	必須	不混淆工作量
Rubber material（Phase 2）	必須加入同一介面	必須能實驗／檢視	physics calibration/parity


最低 RL 訓練相容性煙霧測試：用隨機或確定性 policy 驅動 N 個環境，重複 partial reset、保存觀測／行動／reward/done，確認可被常見 RL loop 消費。此處只做 pipeline smoke，不需真正完成 PPO 或 Cutting 訓練。
P1-6 — 外部 agentic optimizer 自主改善 RCS 效能（可與 P1-1～P1-4 迭代）
外部 agentic optimizer 使用既有研究樹、自主環境工程工具、external-simulator 評估、GPU job、formal；新增最薄的 performance worktree / proposal / benchmark / PR adapter：
外部 agentic optimizer studies RCS profiling
    → decides bottleneck
    → edits isolated RCS worktree
    → CPU + GPU + physics regression
    → frozen RCS candidate commit
    → independent full performance/physics acceptance
    ├─ FAIL: record diagnostic; RCS main untouched
    └─ PASS: create RCS PR with evidence
                   → merge to canonical RCS
                   → update 外部 agentic optimizer RCS commit pin
                   → smoke/contract test
                   → SYNCED
- RCS 是 simulator source of truth；所有 kernel/runtime/material/API 改進必須保存在 RCS。
- 外部 agentic optimizer 可以讓 agent 自主研究，而非人預設全部超參數；但不能透過改 benchmark/evaluator 取得假加速。
- external-simulator 評估 既有 frozen checkout 是正式評估保護；候選 worktree 與 frozen evaluator checkout 分離。
- 跨 repo 非原子：RCS 已 merge 而 外部 agentic optimizer pin 更新未完成時記 MERGED_RCS_PENDING_OPTIMIZER_SYNC；不得宣稱已完全同步。
- 合併依 GitHub repo 權限／PR rules；不繞過 branch protections，不把開了 PR 當成程式已 merge。
- RCS 必須可獨立安裝／執行；不依賴 外部 agentic optimizer 的私有路徑或每次透過 agent 才能運行。
Phase 1 Done 判定
- [ ] N=1、4、至少 8–16 env 的真正 GPU batched 執行與 partial reset，出具 throughput / memory / step latency。
- [ ] RCS 原有代表性 CUT/TRIM 在 RL headless 物理模式實跑成功／失敗評估；缺口有明確狀態，不能假稱全部已完成。
- [ ] 單 env reference 與 batched parity 的數值／切割歷史回歸測試，以及舊 frozen profile regression。
- [ ] GPU-native action/observation/reward/done、環境獨立 seed、可重現場景、API tests。
- [ ] 長時間 rollout、OOM/記憶體／reset 壓力測試，真實 GPU telemetry。
- [ ] Headless、視覺／非視覺及 full Native workload 各自獨立報 speed；沒有作假性能比較。
- [ ] 外部 agentic optimizer 若參與優化，正式通過的成果已回寫 RCS 並完成版本同步；若只開 PR 標 PENDING。
Phase 2 — Rubber Material Physics inside RL-Ready RCS
Phase 2 的材料模型直接進入同一個 RCS 材料系統與 batched physics 介面，不能另外做只有單環境才可用的橡膠 demo。
5. 材料模型基本要求
使用者提供的圖片看起來是可大幅彎曲的黑色長條，但僅靠影像無法確定材料是天然橡膠、EPDM、TPE、矽膠或其他聚合物。RCS 既有 elastic_control.yaml 的未校準參數不是橡膠真實 ground truth。

本階段研究：
1. Hyperelasticity：大應變、彈性回復、近不可壓縮。先以 compressible Neo-Hookean 完成基線，再按資料選 Mooney–Rivlin／Ogden 等。
2. Viscoelasticity：依速率變化的應力、stress relaxation、creep、loading–unloading hysteresis，選合適的 Prony/Maxwell 或其他模型。
3. Contact/Friction：橡膠與桌面／夾具及自身接觸、滑移、碰撞、阻尼；不要把阻尼調大就當成黏彈性校準。
4. Dynamic fidelity：自由彎曲、重力下垂、外力釋放後回彈與振動、拉伸卸載、摩擦滑動。
5. GPU efficiency：新增材料的運算必須兼容 single-env reference 和 vectorized RL headless，並量測 throughput 降幅。
5.1 Solver 選型
可考慮：
- 擴充 TLMPM：延用現有粒子／grid、雙向 contact/robot coupling、既有任務契約，但必須實作適用的大變形彈性／黏彈性本構；既有 clay-like 切割 solver 不自動成為橡膠模型。
- Newton VBD／FEM：做隔離小型 benchmark；評估大變形、接觸、GPU batching、材料參數可辨識、與既有 RCS robot coupling 的整合成本。
- 若某條路線只提供快速位移動畫、沒有可靠接觸／回彈，就不得標為完整材料物理。允許 外部 agentic optimizer 根據實測結果選擇實作方法，不預先強制重寫整個求解器。
兩者可共存為 opt-in material backends；保留 RCS 原有 v5/v7 系列與 CUT/TRIM。
6. Phase 2 工作包
P2-0 — Rubber Material & Calibration Dataset Interface
- 定義 rubber_material_v1 recipe：密度、材質模型、彈性參數、黏彈性參數、friction/contact、數值範圍及校準狀態。
- 設計可讀入實驗資料的簡單 JSON/CSV manifest；材料參數與 SI 單位統一。
- 實驗資料至少需要（取得時）：尺寸/質量、拉伸／卸載的力位移曲線、不同速率、懸臂回彈時間影片、接觸滑動／摩擦，以及相應邊界條件。
- 若沒有實物數據，標 SIMULATOR_ONLY_UNCALIBRATED，仍可跑數值驗證；不能宣稱 real-world calibrated。
P2-1 — Rubber Deformation（無刀具）
新增 RUBBER-PHYS-01 等物理小測試：
- 靜態重力下垂、cantilever bending；
- 單軸拉伸、釋放後恢復；
- 不同厚度／長度／截面；
- 能觀察 material surface、幾何 deformation、time history；
- 不強加切割／裂紋分離。
必須可在 RCS single-env reference 與 batched N>1 路徑運作；同 seed 有可測差異範圍。
P2-2 — Hyperelasticity
- 選擇 energy-based 本構，測 stress–stretch、volume ratio J=det(F)、large deformation、loading/unloading、elastic recovery。
- 對 near-incompressibility 做合適數值處理；不能簡單把 ν 設到 0.499 然後忽略數值鎖死或不穩定。
- 與可用 analytic/simplified reference 及 dt/spacing refinement 比較。
- 提供 batch GPU kernel profiling，避免新 constitutive model 破壞 Phase 1 性能架構。
P2-3 — Viscoelasticity & Dynamic Response
- 速度依賴與內部狀態演化：stress relaxation、creep/hysteresis、短暫振動、回彈；
- 用多種載入速度做 validation，單一速率擬合不是泛化；
- 可校準 relaxation time／viscous parameters，保持不同 dt 下時間尺度一致。
- partial reset 必須重設每個 env 的 viscoelastic internal variables，避免上一集殘留材料記憶。
P2-4 — Contact / Friction / Dynamic Coupling
- 橡膠－桌面、橡膠－夾具、橡膠－機器人手爪、必要時自接觸；
- 測 contact penetration、slip onset、反作用力、能量耗散與穩定性；
- 既有 20 ms force handoff 有耦合延遲，必須評估動態橡膠在高速運動中是否可接受；不能把耦合誤差當黏彈阻尼。
- GPU batched 時檢查 per-env wrench/state isolation，不可資料串流污染。
P2-5 — Calibration + Generalization
以實際材料資料建立可重現參數擬合與獨立測試：
- 訓練／擬合：部分拉伸速度、幾何和影片；
- held-out：不同載入率、長度、厚度、起始變形；
- 指標：force-displacement RMSE/relative error、端點軌跡、回彈時間、振動衰減、slip distance、體積與能量誤差。
- 允許 外部 agentic optimizer 自主選擇 CMA-ES、Bayesian Optimization、其他參數辨識方法；不允許為使 RL reward 高而調出與真實材料相反的物理參數。
- 驗收容差要由量測誤差與 pilot 界定，在正式 held-out 前凍結；沒有真實資料的測試只能通過 simulator/numerical validation。
P2-6 — RL-Ready Rubber Physics Acceptance
- 同一個 rubber scene 可以使用 N=1 / 多 env 批次 reset/step，正確對每個 env 隨機化材料、長度、厚度、摩擦、外力與初始形變。
- state observation、action、reward/done（如簡單跟隨變形的 benchmark）完整，但不要求訓練 policy。
- GPU throughput / VRAM / dt/spacing convergence 和 Phase 1 對照。
- 同一材料 recipe 必須能在 Full Native 的觀測／動態驗證中被使用（允許表面處理差異，但不允許未知材料物理差異）。
- 保留 freeze / source SHA / measurement provenance。
Phase 2 Done 判定
- [ ] 新 rubber recipe 與 hyperelastic/viscoelastic/contact 模型有實際可重現運行證據。
- [ ] 大變形、重力、拉伸／釋放、回彈、滯後、摩擦測試在物理與數值層都通過。
- [ ] 同一 rubber material 能在 RCS multi-env RL headless 與 Full Native reference 進行實驗；至少 8 env 的 rubber batch 測試有可靠結果。
- [ ] 如果已有實物資料，獨立 held-out 動態對照達標，才可標 REAL_CALIBRATED；若沒有，狀態保持 SIM_ONLY。
- [ ] 舊 CUT/TRIM/V5/V7 baseline 未受破壞。
- [ ] 材料模型、測試、scene、文件正式合併至 RCS；外部 agentic optimizer pin 更新。
7. 外部 agentic optimizer 與 RCS 之間的自主性能演化、版本同步
不能把 外部 agentic optimizer 內的外部工程候選當作 CuttingSim 正式功能。適用於 Phase 1 的 kernel/runtime、Phase 2 的 material/solver。
步驟	執行位置	必要紀錄
Benchmark/profiling → 優化假說	外部 agentic optimizer research	problem、baseline、預期影響、研究 node
實作候選	RCS 隔離 worktree	patch、candidate commit、數據
快速單元 + GPU dev tests	RCS + 外部 agentic optimizer orchestrator	errors、speed、physics diff
凍結版本 + independent formal	外部 agentic optimizer 沙箱外評估	source hash、held-out results、verdict
提交正式程式碼	RCS branch/PR	變更原因、物理影響、rollback
合併通過	RCS canonical repo	actual merged commit
更新環境版本	外部 agentic optimizer descriptor/manifest	RCS SHA pin、smoke、sync receipt
進入下一輪	外部 agentic optimizer research	已同步的 canonical RCS baseline


錯誤狀態：
- REJECTED_PHYSICS
- REJECTED_PERFORMANCE
- RCS_PR_PENDING
- MERGED_RCS_PENDING_OPTIMIZER_SYNC
- SYNCED
正式驗收不可由正在優化的 agent 修改 evaluators；GPU 保護與執行資源沿用現有環境。 任何需要改變材料語義的優化都用版本化 opt-in profile，通過獨立材料驗證後才考慮推為預設。
8. 實作順序與每輪交付
順序	工作包	檢查成果
1	P1-0 Baseline + RL throughput benchmark	量得到 single、headless、batch 的差距與瓶頸
2	P1-1 真正 batched material + robot physics	N>1 的同 GPU 同進程物理推進，非 N 個完整 Kit process
3	P1-2 VectorEnv + partial reset	Torch action/obs/done、獨立 clocks/seed/state
4	P1-3 Headless + kernel/runtime performance	物理不變式、速度與 VRAM 可比
5	P1-4 Memory / long-running rollout	大量 step/reset 不崩潰、不串狀態
6	P1-5 Original CuttingSim feature parity	CUT/TRIM/robot/contact/sensors／evaluator 實跑
7	P1-6 外部 agentic optimizer 自主優化及 RCS 同步	真實 RCS PR/merge 與 外部 agentic optimizer pin 收據
8	P2-0～P2-2 Rubber baseline + hyperelasticity	在 batched RCS 裡的大變形橡膠
9	P2-3～P2-4 Viscoelasticity + contact	動態回彈、速率依賴、摩擦
10	P2-5～P2-6 Material calibration + RL-compatible validation	同時有精度、穩定性與吞吐量報告


分批實作建議：先 P1-0/1/2 建立可運行的真正 batch baseline；再根據瓶頸優化，而非先把單 env CUDA kernel 最佳化到極致。Rubber 一旦進入 batched 結構，後續更換材料不需要重造 RL 環境。
9. Codex／Claude 的執行約束
1. 實作主體是 RCS。先在 RCS repo 讀最新 code、AGENTS.md、README、既有 Physics/Isaac Lab contracts；再讀 外部 agentic optimizer 的自主環境工程與 external-simulator 評估工具。避免兩個 repo 無必要大重構。
2. 第一輪只做 Phase 1；暫時不要開發 knife/scissors/moving-material cutting 或 policy training。Phase 2 的 material interface 可以先為向量化設計留必要 extension point，但不提前實作材料模型。
3. 先證明 N>1 真正 parallel physics 再優化 throughput；不要只修改 num_envs=1 或逐個 Python loop 呼叫舊環境。
4. 維持兩種 profile：`RL_GPU_HEADLESS` 用於 batched training-ready rollout；`FULL_NATIVE_REFERENCE` 用於完整物理／相機驗證。相同 schema、具體 parity，不一定相同渲染成本。
5. 不宣稱虛假的 RL-grade 數據。需報 N, particles/env, substeps, camera mode, exact hardware, GPU VRAM, control steps/s、warm/cold；物理改動和效能提升必須分別評估。
6. 不破壞舊 CUT/TRIM。若新向量化 backend 初期缺刀具接觸，標 PARTIAL，直到現有代表性切割任務真能在向量化路徑運作才可稱整個 RCS RL-ready。
7. 與橡膠相容：材料 backend 抽象化要允許新增 hyperelastic/viscoelastic internal state，且每個 env reset 能完全清空。
8. 外部 agentic optimizer 只負責自主研究與演化，可自動建立 RCS PR；RCS canonical 合併完成後再更新 外部 agentic optimizer commit pin。不得在 外部 agentic optimizer 保留一套分叉 solver。
9. 實作的每個增量均要有測試、效能數據、物理 regression、source hash、實際 commit/PR、rollback 和已知限制。
10. 不必設過多限制 外部 agentic optimizer 的研究方法；但 physics invariants、held-out 驗證、原有 ABI／API 與來源可信度不可被繞過。
10. 最終交付清單
RCS repo
- Batched GPU physics + RL-ready vectorized env API。
- Torch/Gymnasium/Isaac Lab adapter；完整 reset/step/observe/reward/done。
- Headless training runtime、記憶體／kernel／throughput benchmarking。
- 保留並驗證 CUT/TRIM/physical tool/robot/sensor 的代表性功能。
- Rubber materials、hyperelasticity/viscoelasticity/contact、數值與實物校準測試。
- docs/ 報告／source receipts／versioned profiles／回退方法。
外部 agentic optimizer repo（如使用 外部 agentic optimizer 自主演化）
- RCS 最新正式 commit 的環境 descriptor 與實驗資產。
- 可實際編輯 RCS worktree、凍結版本、驗收、建立 RCS PR 的研究流程。
- 合併成功才更新 pin；完整 sync receipt 與失敗／性能／物理證據。
應產出的測量
- 單 env 與 N=4/16/32/64（可行時）的控制步吞吐量、模擬秒數比、GPU/RSS、reset latency。
- Full Native wall / Physics-only wall / Vectorized throughput 三組數據分開。
- 代表性 CUT/TRIM physics equivalence + regression。
- 橡膠拉伸／回彈／摩擦曲線及誤差（有實物資料時）與 batch 開銷。
- 採用的 RCS commit、外部 agentic optimizer 對應 pin、回退路徑。
總結與執行指令
請把原本的 RoboticCuttingSim 升級成可以進行高吞吐量強化學習的模擬平台，並在同一平台內開發可信的橡膠材料動態物理。先實作模擬性能／多環境並行／RL-ready API，再實作橡膠本構與動態校準。保留既有 CuttingSim 的物理與場景，暫不開發刀切、剪刀任務或訓練控制策略。任何由 外部 agentic optimizer 自主改善的 simulator 程式碼必須在驗證通過後回寫 canonical RCS，再更新 外部 agentic optimizer pin。

---

# Part B — Execution Brief v2

Execution Brief v2 — Upgrade all RoboticCuttingSim to RL-Ready, then Rubber Physics
閱讀主文件：`RoboticCuttingSim_RL_Ready_Rubber_Physics_Implementation_Plan_v2.md`
核心需求（避免理解錯誤）
不是「另外加一個快速 RL simulator」。現有 RoboticCuttingSim 自身必須成為 RL-ready：原有 CUT/TRIM、機器人、材料與接觸 API 能與 GPU batched / vectorized / headless rollout 共存。橡膠模型也必須整合進這個相同的 RCS 系統，在 batch 與 Native reference 路徑均可使用。

只做：
1. Phase 1：RL-Ready Performance：真正 GPU 多環境並行、批次 MPM/contact/robot dynamics、vectorized reset/step、Torch action/obs/reward/done、快速 reset、memory、headless、profiling、full physics regression。
2. Phase 2：Rubber Physics：hyperelastic / viscoelastic / contact、回彈與動態形變、材料校準、single/batched physics parity。
3. 跨 repo 演化：外部 agentic optimizer 可以自主修改 RCS 候選，但通過驗證的程式碼需要正式回寫 RCS；RCS merge 後 外部 agentic optimizer descriptor pin 更新。
暫不做：刀切或剪刀新任務、橡膠裂紋與分離、移動條材高速剪切、agent control 研究、PPO/SAC/VLA 正式訓練、額外 UI、複製一套 RCS simulator 到 外部 agentic optimizer。
先查再改
請檢查兩個 repo 的最新 HEAD、`AGENTS.md`／`CLAUDE.md`、已有測試、runtime 與基準。已知：
- src/robotic_cutting_sim/gym_env.py 是 single-env Gym wrapper，不能以它代表 vectorized GPU RL。
- src/robotic_cutting_sim/isaac_lab_env.py 目前 scene num_envs=1，material 狀態由外部 TLMPM 進行；單純改數字無法實現多環境。
- src/robotic_cutting_sim/policy_env.py 已有 deployable observation / reward / reset 契約，可以重用。
- WP16/17 已有 kernel 最佳化，不能忽視成果；WP17 存在 final cut-code differences，不可無條件宣稱 physics-equivalent。
- 外部 agentic optimizer 已有 research tree / autonomous environment engineering / external-simulator evaluation；要避免另起一套 agent framework。
第一批實作（Phase 1）
1. Baseline：分開測 Full Native、Physics-only headless、不同 N 的 batched control steps/s、VRAM、reset、kernel stage；固定 source/seed/particle/dt。
2. True batch：RCS 中的 material states、grid、contact、cut history、Franka／tool pose、reaction wrenches、clock/seed 狀態以 env id 分離；一個 GPU 進程 batched kernel／solver，不是 N 個 Kit processes。
3. VectorEnv：新增/擴充 open_vector_env(num_envs=N, device='cuda') 形式的 API；`step([N,A])` 和 reset(env_ids=[...])；各 env 有自己的 reward、done、observations；Torch GPU tensors。
4. Physical correctness：同 seed 同 action 對照原有 reference；CUT/TRIM 真實切割、material mass、contacts、cut-code history 不變形、不串 env。
5. Headless/runtime：沒必要時不 render，不每個 step CPU sync；CUDA graph/memory reuse/surface/recording rate 按 profiler 改，不能偷降物理。
6. Stress：N=1、4、8–16 真的並行；N=32/64/128 視 VRAM/physics cost 測 scaling；long rollout + partial reset 無 memory leak。
7. RL smoke：隨機或 scripted actions 跑多個 env，驗證 API 可供 RL 使用；此輪不要求訓練 policy。
8. 外部 agentic optimizer sync：如 外部 agentic optimizer 研究出有效改動，將 RCS candidate frozen → tests → PR → merge to RCS → 外部 agentic optimizer pin。未 merge 不得標 SYNCED。
Phase 2（Phase 1 驗收後）
在同一套 RCS batch／Native 架構中加入 Rubber Material：
- rubber_material_v1，不是拿未校準 elastic_control.yaml 當真實橡膠。
- Hyperelasticity：Neo-Hookean 基線、視需求擴充 Mooney–Rivlin／Ogden；大變形、體積約束。
- Viscoelasticity：速度依賴、relaxation／hysteresis、內部材料記憶的 per-env reset。
- Contact：rubber-table/gripper、滑移／摩擦、動態力回饋。
- 無切割 benchmark：cantilever、stretch/release、gravity sag、oscillation、surface motion。
- 真實實驗資料校準與 held-out speed/geometry；無實物量測時只能標 SIM_ONLY，不能稱 REAL_CALIBRATED。
- 相同材料模型在 batch 和 Full Native 具可測 physics parity，並有 GPU throughput/VRAM 回歸。
回報內容
請提供每一輪的 RCS changed files、commit/PR、真實執行指令、已通過/未執行測試、exact configurations，以及性能表：

N | particles/env | control Hz | material dt | cameras | env-steps/s | aggregate sim-time/wall | VRAM peak | reset p95 | physics verdict

另提供 外部 agentic optimizer 更新的 descriptor pin、sync status，說明下一個真實瓶頸與未解差異。所有結果需以真正執行為準；不能從 DirectRLEnv 名稱推定 RL-ready，也不能只用慢速單環境 Python loop 假裝 vectorization。

第一輪先完成可執行的 P1-0、P1-1、P1-2 最小閉環，接著才做大型 kernel 優化。
