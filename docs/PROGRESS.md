# Progress — 2026-10-28 Asia/Seoul

## 最新：真實 Isaac runtime 整合與驗收

本輪已在原機器、原 driver 570.211.01 實際執行 Isaac Lab 3.0 GA 與
Isaac Sim 6.2／Kit110.4；使用官方支援的 Torch2.11/cu128 隔離環境。
先前將完整 runtime 一概列為 CUDA13 BLOCKED 的判斷已被實測取代。
沒有升級 driver、重開機、修改 RSI 或其他專案環境，也沒有訓練任何模型。

| 實際路線 | 完成內容 | 驗收邊界 |
|---|---|---|
| Live Lab | 真實 DirectRLEnv／InteractiveScene／NewtonManager；400steps、8s；實際雙 Franka、固定刀具、TLMPM 與反作用回饋；1.656g切屑、兩個機械分量 | Kit-less 相機是明確標記的全場景 ideal Warp RGB-D；不算 native D455 |
| Live Kit | Sim6.2原生 PhysX；400控制／4000剛體steps、8s；兩路native SingleView D455＋1280×960 Global、永久分離、實際相機資料及影片 | 完成 episode 與全部18項驗收分開；不宣稱 full integration PASS |
| 軌跡修正 | 可選 joint-velocity feedforward；同刀路／材料／gains；Lab active max0.507mm、Kit0.480mm，通過1mm門檻 | 原控制器1.24–1.26mm失敗保留；改進追蹤不自動代表耦合能量或整平通過 |
| Native depth | OFF／ON與開發／held-out控制事前固定；實際OVRTX和實際Kit的取樣對應各自通過 | RAW Z完整保留；不是實體D455校準；原H0失敗、噪聲、invalid pixels均保留 |
| 幾何控制 | 已知閉合box／旋轉／解析volume測試，分開指出2mm取樣的16mm等效底座與density0.25表面膨脹 | 不藉縮小渲染表面冒充物理整平；新profile仍須deformation／task驗證 |

目前明確未通過：D0重建表面最大殘留約1.9–2.0mm；FF的20ms延遲耦合介面功
相對殘差超過5%；部分FF撤刀後沒有滿足跨分量再接觸門檻；Kit嚴格camera reset／
逐粒子重現性也失敗。這些保留為 FAIL／NOT_RUN，沒有放寬事前門檻。
Lab初始reset的q／材料／ideal-depth差異為0；完整repeat與原生剛體穿透稽核正在收尾。

主要證據：`artifacts/isaac_lab_d0_001`、`isaac_lab_d0_feedback_off_001`、
`isaac_lab_d0_feedforward_001`、`isaacsim_live_native_d0_001`、
`isaacsim_live_native_d0_ff_001`、`single_view_kit_audit_003`。
詳細見 [ISAAC_LAB_INTEGRATION](ISAAC_LAB_INTEGRATION.md)、
[ISAAC_SIM_VALIDATION](ISAAC_SIM_VALIDATION.md)、[CAMERA_VALIDATION](CAMERA_VALIDATION.md)。

使用者新增的下一階段為 CUT-01 長條完整切斷 → TRIM-01 長條單凸起整平 →
隨機化 → 回歸測試。會在目前整合證據與未解決問題保存後，以共用 Task／
Geometry／Fixture、workpiece-local目標及統一API接續；原D0–D3基線保留。

## 先前 standalone 基線與歷史紀錄

本輪已沿 Physical Cutting → Franka Control → RGB-D → Simulation Benchmark
實際執行並整合，交付可接續研究的 standalone 環境。使用者授權 public
GitHub、持續 commit/push、agents，以及合理調整原報告的階段門檻。數值與
能力狀態不因門檻調整而改寫；未修改 RSI、其他專案、driver 或訓練 Policy。

| Package | 已交付／實測 | 仍待驗證 |
|---|---|---|
| WP-00 | 獨立 repo、環境／版本／官方 source pins、官方 standalone Newton smoke、CPU CI；本輪另完成真正Lab／Kit live D0 | 歷史四個source-only失敗保留；cu130與本輪cu128相容性分開 |
| WP-01 | 連續基座＋凸起、形狀／seed、Newton MPM 靜置／壓入／刮推、治具、trace／影片 | 材料未校準；完整收斂待完成 |
| WP-02 | CRESSim 七項源碼修正＋16個 actual-header controls及GPU穩定性；新增獨立 Warp TLMPM online persistent cut | CRESSim 無持久分離；自研 TLMPM 為未校準原型，非 BladeMaster 官方程式 |
| WP-03 | Current-mechanical-component ParticleSurface、live 全場景理想雙 RGB-D；本輪另完成native Kit管線與held-out取樣對應 | 硬體噪聲校準、嚴格reset與完整動態depth一致性仍分別驗收 |
| WP-04 | 雙 Panda 動力學、IK／joint／Cartesian、碰撞、固定同幾何楔形刀；材料反作用實際回饋 |20 ms explicit delay，未做單體式或耦合能量收斂；第二臂停泊 |
| WP-05 | ResearchCuttingEnv reset/step/observe/render/close，400 steps完整測試；D0–D3四個物理＋影像 benchmark | 多凸起固定路徑仍有殘留；D3部分參數未接入；沒有學習或任務成功率宣稱 |
| WP-06 | 程式、命令、精簡hash證據、數值trace、影片路徑、負結果、CI與本文件 | 大型raw artifacts保留本機；完整科學benchmark驗收尚未完成 |

可用能力：`persistent_cutting_D0=PASS_EXPERIMENTAL`、`articulated_control=PASS`、
`coupled_cutting_D0=PASS_EXPERIMENTAL`、`live_ideal_rgbd=PASS`、`standalone_rtx_rgbd=PASS`、
`benchmark_executable=true`。原始完整科學旗標 `PHYSICS_PROVEN / VISION_PROVEN /
WORKCELL_PROVEN / BENCHMARK_READY` 仍不宣稱全部通過；它們不等於目前環境不可使用。

## 實測重點與正式邊界

- Frozen TLMPM matrix：11個完成的物理案例＋保留的 setup failure；單刀、撤刀、獨立切屑運動及再接觸有工程證據。19,340粒子、單刀140粒子／1.680 g切屑；七個實際 GPU 數學／局部 impulse controls通過。
- 同seed重跑位置最大差10.59 µm、速度0.00452 m/s，符合事前工程容差；不是 bitwise deterministic，曾有一個中間frame的單粒子label時機不同。最後labels/components相同。
- Frozen第二刀為有限局部切口，沒有額外斷開，T6 FAIL原樣保留。後續改為有平坦剛性支撐、移除邊牆且跨完整截面的交叉刀路：1→2→3分量、4.5 s、最低detF0.9027、既有digits不改寫。兩個失敗free-specimen控制仍保留；這不是雙方向Franka驗證。
- 完整耦合 D0：8 s、1.656 g實際切屑、最大材料反作用1.041 N；ON/OFF回饋對照造成刀具0.345 mm差異，確認反作用會影響機器人。20 ms延遲；介面功殘差0.326 mJ只是診斷，非全能量收斂。
- Live API：400steps／8 s、164.55 s wall、每步雙640×480RGB-D及量測proprio；reset初始q/depth誤差0、截斷通過，觀測無force／oracle／global。
- Native RTX coupled002：41個目前機械表面、401個量測robot poses；切屑可見於兩路depth；mesh-depth QA通過。完整 native D455後處理與理想RGB-D分別記錄，未將fallback冒充感測器驗證。
- Native SingleView fresh斜面 noise OFF／ON：保留實際RAW depth的註冊helper已實作；p95 1.309/1.344 mm超過事前1 mm門檻，主要噪聲變化在valid mask，`simulated_d455`維持BLOCKED/prototype，見[SINGLE_VIEW_REGISTRATION](SINGLE_VIEW_REGISTRATION.md)。
- D0–D3 seed42：四個8 s物理＋5幀全場景理想RGB-D case皆PASS_EXECUTION；D2/D3殘留高度proxy21.7/16.0 mm，不算任務成功。
- 4/3/2 mm與半dt探查顯示chip mass仍變化，尚未收斂；coarse recontact overlap、J2材料參數及sharp-blade模型限制已明列。
- 乾淨CPU-only隔離環境（無Newton/Warp/CUDA）93項測試通過；實際GPU／RTX／控制與CPU tests分別記錄。

## 主要入口與本機證據

```bash
.venv/bin/python -m robotic_cutting_sim.cli research-demo --output-dir artifacts/fresh_live
.venv/bin/python -m robotic_cutting_sim.cli benchmark --output-dir artifacts/fresh_benchmark
.venv/bin/python -m pytest -q
```

| Evidence | 本機 artifact |
|---|---|
| Frozen physics／repeat／refinement | `artifacts/TLMPM_validation_001`、`TLMPM_validation_two_cuts_001` |
| 完整截面第二刀 | `artifacts/tlmpm_supported_cross_cuts_001` |
| 實際機器人反作用耦合＋對照 | `artifacts/coupled_robot_cut_002` |
| Live API | `artifacts/research_env_full_001`，`research_env_demo.mp4` |
| Actual cutting RTX | `artifacts/ovrtx_coupled_002`、`ovrtx_coupled_closeup_002`及`coupled_rgbd_qa_002` |
| D0–D3 benchmark | `artifacts/simulation_benchmark_001` |
| CRESSim修正前後 | `artifacts/CRESSim_candidate`、`cressim_scrape_promoted_001`；歷史FAIL不刪除 |

詳細命令／hash／負結果見 [TLMPM_VALIDATION](reports/TLMPM_VALIDATION.md)、
[COUPLED_ROBOT_VALIDATION](COUPLED_ROBOT_VALIDATION.md)、[WP-03-cutting](reports/WP-03-cutting.md)、
[RESEARCH_ENVIRONMENT](RESEARCH_ENVIRONMENT.md)、[BENCHMARK_REPORT](BENCHMARK_REPORT.md)。
Public repo：https://github.com/AnsonNing/RoboticCuttingSim 。大型資料與官方資產不提交 Git。
