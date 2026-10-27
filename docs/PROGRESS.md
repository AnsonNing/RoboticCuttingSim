# Progress — 2026-10-28 Asia/Seoul

本輪目標是交付可接續研究的場景與環境原型。使用者已授權建立public GitHub repo、持續commit、使用協作agents，並允許放寬原始規劃的階段性門檻。因此先搭建完整靜態工作站與可用Python API，不以完整科學驗收作為場景交付的前置條件。原始規劃保留；數值結果與實際能力不放寬或改寫。

| Work package | 本輪交付狀態 | 後續能力 |
|---|---|---|
| WP-00 | Repo、隔離環境、版本／硬體／授權稽核完成；standalone官方Newton smoke通過；官方資產可取得 | 完整Lab3.0／Sim6.2 CUDA13/RTX runtime仍BLOCKED；四個Lab source-only smoke保留精確失敗 |
| WP-01 | 連續基座＋凸起、D0/D1與四種形狀、剛性治具、刀具CSV、MPM靜置／壓入／刮推、影片／trace已實測 | 材料未校準；網格／時間步收斂待完成 |
| WP-02 | Newton／CRESSim均實際建置／執行，失敗與源碼問題有紀錄 | CRESSim候選數值FAIL；兩者均未實作可信的持久材料分離；未正式決選 |
| WP-03 | ParticleSurface及三路理想RGB-D通過數值檢查；live環境連續輸出兩路觀測與Global debug | 實際D455 RTX、stereo noise、真切口深度待驗證 |
| WP-04 | 雙Franka、雙D455原生資產、Global camera、桌面／試件／治具／刀具的靜態USD工作站已建立並本機組合 | 手臂動力學／IK／碰撞／持刀雙向耦合未實作；目前運動學刀具獨立於兩臂 |
| WP-05 | CuttingEnv reset/step/observe/render/close、直接pose action、vision observation邊界及研究demo | Reward固定0；沒有正式benchmark、Policy或訓練 |
| WP-06 | 說明、重現腳本、CPU CI、source pins、精簡hash證據與本機原始artifacts | 完整research-ready切削驗收作為後續研究，不宣稱本輪已達成 |

正式能力狀態：PHYSICS_PROVEN=false、VISION_PROVEN=false、WORKCELL_PROVEN=false、BENCHMARK_READY=false。這些是原規劃的高階驗收標記，不等於目前場景或理想RGB-D API不能使用。實際可用部分見README／WORKCELL／API文件。

已實測：

- 修正後Newton：19,340粒子、2s靜置／壓入／刮推，4/3/2mm解析度探查。
- 彈性對照與yield-dependent殘餘變形；同seed重建121幀位置／速度完全相同。
- 理想深度QA：252,160有效像素、valid mask零差異、最大數值誤差8.41e-7m。
- Live CuttingEnv：120steps/2s，兩路640×480 RGB-D，正式觀測無force／oracle／global。
- 一鍵research_handoff_001已跑通：9個surface snapshots與雙臂／雙D455場景合成USD回放，tool timestamps核對通過。
- 額外GPU API檢查通過：直接pose action、truncation、D1同seed reset、close。
- 本機USD組合：13個官方USD依賴、21.4MB，雙Panda各58個mesh及原生D455 hierarchy。
- CPU suite擴充至23個案例；GPU與USD integration另存，不用CPU pass冒充物理驗收。

重要修正：早期Newton blade state-buffer同步問題已修正並重跑；CRESSim grid bounds錯誤和upstream數值問題保留。早期PASS僅指當時執行完成，失效分類見artifacts/RUN_VALIDITY.json。沒有刪粒子或預切凸起，也沒有把視覺刀痕當切削完成。

交付資料：docs/evidence/validation_summary.json保留數值／版本／hash；artifacts/保留USD、MP4、NPZ、trace與失敗log，不上傳大型資料。Public repo：https://github.com/AnsonNing/RoboticCuttingSim 。只修改此獨立repo。

後續：先使用workcell/API做感知與場景研究；controller與RTX sensor可獨立接入；真正切削以狹窄的persistent separation/recontact實驗繼續，不讓它阻礙現有原型使用。
