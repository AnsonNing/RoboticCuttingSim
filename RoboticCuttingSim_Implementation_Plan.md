RoboticCuttingSim — 物理式表面切削模擬環境實作規劃
Codex 執行版 | v1.0 | 2026-10-27
專案類型：獨立的 Isaac Lab／Isaac Sim 模擬研究平台
主要使用者：機器人、物理模擬、視覺感知與機器學習研究同學
本文件是待執行規劃，不是已完成實作或已驗證的效能報告。

0. 給 Codex 的執行指令（先讀）
你負責從零建立一個獨立 repository，暫名 RoboticCuttingSim，用於「固定在桌面的軟質／塑性材料存在不規則凸起，機械手臂使用刀具去除凸起，使表面接近目標平面」的物理模擬環境。

這一階段只實作模擬環境與可驗證的研究工具，不進行 RSI、RL、VLA、Diffusion Policy、Agent 操作策略的實作或訓練。 場景必須保持未來可接入各種演算法的標準介面。請勿修改 AnsonNing/RSI、任何既有 RSI 工作目錄或其環境。

執行規則：
1. 先讀官方版本相容性文件，再動程式。 固定版本、依賴、GPU／driver、資產來源及可使用的 API；不要混用舊版 Isaac Lab 2.x／Isaac Sim 4.x 範例。
2. 優先解決物理與渲染可行性。 先做運動學刀具＋可變形材料的純切削 testbed，再接雙 Franka。另一個並行 testbed 專門驗證 MPM 動態表面能否進入實際 RGB-D 輸出。
3. 最終物理不可使用假切削取代。 純 heightfield 削減、直接刪去幾何、碰到凸起就移除預設剛體，只能當診斷 baseline，不可宣稱「完成物理切割」。切下材料必須具備真正的分離狀態、獨立後續動力學及質量帳務。
4. Newton MPM 與 CRESSim-MPM 都要實際評估。 不能只憑文件評斷，至少完成可重現的最小範例、API／耦合限制盤點及相同的單刀切削實驗。若某方案無法建置，記錄精確阻塞與證據。
5. 每個工作包需有：程式／設定、測試、最小 demo、執行命令、結果（影片或圖像）、性能與已知限制。完成後更新 docs/PROGRESS.md；重要設計決定記入 docs/adr/。
6. 不憑空宣稱通過測試、物理真實性或加速倍率；不能使用的 GPU／API 應明確標記 BLOCKED／NOT_RUN，交付可用的 CPU 測試與可重現腳本。
7. 遵守 repository 權限：沒有指定 GitHub 遠端時，先建立本機 Git repository；不要自行 push、發布大型模型／資產或修改其他專案。外部資產及第三方求解器依實際授權引用，完整保存來源與 commit。
8. 自主查閱文件、閱讀原始碼、執行實驗並做合理工程決策；只有會改變任務核心定義、需要無法取得的權限／硬體或必須付費時才需要人類決策。遇到不可排除的技術限制，保留獨立可運作部分並提供比較證據，不要改寫驗收標準掩蓋失敗。
第一個正式驗收目標不是 RL 成功率，而是： 用同一塊連續的軟質材料，讓一支刀具產生可測量的永久材料分離；切屑脫離、繼續受力／碰撞，餘下表面可從物理狀態與 RGB-D 相機分別量測，並能 reset、重現與定量驗證。
1. Scope、核心規格與非目標
1.1 任務定義
工作站含桌子、固定的軟質材料試件、兩台 Franka Panda、兩個 RealSense RGB-D 相機與一個全域相機。第一台 Franka 持刀沿試件表面shaving / planing / scraping，移除一個或多個凸起物；第二台 Franka 初期停在安全姿態。材料原本的平面與凸起必須是一個連續體，不能把凸起事先做成碰撞後會掉落的獨立剛體。
- 主要材料：Plasticine-like（油性塑形黏土／油土）的彈塑性近似；使用者不要求只能是黏土，可在材料模組加入 soft wax、foam 等，但第一版只集中一種材料。
- 核心物理：3D deformation、plasticity、刀具–材料接觸、可解釋的 failure / separation、persistent cut、chip dynamics、mass bookkeeping。
- 觀測：以 RGB-D 為主；不把接觸力／力矩作為未來策略的觀測。模擬器內部仍必須正確求解接觸力。建立 vision_only、vision_proprio 兩種可選觀測 preset；不要把 oracle 粒子、應力與目標平面偷放到正式 observation。
- 機器人：Franka × 2，初期 arm_0 持已固定刀具、arm_1 停泊。保留日後加入另一手臂、拿刀／換刀的空間，第一版不做自主抓刀。
- 相機：固定 D455 × 2＋global camera × 1。Global 預設只做錄影及除錯；D455 先做理想深度模式，再做符合 sensor model 的 noisy depth。
- 實驗模式：physics_only（無機器人運動學刀具）、workcell（完整雙臂）、benchmark（統一 step/reset/observation/metrics）。
1.2 本期不做
- 不做 RSI repository 整合、自主 agent、RL／模仿學習／VLA 訓練或演算法效果比較。
- 不做觸覺 sensors、策略輸入的 force sensing、真實機器人部署、ROS 2 網路控制、雙臂同時協作切削。
- 不以畫面像真的取代物理可驗證性；不以「可預設刪掉材料的動畫」作為材料分離結果。
- 不預設千環境平行化、即時 FPS 或 sim-to-real 已成立；先追求單環境物理品質、可重現和可評估。
1.3 分層驗收（重要）
等級	含義	最低證據
PHYSICS_PROVEN	真正的切削物理原型成立	材料分離、持久切面、切屑後續運動、質量帳務、數值穩定，附完整 log
VISION_PROVEN	變動材料表面可被相機量測	動態切前／切後 RGB-D、深度幾何與物理表面比對、timestamp／標定
WORKCELL_PROVEN	完整雙臂工作站可安全驅動刀具	articulation、IK／Cartesian path、無非預期碰撞、刀具－材料交互作用
BENCHMARK_READY	能供其他研究者直接做實驗	reset/seed、觀測與動作規格、變體、評估、CI／測試及重現文件


不得跳級宣稱完成： 如果只有幾何切除動畫，就算工作站與 API 完成，也只能記錄 GEOMETRIC_BASELINE_ONLY，不能稱為 PHYSICS_PROVEN。
2. 建議技術基線與硬體診斷
2.1 固定相容版本（截至 2026-10-27）
元件	建議基線	備註
Isaac Lab	v3.0.0 GA release	優先由正式 release/tag 取得，記錄完整 commit SHA
Isaac Sim	6.2.0	用於 USD、Franka、RTX、RealSense D455；不強制全部流程依賴 Kit
Newton	1.6.1	與上述 release 對應，MPM／MJWarp coupling experimental
NVIDIA Warp	1.17	依官方鎖定依賴，不自行混裝版本
Python	3.12	依官方環境
PyTorch / CUDA	Isaac Lab release 所釘選組合	release notes：Torch 2.11、CUDA 13；以實際安裝結果記錄
CRESSim-MPM	鎖定一個 tested commit	第二候選、獨立編譯；不要假設它與 Isaac Lab 使用同一套 ABI


官方基線：https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0 以及 https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html
2.2 專案啟動前的機器稽核（P0 第一步）
- 收集 uname -a、OS、CPU、RAM、free disk、GPU 型號／VRAM、driver、CUDA runtime、nvcc、Python、nvidia-smi。
- 比對 Isaac Sim 6.2 requirements；官方 x86_64 最低列 32 GB RAM、16 GB VRAM、RTX GPU，需實際驗證複雜場景的 headroom。無 RTX GPU 不代表 Newton MPM 不可測，但不能宣告 RTX／D455 rendering 功能已驗證。
- 檢查資產連結和網路連通、是否能取得 D455／Franka USD，並保存來源／hash。
- 以個別 venv／容器隔離 Isaac Lab 與 CRESSim；記錄 driver／CUDA／CMake compat。大型 datasets、video、checkpoint、asset cache 放在 repo 外，避免提交二進位檔。
- 生成 artifacts/environment/system_report.json 與 docs/COMPATIBILITY.md（測試執行時才產生的 artifacts 檔案不入 git）。
2.3 先跑官方 smoke tests
官方 Isaac Lab v3.0.0 工作目錄下：
uv run isaaclab example mpm-granular
uv run isaaclab example mpm-two-way-coupling
uv run isaaclab demo snowball-smash
uv run isaaclab demo teapot-fill --visualizer newton_gl --fluid_render_mode surface
若某個命令／extra 因 build 不同而變動，以 checkout 內 --help 和官方 3.0 文件為準，修改本 repo 文檔並記錄，而不是直接把錯誤忽略。每項記錄：啟動狀態、GPU memory、frame time、資產下載問題、log 截圖、實際執行命令。
3. 系統架構（物理、渲染和任務解耦）
                     Project configuration (YAML/Hydra)
                                  |
                     Parameterized scene generator
           +----------------------+-------------------+
           |                      |                   |
    Robot/fixture/tool       Material generator    Cameras / lights
   (USD / articulations)   (volume -> particles) (D455 x2 + global)
           |                      |                   |
           +--------------+-------+                   |
                          v                           |
                   Cutting physics                    |
                +---------+---------+                 |
                |                   |                 |
            Newton MPM        CRESSim testbed          |
          + cutting extension   + adapter              |
                |                   |                 |
                +---------+---------+                 |
                          |                           |
            Physical state + separated chips          |
                          |                           |
                Dynamic surface extraction ------------+
                          |
                RGB / Depth sensor outputs
                          |
             Environment API / evaluator / logger
                          |
               Future: RL / Policy / Agent
                    (NOT this phase)
3.1 必須固定的資料契約
- SceneSpec：robot placement、camera extrinsics、fixture、tool geometry、material geometry seed、solver config。
- MaterialState：粒子位置／速度／質量／體積、材料內部狀態、分離標籤／損傷變數（若 solver 支援）、chip 元資料；正式觀測不暴露。
- CuttingToolState：刀刃座標系、pose、twist、tool collider ID、刀具實際掃掠體積。
- RenderState：重建的表面幾何、版本／時間戳、visual／collision 表示的對齊資訊。
- Observation：rgb[d455_0, d455_1]、depth_m[d455_0, d455_1]、相機內外參、timestamp、可選 robot proprioception。
- Metrics：殘餘凸起體積、過切體積、切屑質量、質量守恆誤差、分離狀態、最大穿透、求解器殘差、耗時。
- EpisodeManifest：版本 SHA、config digest、seed、physics/renderer backend、資產 ID、env mode、測試結果。
各模組用明確的 SI units（m、kg、s、Pa），相機深度以 m 輸出；座標 convention 與 quaternion 順序在 docs/COORDINATES.md 寫清楚。Isaac／外部 solver 邊界做顯式轉換與單元測試。
3.2 求解器接口（按實際需求擴充）
不要為還不存在的後端預先做大型抽象框架；但第一個切削 PoC 應用薄接口保留比較可能：
class MaterialCuttingSolver:
    def initialize(self, specimen_spec, material_params, tool_spec): ...
    def reset(self, initial_state): ...
    def step(self, dt, blade_pose, blade_twist): ...
    def get_material_state(self): ...
    def get_surface_data(self): ...
    def get_diagnostics(self): ...
真實 Franka articulation 不直接放進此接口：初期先以運動學刀具驗證 solver，完整場景再透過 coupling adapter 交換剛體狀態與作用力。
4. 材料建構與場景參數
4.1 第一個連續試件
建議起始樣品（只是工程初始值，須經格距／切削實驗調整）：
- 試件：120 × 80 × 15 mm 連續基座；底部由 rigid tray／側限位固定，不將全體粒子鎖死。
- 凸起：1 個 smooth dome，直徑約 20–30 mm，最高點比原平面高 5–10 mm。
- 目標：把凸起上方多餘材料刮／削去，使它接近基準平面；允許多刀 passes，但每次 cut 必須可量測。
- 初期工具：刀具先剛性固定於運動學 proxy，避免抓取誤差干擾物理驗證；手臂整合後把刀具固定在 Franka 的 tool frame。剛性連接不代表模型自動具有接觸力回傳，需要獨立耦合測試。
用連續 height function h(x,y) 定義未切削形狀，再在 z_bottom ≤ z ≤ h(x,y) 內按 grid／particles-per-cell 體積取樣材料點；保持材料體積連通。新增 gaussian_bumps、ellipsoid_cap、ridge 參數化 shape family，測試 seed 的生成可重現性與不同 seed 的差異性。
4.2 材料模型初版
先用可變形／塑性材料模型近似 Plasticine-like behavior，參數包含密度、Young's modulus、Poisson ratio、yield／cohesion／hardening、接觸 friction、可選黏附與 rate dependence。不要冒充任何特定市售黏土的實測參數；在 configs/materials/plasticine_provisional.yaml 註記 uncalibrated。

需做三種獨立材料校準／合理性試驗：
1. 自由靜置／支撐：材料不出現嚴重非物理塌陷、穿透或自身振盪。
2. 平頭壓入＋卸載：可呈現不可逆塑性變形；保存 force–displacement 曲線供將來校準（不屬於 Policy observation）。
3. 單刀側向推進：材料能合理受剪、接觸、局部流動，且接觸後步進穩定。
4.3 隨機化（完整求解器通過後再做）
- D0：單個固定凸起，固定材料、光源及相機。
- D1：單個凸起隨機位置、高度、寬度／方向。
- D2：2–5 個凸起，混合 Gaussian dome／ridge／不規則形狀。
- D3：材料屬性、初始位置、相機外參、光照、刀具刃角等參數擾動。
抽樣範圍不得生成穿透桌面、穿透刀具、機器人無法到達或不符合既定試件條件的初始狀態；每次 reset 記錄實際抽樣參數、粒子數及材料質量。
5. Cutting Physics Solver：選型、實驗與強化
5.1 必要物理現象
一個切削模組必須清楚區分：
- 塑性流動（deformation）：形狀改變，不代表材料切開。
- 材料失效（failure）：根據屈服、累積塑性應變、damage／fracture criterion，允許局部失效。
- 位移不連續（separation）：切口兩側的運動自由度／網格傳輸不能被錯誤耦合。
- 持久切面（persistent discontinuity）：刀具離開後，分離狀態不能自動消失；切面可以再次相互接觸，但不能重新焊接。
- 切屑動力學（post-cut dynamics）：切屑保有質量與動量，能掉落、旋轉、滑動、接觸工作台。
- 質量守恆與可追溯性：任何粒子 inactive／delete／spawn 都記錄原因與質量去向。
不是最終解法的手段：刀具軌跡掃過即把網格或粒子直接刪除；把凸起預切成獨立物體；只在渲染時遮罩掉凸起；只更新 heightmap 不更新真實物理接觸。
5.2 候選方案
ID	方案	應實際驗證	主要風險
S-A	Newton implicit MPM + 自訂 failure／persistent separation	官方 MPM、碰撞、可擴充 solver 的介面；CPIC／side-label 能否加入內部 P2G／G2P 或接觸流程	侵入式 solver 修改、particle/grid 相容性
S-B	CRESSim-MPM (MLS／PB／Standard MPM、cutting geometry) + adapter	編譯成功、QuadSlicer 類切割範例、切口／切屑持續性、particle export、C API	非 thread-safe、外部剛體耦合、mesh 限制及相機整合
S-C	DiSECt／FEM cutting（研究參考／條件式原型）	Virtual nodes + cutting springs 是否更符合削平而非切半	幾何重切網格、版本舊、授權與整合成本
S-D	CPIC／BladeMaster／STC-MPM 技術重現	persistent side labels、damage evolution、tool swept interface	需重新設計／重現，不一定有可直接使用的代碼


首選比較 S-A、S-B，S-C/S-D 用於演算法查核及針對性強化。DiSECt 的 source code 使用 NVIDIA Source Code License，資料集還有 CC BY-NC 限制；不可未確認就合併或再散布。 CRESSim-MPM 為 BSD-3-Clause，taichi_mpm 為 MIT，仍需保留 notices 與來源資訊。
5.3 統一 Solver Bake-off 測試
同一個 specimen.json、blade_trajectory.csv、起始材料參數與單位下，比較：
- T1 indent_unload：材料塑性和回彈。
- T2 shallow_scrape：刀刃接觸、局部剪切，區分推擠與真切除。
- T3 through_bump_cut：刀刃沿接近目標平面移動，應有切屑分離。
- T4 blade_withdrawal：刀退回，切面繼續獨立。
- T5 two_pass：兩次不平行切割、切屑重新接觸及切面是否誤焊。
- T6 grid_dt_refinement：至少三個網格尺度／時間步配置，量測收斂而非只比較動畫。
- T7 reinitialize：同 seed reset 後初始粒子、內部狀態及切削結果可在數值容差內重現。
- T8 surface_export：切後表面及切屑可轉成外部幾何表示供相機／分析工具使用。
每項存：原始資料、曲線、影格、失敗分類、時序、質量誤差、GPU VRAM、simulation seconds / wall seconds、版本與 commit。
5.4 Solver 選型硬條件與評分
必須先通過以下硬條件，才進入加權比較：
- 至少一個 through_bump_cut case 形成持久材料分離；切屑與餘料可以獨立運動。
- 不發生隱藏的大規模質量流失、爆炸性速度或系統性穿透。
- 可 reset，且粒子／分離狀態可導出；能夠定義與 D455 渲染對接的表面。
- 有合法、可維護的使用授權與 API／kernel 修改路徑。
過硬條件後：分離物理 30%、數值／材料行為 15%、表面與相機整合 15%、剛體耦合 15%、穩定與 reset 10%、計算效率 10%、維護成本 5%。在 docs/reports/SOLVER_SELECTION.md 逐項列測量值與理由。

如果 S-A/S-B 都不能通過硬條件：暫停正式 solver 決選，選一個更窄的 material-separation problem 重現 CPIC／persistent labels，並交付清楚的失敗報告。不能用幾何刪除替代並宣稱達標。
5.5 解析度與時間步
- 以 voxel_size 約 4 mm 作粗略探查，再測 2 mm 和 1 mm 等配置；這只是測試矩陣，不是聲稱 1 mm 一定可執行。
- 依接觸穩定性、刀刃厚度與 grid convergence 選 substeps／dt，不預先保證 240 Hz 或「真實即時」。
- 物理 step、控制 step 和相機 step 分別配置，並記錄時間戳及同步策略；避免相機顯示上一個物理步的切面。
- 若切割需要 sub-grid 精度，需比較 tool SDF／proxy contact／局部幾何重建；不能只縮小 mesh 後忽略計算成本。
6. 動態表面重建與相機成像（第二條優先研究線）
6.1 必須先驗證的問題
Newton MPM 的 state 是粒子。官方 ParticleSurface / ParticleSurfaceRenderer 可重建並渲染表面，但官方文件說 Newton GL／RTX visualizers 支援重建 surface，Kit visualizer 仍直接顯示 MPM particles。因此「可看到美觀表面」與「D455 depth annotator 可讀到新切面」是兩個不同的驗收。

請建立以下兩個獨立 PoC：
1. Newton renderer path：依官方 teapot-fill 表面渲染建立 ParticleSurface；導出切後 vertex／triangle／normal 與時間戳。測試其是否可供實驗相機輸出，明確記錄使用的 renderer。
2. Isaac Sim RTX / D455 path：將同一個隨時間變動的 surface 提供給 Isaac Sim 的 render product；以官方 RtxCamera、SingleViewDepthCameraSensor 取得切前／切後 RGB、depth。若需要中間動態 mesh staging，先量測更新延遲、GPU memory、scene sync 與有效性。
任何一條路線如果只在 GUI 有畫面但無法產生程式可讀 RGB-D tensor，標記 VISUALIZER_ONLY，不算 VISION_PROVEN。
6.2 Camera 方案
- Isaac Sim 6.2 RealSense D455 官方 asset：Isaac/Sensors/RealSense/D455/rsd455.usd（以實際 asset root 解析）；D455 理想工作距離約 0.6–6 m，最大解析度下的最小深度距離約 0.52 m。
- D455_0：上方斜俯視固定相機，初始距離約 0.65–0.85 m，覆蓋全部試件及刀具路徑。
- D455_1：側前方固定相機，初始距離約 0.6–0.85 m，觀察刀刃入料與切屑。若近距離不能可靠估算毫米級深度，不硬把 D455 放進最小工作距離內；另外以 ground-truth mesh 做物理 QA。
- Global：寬視角錄影與除錯，預設非正式觀測。
- 解析度依測試用 640x480 作第一版可設定值；真實 D455 像素與後處理選項另依官方規格調整。
需要提供兩種深度 preset：ideal_depth 和 realsense_modeled_depth。前者驗證幾何與遮擋；後者使用可支持的 stereoscopic post-processing、noise、disparity/outlier logic。二者均保留相機 intrinsic、extrinsic、Depth encoding（單位 m、invalid pixel 的定義）。
6.3 視覺 QA
- 固定幾何棋盤格／標定板：內外參及 frame mapping 近似正確。
- 靜態平面：理想深度與幾何距離誤差落在數值／renderer 可解釋範圍。
- 動態鼓包：凸起變形前後，在相機視角可辨識高度變化。
- 真切面：切後的 surface geometry 與 RGB-D 中切口邊界、切屑位置一致；不許仍渲染舊表面。
- 深度 valid mask、遮擋區、刀具遮罩、不連續邊界都應有測試；不以平均深度正確掩蓋切口缺失。
- 對比物理真值時先使用 ideal_depth，realsense_modeled_depth 的誤差單獨報告，不用感測器雜訊假象來替代幾何變化。
7. Franka 雙臂工作站（在 Physics / Vision proof 之後整合）
7.1 場景資產
- Franka Panda × 2：從官方 USD／Isaac Lab 配置取得；檢查所選資產在 newton_mjwarp 下 articulation、joint drive、collision 是否受支援。
- Table：rigid/static support。
- Fixture：托盤／側邊機械限位，避免試件被整塊推出；固定方式需經壓入／側向推動實驗確認。
- Knife：高品質 visual mesh、獨立可配置 collision mesh／SDF；刀刃厚度、刃角、有效刃長、刀具摩擦與 tool frame 都記錄。
- 相機 x3：每台有固定 pose、intrinsic、extrinsic、render preset。工作站本身定義世界坐標系與刀具／試件局部坐標系。
SimReady Central 可協助尋找桌子、治具、支架及機器人等 OpenUSD 場景資產；但資產若沒有符合本次 physics backend 的 collision profile，必須修正／替換。它不是切削引擎。
7.2 階段性耦合
- workcell_static：無材料切削，雙 Franka 載入並驗證姿態、joint limits、碰撞、IK 可達性、相機遮擋。
- kinematic_blade：用運動學指定的刀具軌跡驅動材料求解器，做確定性物理測試。
- coupled_arm_blade：只有 solver 已通過時才讓 arm_0 帶刀進入；使用 Newton MJWarp + MPM proxy coupling 的已維護範例作起點。初始 lagged，先比較單向與雙向力回傳的運動差異，再調整 coupling iterations。
- arm_1 初期鎖定在已確認不碰撞、不遮擋主要視線的停泊姿勢，不以完全移除第二隻手臂取代規格。
注意 MPM 自己不解 articulated rigid dynamics；直接讓刀具按指定 pose 移動只證明可變形接觸，並未證明對 Franka 的反作用力已耦合。正式「機器人用刀切材料」需驗證 rigid/deformable bidirectional feedback（但不將 force 提供給未來策略）。
7.3 工具與控制接口
先提供 deterministic scripted trajectory player，不需學習：
- joint_position：兩臂 articulation joint commands。
- cartesian_pose / cartesian_delta：arm_0 的 tool frame 目標，搭配 IK／controller；對外觀測仍是 vision first。
- kinematic_blade_trajectory：物理 QA 專用，讀 CSV time/pose/velocity；不經 Franka，保證可比較 solver。
把 robot controller 的求解結果與 knife tip pose 分別記錄，確認控制誤差不會被誤判為材料物理誤差。
8. Benchmark API 與研究資料（本期先建，不訓練模型）
8.1 對外 API 期望
可以實作一個 Python facade，維持 Gymnasium 類語義：
sim = CuttingEnv(config)
obs, info = sim.reset(seed=42)
obs, reward, terminated, truncated, info = sim.step(action)
frame = sim.render()
sim.close()
這是目標 API，不代表 Isaac Lab 已有完全同名的 CuttingEnv。 必須由本 repo 實作並編寫最小用例。Reward 可以先只是固定的 benchmark score；本期不用訓練或調 reward shaping。

Observation profile：
- vision_only：兩個 D455 的 RGB、Depth、valid mask、timestamp、camera calibration；不輸入力或 ground truth 物理狀態。執行器仍可在內部使用 robot encoder 作低階控制。
- vision_proprio：上述觀測＋q、qdot、gripper/tool pose，日後可做容易收斂的 baseline。
- debug_oracle：粒子、表面網格、目標平面、材料參數、真實切屑標籤；僅供研究／評估，不可與前兩者混用。
- global_camera：錄影／debug，可由 config 開放給其他實驗，但預設不放在 official benchmark observation。
8.2 必要 metrics
指定 ROI、基準平面 z0、切削後真實表面 h(x,y)：
- remaining_bump_volume = ∬ROI max(0, h(x,y)-z0) dxdy
- overcut_volume = ∬ROI max(0, z0-h(x,y)) dxdy
- remaining_height_p95 = P95(max(0, h-z0))
- roughness_rmse = sqrt(mean((h-z0)^2))（需明確指定有效 ROI／mask）
- chip_mass, substrate_mass, material_mass_error：|M0-(M_substrate+M_chips+M_accounted)|/M0
- persistent_separation_valid：依 solver 真實切面／side-label／連通狀態與後續運動確認，不能只靠像素距離。
- blade_material_penetration, nonfinite_count, solver_residual, physics_walltime, render_walltime。
若表面無法表示成單值 heightfield（例如材料卷起、切屑覆蓋），必須用 3D mesh／signed-distance／體素或粒子方法，將主試件與切屑分開評估；不得偷偷只保留可見最高表面使評分失真。
8.3 Episode 與資料輸出
- 每集唯一 run_id、seed、scenario variant、solver name/version、renderer name/version、material params、camera params、git SHA、timestamp。
- 每集可輸出 manifest.json、metrics.json、物理 trace（按需求 npz／parquet）、相機影像與低頻 preview、失敗截圖與 solver log。
- 把 rendered RGB-D 與 oracle physics state 放不同路徑，不讓未來訓練管線誤用。
- 每個模組文件指定資料欄位、dtype、shape、坐標系、單位和 missing-value handling。
9. 實驗驗收與品質門檻
9.1 必須跑的測試清單
Test ID	測試	需要產出的證據	允收邏輯
PHY-01	材料靜置	drift/penetration/time-series	無爆炸、形狀穩定，偏移相對目標幾何可接受
PHY-02	壓入／卸載	load–displacement、殘餘變形	可呈現且重現塑性／彈性成分
PHY-03	刀具接觸及摩擦	blade pose、接觸 trace、剖面	無非物理大穿透，solver 不崩潰
PHY-04	單次切削	切面、切屑質量、分離標籤	真正分離，切屑與底座後續可不同運動
PHY-05	刀具撤出與切面重接觸	withdrawal＋rest trace	可以接觸但不自動焊接回連續體
PHY-06	質量帳務	mass budget CSV	無不明質量損失；門檻待材料/離散化測試校準
PHY-07	網格／dt 收斂	3 組配置的 metric curves	主要觀測量隨 refine 趨於收斂，不能只看單一 seed
PHY-08	reset 再現性	repeated seed results	在相同 backend／GPU 條件下具容差內再現性
VIS-01	D455 靜態深度	ideal depth vs known plane	單位／外參／valid mask 正確
VIS-02	動態材料表面進相機	before／during／after RGB-D	不得只在 viewer 顯示，程式要可讀 tensor
VIS-03	切口／切屑的動態幾何	depth + physical surface overlay	新切面位置／遮擋／切屑在正確相機座標中
ROB-01	雙臂場景	stage、articulation、collision	雙臂可正常載入，parked arm 不意外碰撞
ROB-02	arm_0 持刀執行固定 trajectory	tool-pose log、影片	可到達且位置誤差可量化
ROB-03	耦合的剛體反作用	kinematic/coupled A/B	雙向耦合有可解釋力學回饋，不當作 policy force sensor
ENV-01	reset/step/observe/render/close	API smoke＋重複運作	無殘留髒狀態、明確退出／錯誤報告
ENV-02	D0–D2 procedural variants	seed／形狀樣本集	隨機化且不破壞合法初始狀態
ENV-03	benchmark/scoring 分離	observation contract tests	vision_only 不包含 oracle state／force


9.2 數字門檻的制定方法
不要現在就宣稱固定的 mass/force/depth 精度已達成。Codex 應在 P2 研究報告中，以測量結果提出可實際達成的門檻與理由，例如 material_mass_error <= 0.5% 可作初始調查目標，但若材料演算法不能質量守恆，就不應靠調寬門檻通過。深度誤差門檻需分 ideal_depth 與 modeled_depth，並結合 grid spacing 與場景尺度定義。所有數值門檻在 formal benchmark 前凍結、經 review 後放入 config。

數值收斂的結果與「符合真實油土」分開宣告：若未取得真實物性／實驗資料，最終狀態只能說 PHYSICS_VALIDATED_IN_SIMULATION，不能說已完成真實世界校準。
10. Repo 結構（建議目標，不要一次性產生大量空殼）
RoboticCuttingSim/
├── README.md
├── AGENTS.md                         # 本文件第 0 節與執行規則摘要
├── pyproject.toml
├── uv.lock / requirements lock       # 依官方安裝策略選擇
├── .gitignore
├── configs/
│   ├── simulation/{physics_only,workcell,benchmark}.yaml
│   ├── materials/plasticine_provisional.yaml
│   ├── solvers/{newton_mpm,cressim_mpm}.yaml
│   ├── cameras/{ideal,d455_modeled}.yaml
│   └── tasks/{d0,d1,d2,d3}.yaml
├── assets/
│   ├── tools/                        # 自有刀具 USD／mesh
│   ├── fixtures/                     # 自有治具 USD／mesh
│   └── ASSET_SOURCES.md              # 官方／第三方來源、license、commit
├── src/robotic_cutting_sim/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── scene/{workcell,robots,fixture,tool}.py
│   ├── materials/{geometry,particles,constitutive}.py
│   ├── physics/{base,newton_mpm,cressim_adapter,cutting}.py
│   ├── rendering/{surface,camera,d455_bridge}.py
│   ├── envs/{cutting_env,observations,actions}.py
│   ├── evaluation/{metrics,validation,recording}.py
│   └── utils/{transforms,units,seeds}.py
├── scripts/
│   ├── doctor.py
│   ├── run_physics_probe.py
│   ├── run_solver_comparison.py
│   ├── validate_depth.py
│   ├── launch_workcell.py
│   └── run_benchmark.py
├── tests/
│   ├── unit/                         # CPU 可跑
│   ├── integration/                  # GPU / IsaacLab
│   └── physics/                      # 嚴格材料與切削測試
├── experiments/
│   ├── materials/
│   ├── cutting/
│   └── renderer/
├── docs/
│   ├── SETUP.md
│   ├── COMPATIBILITY.md
│   ├── COORDINATES.md
│   ├── PHYSICS_VALIDATION.md
│   ├── CAMERA_PIPELINE.md
│   ├── BENCHMARK_SPEC.md
│   ├── PROGRESS.md
│   ├── adr/
│   └── reports/
└── artifacts/                         # gitignored: outputs, videos, traces
使用 .gitignore 排除大檔、下載資產 cache、GPU trace、videos、模型檔。保留小型、可再生的 scene／test fixtures；第三方程式碼原則上以 pinned submodule／source ref 或可重現建置腳本取得，不要偷拷 license 不相容的實作。
10.1 CLI 最終期望（由 Codex 實作，不是已存在指令）
python -m robotic_cutting_sim.cli doctor
python -m robotic_cutting_sim.cli physics-probe --case through_bump_cut --solver newton
python -m robotic_cutting_sim.cli physics-probe --case through_bump_cut --solver cressim
python -m robotic_cutting_sim.cli compare-solvers --suite cut_v1
python -m robotic_cutting_sim.cli camera-check --scene dynamic_clay
python -m robotic_cutting_sim.cli scene --preset workcell --seed 42
python -m robotic_cutting_sim.cli benchmark --variant D0 --seed 42 --trajectory reference
不要求 CLI 一開始就全部存在；每個工作包完成時再加入確實有 end-to-end 測試的子命令。所有 CLI 要有 --help、非零 exit code、機器可解析的結果 JSON、指定 --output-dir。
11. Codex 工作包與實作順序（依相依關係執行）
WP-00 — Repo / hardware / compatibility（首要）
- [ ] 新建獨立 Git repo、README、AGENTS、config/test 基本架構；不可動 RSI。
- [ ] 機器診斷＋ Isaac Lab 3.0 GA / Isaac Sim 6.2 相容環境。
- [ ] 完成官方 MPM smoke demos，記錄實際版本、命令、GPU memory 與影片。
- [ ] 確認是否有可用 D455／Franka 資產、license／下載策略。
- DoD：另一位研究者能照 docs/SETUP.md 安裝或至少清楚重現阻塞；環境紀錄和 smoke log 齊全。
WP-01 — Procedural material and blade-only Newton testbed
- [ ] 建立 continuous base＋bump 形狀、粒子取樣、材料質量計算、種子決定性測試。
- [ ] 在 Newton MPM 中載入塑性材料、rigid tray、kinematic blade；執行 settling／indent／scrape。
- [ ] 建立可讀 CSV 的 tool trajectory player 和 physics trace。
- DoD：不用任何 Franka，也能從指令重現 material/tool 互動；能證明塑性變形與接觸，不提前宣告材料分離。
WP-02 — Cutting solver bake-off（首要研究里程碑）
- [ ] S-A Newton：調查內部 extension points，確認材料分離所需 P2G／G2P／side labels／damage 支援缺口。
- [ ] S-B CRESSim：pinned commit、建置、官方 cutting demo、C API、material export、thread safety／mesh 警告稽核。
- [ ] 執行 T1–T8；若兩者不完整，最小重現 CPIC／cutting geometry／persistent discontinuity 的缺口。
- [ ] 形成 SOLVER_SELECTION.md、最小影片、曲線、source attribution、決策 ADR。
- Gate G2：有實際材料切斷、切屑分離和撤刀後持久性的證據，並選擇後續主要求解器；否則明確標記阻塞並繼續獨立 renderer/workcell 工作，但不可宣告 cutting 完成。
WP-03 — Dynamic surface + RGB-D proof（與 WP-02 可並行）
- [ ] 用可變形 particle world 產生時間序列，測 ParticleSurface extraction、mesh update 與正常／異常拓樸。
- [ ] D455 資產、camera sensor、ideal depth、modeled depth；兩個 RealSense 和 Global camera 正常輸出。
- [ ] 驗證渲染器看到的確實是最新 material state，有切前／切後 depth overlay／誤差。
- Gate G3：真正從 camera API 取得連續材料／切屑的 RGB-D tensors；若只在 visualizer 正常顯示不得宣稱通過。
WP-04 — Dual Franka workcell and coupled cutting
- [ ] 正式 OpenUSD workcell（table、tray、tool、Franka×2、D455×2、Global）；path/asset metadata。
- [ ] 安全固定第二手臂；第一手臂持刀、IK path、joint limit 和 collision check。
- [ ] 以已選 solver 橋接刀具與材料；Newton 時由維護的 MJWarp–MPM proxy coupling 範例調整，驗證雙向作用力與 solver 穩定。
- [ ] 量測 kinematic vs robot-coupled trajectories 的接觸與運動差異。
- Gate G4：一個真實的 arm_0 cutting trajectory 可在場景內執行，不只是 detached blade proxy demo。
WP-05 — Benchmark, randomization, evaluation
- [ ] CuttingEnv reset/step/render/close、action/observation profiles。
- [ ] D0→D2 randomization；測試 seed、初始合法性、不同 seed 覆蓋分布。
- [ ] 工具軌跡／scripted reference 作演示和回歸，不訓練 policy。
- [ ] 切屑／餘料的真實幾何與 mass metrics，另以 RGB-D 檢查視覺一致性。
- [ ] 追蹤性能、可靠性、配置 hash；正式樣本與調參 dev seeds 分離。
- Gate G5：第三方研究者依 README 啟動單集、拿到相機和 metrics，無需知道 solver 內部細節。
WP-06 — Release / reproducibility / handoff
- [ ] CPU unit tests、GPU smoke／physics tests、known issues、外部資產 source/license。
- [ ] 圖片／短影片、完整 architecture／material／camera／benchmark 文檔。
- [ ] 實測性能表：GPU memory、各階段 wall time、simulation seconds / wall seconds、最終每集成本。
- [ ] 將正式使用版本凍結、提供 reproduce_demo.sh 或等價腳本。
- [ ] 新同學 README walkthrough、PR guide、測試指令、場景新材料的擴充指南。
- DoD：PHYSICS_PROVEN、VISION_PROVEN、WORKCELL_PROVEN、BENCHMARK_READY 的證據鏈可查。如果其中一項未成立，文件分開標明而非把整體標成已完成。
11.1 優先關係與可並行工作
WP-00
  |
  +-- WP-01 ---- WP-02 (cutting solver bake-off) --+
  |                                               |
  +------------- WP-03 (dynamic surface/RGB-D) ---+--> WP-04 --> WP-05 --> WP-06
WP-04 前不需要等待所有材料參數的真實校準，但必須有可信的基本材料分離實作。若 WP-02 卡住，可持續做 WP-03 與 workcell 靜態測試；不可用 cosmetic cut 接管最終 benchmark。
12. 開發與合作規則
12.1 研究循環
每個工作包以 hypothesis -> implementation -> minimal test -> metrics -> diagnosis -> revision -> reproducible artifact 迭代。每個重要問題留下可重現 failure case、配置、資料與最小修正；不得只以截圖作為正確性證據。
12.2 文檔與 Git 習慣
- docs/PROGRESS.md：每完成一個子工作包記日期、commit、測試執行與結果、未解問題、下一步。
- docs/adr/：只記影響架構／solver／material／render backend 的決策，說明被拒方案。
- docs/reports/WP-XX.md：精簡報告：What changed / Commands / Evidence / Numbers / Known limitations / Next step。
- 一個 PR／branch 解決一個可單獨驗證的工作包；不要把 physics solver、USD assets、benchmark API 的大改動混成難審的單次提交。
- 對其他研究同學公開 CONTRIBUTING.md、CPU-only unit tests、需 GPU 的 marker 和資產取得方式；不提交密鑰、私人資料或超大 log。
12.3 測試分層
1. unit/cpu：幾何、數值單位、seed、transforms、metric math、config schema；一般 CI 每次必跑。
2. integration/gpu：Isaac／Newton 載入、相機、Franka、場景 reset；有相容 GPU 才跑。
3. physics/expensive：網格收斂、多次 cut、多 seed、RGB-D 大批比較；每次關鍵 solver 修改或 release 前跑。
4. manual/visual：除錯用影像與影片，但必須連結物理量測而非獨立當成測試。
測試不能以硬體不支援時的 skip 當成功通過，報告必須列 PASS / FAIL / SKIP / BLOCKED / NOT_RUN 及原因。
12.4 安全與版本管理
- 只對新 repo 操作；下載第三方程式與執行不受信任範例前檢查授權、依賴及來源。
- GPU 訓練／模擬不能搶占同一機器上其他研究者的工作，先檢查現有佔用量；運行資源密集任務要可以中斷並保存結果。
- 研究資料及 output 按 experiment／commit 分目錄保存，不默默覆寫正式結果。
- 先選一個能被管理的完整版本組合；不因某個新功能就盲目升級整個 Isaac Lab 主版本。
13. 風險、回退與決策觸發
風險	發現方式	工程對策	不可做的事
MPM 只能推擠不能永久切斷	T3/T4/T5	比較 CRESSim、CPIC、persistent labels；必要時自訂求解器	遮罩掉材料便宣稱完成 cutting
難以建立黏土參數	壓入／切削行為不一致	使用 provisional material，明確標記 uncalibrated；建立校準待辦	假稱真實材料一致
MPM 粒子不能進入 D455 RGB-D	VIS-02	研究表面 extraction、dynamic mesh staging、renderer pipeline；記錄限制	只提供 GUI 截圖取代 sensor tensor
CUDA／GPU／RTX 相容性問題	WP-00 smoke	按官方 pinned config 修正；分離 solver env；完成能做的 CPU test	混裝舊版套件並忽略 ABI 問題
CRESSim 切割不會產生獨立切屑	T3/T4/T5	追查其 cutting semantics，加入 persistent-separation 或改採別的方案	只見到一條刀痕就當成功
Cutting 性能過慢	benchmark profiler	降低實驗規模、cache、GPU profiling、調小 region；高／低 fidelity 分離	不量測就宣稱可做大量 RL
reset 的材料歷史殘留	T7	明確 reset solver history、驗證 same-seed replay	只重設粒子位置、不處理 plastic history
產生大量大檔	disk dashboard	影片取樣、artifact TTL、只保留基準資料和代表錯誤	將 raw dump 直接提交 Git


Newton 官方特別說明獨立 MPM worlds reset 需處理 solver history，並非一般 asset reset 就等同完全重置；此處一定要寫回歸測試。
14. 第一輪 Codex 實作的具體交付範圍
請先完成 WP-00 和 WP-01，並啟動 WP-02、WP-03 的可行性驗證。 不是第一輪就生成幾百個模板檔案、接好所有 RL framework 或把所有階段標記 done。

第一輪期待收到：
1. 本機新 repo、AGENTS.md、README.md、docs/PROGRESS.md、docs/SETUP.md、版本／硬體／授權稽核。
2. 官方 MPM example 的成功或精確失敗 log。
3. 能在簡化場景生成一個連續基座＋單一凸起物、固定治具、刀具軌跡，並輸出 deformation traces 的程式。
4. Newton 與 CRESSim 兩條路線的初始可行性報告：實際成功執行了哪些最小範例、cutting semantics 是什麼、阻塞在哪裡。
5. ParticleSurface → 相機成像的最小技術驗證方案與第一輪結果，明確分 viewable surface 與 camera depth。
6. 一份下一個工作包優先順序建議，依實際測試結果安排，而不是依本文件的假設直接決選 solver。
第一輪不要求：完整雙 Franka、全部 camera、D3 randomization、最終真實校準、RL 訓練，也不應以這些尚未完成的部分作為已完成物理 PoC 的必要前提。
15. 主要研究來源（請 Codex 從這些官方／原始來源開始）
官方版本／架構
1. Isaac Lab 3.0 GA release：https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0
2. Isaac Lab 3.0 Implicit MPM：https://docs.nvidia.com/isaaclab/3.0/concepts/implicit-mpm
3. Isaac Lab Coupled Solvers：https://docs.nvidia.com/isaaclab/3.0/concepts/coupled-solvers
4. Isaac Lab Agent Skills（僅供 Codex 開發 Isaac Lab，不是本期 RSI）：https://docs.nvidia.com/isaaclab/3.0/developer-tools/agent-skills
5. Isaac Sim 6.2 release notes：https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html
6. Isaac Sim 6.2 requirements：https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html
7. Isaac Sim 6.2 D455 assets：https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_depth_sensors.html
8. Isaac Sim 6.2 stereo-depth processing：https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera_depth.html
9. NVIDIA SimReady Central：https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories
切削物理／材料
10. CRESSim-MPM（BSD-3-Clause）：https://github.com/yafei-ou/CRESSim-MPM
11. MLS-MPM + CPIC（SIGGRAPH 2018、MIT reference）：https://github.com/yuanming-hu/taichi_mpm
12. DiSECt（FEM／Virtual Nodes／Cutting Springs；查授權）：https://github.com/NVlabs/DiSECt
13. BladeMaster（2026、persistent discontinuities）：https://jango6324.github.io/blademaster/
14. PlasticineLab：https://plasticinelab.csail.mit.edu/
15. RoboCook：https://hshi74.github.io/robocook/
來源使用規則：閱讀 repo 的 README、實際源碼、license、issue 和 API 當前版本；不要假設論文實作可以原封不動嵌入 Newton。記錄引用來源的版本、授權與已驗證／尚未驗證的主張。
16. 應遵守的完成定義（最後的驗收契約）
當且僅當下列全部成立，才可以將第一版稱為 research-ready cutting simulation environment：
- [ ] 有連續初始試件與可程序化生成的至少 D0/D1 變體。
- [ ] 材料存在合理的彈塑性回應，並能產生永久切面與分離切屑，有 solver 證據支持。
- [ ] 切屑、剩餘材料、刀具、桌面之間的後續動力學與碰撞可模擬，無大規模非物理質量流失。
- [ ] 有獨立 physics validation suite、收斂／reset／守恆實驗與結論；實測限制清楚。
- [ ] 切削前後的變形材料能被 API 輸出的 RGB-D 觀測，而不只是 UI 動畫。
- [ ] 完整桌面工作站含兩台 Franka、兩台 RealSense D455、Global camera，單臂可持刀執行固定切削軌跡。
- [ ] Robot-coupled 模式已驗證剛體／材料雙向作用，且不把 force 暴露成正式觀測。
- [ ] CuttingEnv 可 reset、step、render、record、score、固定 seed，能讓未來的研究者接不同 policy framework。
- [ ] 有清楚的安裝、版本、GPU requirements、重現腳本、測試與合作貢獻說明。
若其中部分條件未成立，仍需交付現有可用的物理 testbed、相機 workcell 或 API，但精確標記 PARTIAL／BLOCKED，不得把上面的條件降低為假切割後直接結案。
Codex 可直接使用的啟動語句
請以本文件 RoboticCuttingSim_Implementation_Plan.md 作為新獨立 repository 的執行規劃。優先閱讀第 0、2、9、11、14、16 節，先建立本機 Git repo 並完成 WP-00；接著實作 WP-01 並啟動 WP-02 的 Newton/CRESSim 切削原型比較、WP-03 的 ParticleSurface-to-RGB-D 可行性驗證。請自主查官方文件與原始碼、小步實作和測試，逐工作包產生報告並更新 PROGRESS。不要修改 RSI，不做任何 Policy／RL／Agent 訓練，不以幾何刪除冒充真實切削，也不要在沒有測試證據時宣稱完成。若不可取得 GPU／資產，照實列出阻塞、保留可執行的獨立工作並提出下一個最小可驗證動作。