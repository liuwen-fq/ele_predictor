# 电解液配方预测、解释与候选筛选：官网集成 Demo

这是交付给官网团队的冻结模型 Demo。它包含 Streamlit 演示页面、三个只读业务模块、固定 G0-MA 生成结果和已离线筛选的 GDB17 候选目录，用于展示电解液配方设计中的“配方判断 → 解释归因 → 分子推荐 → 实验优先级”闭环。

Demo 不启动训练、不改写 checkpoint、不重建词表，也不会把 GDB17 新分子直接送入旧 diffusion / decoder。它接受受限词表内的交互输入并执行冻结 oracle 推理；不是接受任意客户 CSV、任意 SMILES 并在线训练或计算的通用服务。

当前 Streamlit 版本保留左侧页面导航，便于算法团队分别验收三个阶段。正式官网整合时，应由官网既有页面外壳、导航、鉴权与日志承载，而不是直接暴露开发版 Streamlit 服务。

## 交付效果

页面对应三个独立但连续的研发阶段：

1. **配方预测与解释**：从固定 Bamboo 词表选择多种溶剂和最多两种盐，输入组分组内摩尔份数、温度和总盐摩尔分数，输出主预测头的电导率，以及条件、物性、片段主项和片段对归因；
2. **分子推荐**：在 9,398 个已离线筛选的 GDB17 严格中性候选中，按片段、介电常数、黏度、熔点筛选，并用现有 `coverage_v1` 逻辑完成自动选择或辅助人工选择；
3. **Top-N 实验配方卡**：对固定 Bamboo 62 溶剂词表下的 G0-MA（seed 123）既有生成样本，按目标电导率与 anion ratio 重排序，输出最适合优先实验验证的配方卡。

首页应首先传达的业务价值是：**在固定、可追溯的模型覆盖范围内，把配方与分子空间压缩为可解释、可筛选、可优先实验验证的候选。**

配方预测页只展示主预测电导率，不把解释头的物理读数作为第二个预测结果。解释图只保留具有明确名称的条件基线、溶剂物性项、片段主项和片段对项；不展示没有明确化学语义的残差。

分子推荐页可下载自动或人工锁定的词表 CSV；Top-N 页面可下载实验配方卡 JSON。官网团队可根据产品策略保留或关闭这些下载入口。

## 直接查看

官网终端用户只需要浏览器。运行 Demo 的服务器需要 Linux x86_64、Python 3.11 和网络连接（仅首次安装依赖时），不需要 Conda 或名为 `bamboo` 的研发环境。

> **冻结资产交付前提：** `requirements` 文件只安装 Python 包，不包含 checkpoint、GDB17 目录或 G0-MA 结果。本交付分支将 `assets.sha256` 列出的冻结资产与代码一起保存，并保持清单中的相对路径。如果从研究仓库重新整理交付版，需显式收集默认被忽略的 `run/` 资产。

Debian / Ubuntu 服务器如果缺少 RDKit 和 PyTorch 的底层运行库，先安装：

```bash
sudo apt-get update
sudo apt-get install -y \
  libglib2.0-0 libgomp1 libstdc++6 libxext6 libxrender1
```

然后在仓库根目录创建独立虚拟环境。PyTorch CPU wheel 使用独立软件源，因此需要先安装 `requirements.torch-cpu.lock`，再安装其余锁定依赖：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps \
  --requirement apps/electrolyte_designer/requirements.torch-cpu.lock
.venv/bin/python -m pip install \
  --requirement apps/electrolyte_designer/requirements.lock

sha256sum --check apps/electrolyte_designer/assets.sha256
.venv/bin/python -m streamlit run apps/electrolyte_designer/app.py \
  --server.address=127.0.0.1 \
  --server.port=8501
```

打开 [http://127.0.0.1:8501/](http://127.0.0.1:8501/) 即可查看 Demo。部署到远程服务器时，可将 `--server.address` 改为 `0.0.0.0`，但应由公司网关提供 HTTPS、鉴权和访问控制。

两份 lock 文件面向 Linux x86_64 / CPython 3.11 / CPU 推理。CPU 首次加载 checkpoint 和完成配方预测会比 GPU 研发环境慢。Apple Silicon 或 ARM 云主机需要单独生成并验证对应架构的 PyTorch 依赖。

## 官网团队接入

当前 `app.py` 是 Streamlit 交互演示和业务逻辑参考，不应直接作为公网 API 服务。官网团队可以在其现有前端中重写界面，并将下列只读业务边界迁移到公司的服务框架；也可以先在独立 Python 虚拟环境中作为内网验证版运行。

下表是建议的 API 契约，**不是当前 Streamlit 服务已暴露的 REST endpoint**：

| 建议请求 | 返回内容 | 对应 Demo 阶段 |
| --- | --- | --- |
| `POST /api/v1/electrolyte/predict` | 主预测电导率、归一化组分比例、条件 / 物性 / 片段 / 片段对归因 | 配方预测与解释 |
| `GET /api/v1/molecules?motifs=...&dc=...&vis=...&tm=...` | GDB17 筛选候选、预测物性、片段和风险信号 | 分子推荐 |
| `POST /api/v1/molecules/select` | `coverage_v1` 自动词表提案或供人工锁定的排序结果 | 分子推荐 |
| `POST /api/v1/formulas/rank` | 既有 G0-MA 样本的 Top-N 配方卡 | Top-N 实验配方卡 |
| `GET /health` | 服务版本、模型资产校验状态和 Demo 标识 | 联调检查 |

推荐前端流程：

```text
页面加载
  ├─ GET /api/v1/molecules?...
  │    └─ 展示筛选后的分子、结构图、片段和物性证据
  ├─ POST /api/v1/electrolyte/predict
  │    └─ 展示预测电导率与当前配方的解释归因
  └─ POST /api/v1/formulas/rank
       └─ 按用户目标展示固定 G0-MA 样本中的 Top-N 实验配方卡
```

正式上线时由官网团队接入公司域名、HTTPS、登录鉴权、限流、审计日志、监控、跨域策略和客户配方数据的保留策略。用户提交的配方如属于保密信息，不应写入公开日志或无访问控制的下载文件。

## 展示数据与冻结资产

- **配方预测模型**：`config/test_results/anion_ratio_property_fragment_predict.yaml` 与 `run/anion_ratio_property_fragment_explanation/ckpt_formula_ft/` 中的冻结配方级 oracle。模型同时拥有主预测和解释分支；Demo 对外仅将主预测电导率作为预测值。
- **解释展示**：逐分子物性、片段主项和片段对来自当前配方、当前条件下解释头的归一化归因。正负方向反映模型判断，不是独立实验因果或真实相互作用能。
- **分子候选**：`run/data/GDB17_strictly_uncharged_smiles.csv` 的离线筛选结果，页面实际读取 `gdb17_electrolyte_property_catalog.json` 中的 9,398 个候选及其冻结物性预测。
- **软约束与选择**：介电常数 `dc`、黏度 `vis`、熔点 `Tm` 使用现有约束表；自动和人工排序共用 `coverage_v1`，综合物性窗口、片段边际覆盖、官能团家族、Murcko 骨架、多样性冗余、风险与 anion 倾向。
- **生成案例**：`G0-MA`、seed 123 的既有生成 JSON；固定 Bamboo 62 溶剂词表和当前盐词表。Top-N 是从有限结果中筛选和重排序，不会在线运行 diffusion 或 decoder。
- **完整性校验**：`assets.sha256` 覆盖预测配置与物性归一化配置、两个预测 checkpoint、固定评估指标、GDB17 目录、约束表、G0-MA 生成结果和 Bamboo 词表；安装环境后、启动 Demo 前应手动执行校验。

## 复现与验收

`app.py` 不是训练脚本。它只加载存档的冻结资产；训练配方预测模型、生成模型、GDB17 物性预筛和 G0-MA 采样均不属于此 Demo 的复现范围。

服务层测试覆盖资产可读性、GDB17 筛选和自动选择步骤、以及固定 G0-MA 结果的重排序。测试使用 Python 标准库 `unittest`，不需要额外安装测试框架：

```bash
.venv/bin/python -m unittest discover \
  -s tests -p 'test_electrolyte_designer_services.py' -v
```

环境验收建议在干净 Linux x86_64 / Python 3.11 主机上执行：

```bash
sha256sum --check apps/electrolyte_designer/assets.sha256
.venv/bin/python -m compileall -q apps/electrolyte_designer
.venv/bin/python -m unittest discover \
  -s tests -p 'test_electrolyte_designer_services.py' -v
```

随后按“直接查看”中的命令启动 Streamlit，并在另一个终端检查服务状态：

```bash
curl -f http://127.0.0.1:8501/_stcore/health
```

页面级验收至少包括：默认配方可完成一次预测；选择 `nitrile` 后分子推荐页不出现候选数 8–8 的 slider 错误且可点击查看结构；Top-N 页面可显示卡片并导出 JSON。

交付分支只保留 Demo 运行所需文件：

```text
Battery_discover/
├── README.md                       # 交付、安装、边界与验收说明
├── THIRD_PARTY_NOTICES.md          # 第三方代码、模型和数据声明
├── LICENSES/                       # 第三方许可证文本
├── apps/electrolyte_designer/
│   ├── app.py                     # Streamlit 演示页与业务展示
│   ├── services/                  # 冻结预测、目录筛选、Top-N 重排序
│   ├── requirements.lock          # 除 PyTorch 外的锁定依赖
│   ├── requirements.torch-cpu.lock # 官方 PyTorch CPU wheel
│   └── assets.sha256              # 冻结资产 SHA256 清单
├── formula_design/                 # 预测模型必需的 Bamboo 运行代码
├── scripts/                        # 片段、物性与 coverage_v1 选择逻辑
├── config/                         # 预测器与物性归一化配置
├── run/                            # 仅 Demo 需要的冻结权重和 JSON/CSV 资产
└── tests/                          # 无额外测试框架的服务验收
```

## 适用范围与已知局限

- **冻结词表限定覆盖范围。** 配方预测仅接受当前 Bamboo 词表中的溶剂和盐。词表内的新组合也不保证与训练分布具有相同覆盖度，结果仍需实验验证。
- **比例含义需要明确。** 溶剂与盐的输入值是各自组内相对摩尔份数，应用会分别归一化；“总盐摩尔分数”是 `n_salt / (n_salt + n_solvent)`，不是质量分数或 mol/L。
- **解释不等于因果。** 贡献处于解释头归一化空间；它说明模型在当前条件下依赖哪些信号，不证明片段或物性会独立导致电导率升高或降低。没有明确化学语义的残差不对外展示。
- **GDB17 不是可直接采购清单。** 候选分子的物性来自冻结预测，且 GDB17 结构不表示可合成、可采购、稳定、与盐相容或适用于实际电池。
- **Top-N 不是在线生成。** 它只在 seed 123 的有限 G0-MA 既有样本中排序。若当前允许的溶剂集合没有覆盖任何已有样本，页面会返回空结果，不会补入未选择溶剂。
- **新词表进入生成模型需要重训。** 自动/人工选择出的分子只能作为词表提案；若要替代 G0-MA 的 62 个分子，仍需重建 embedding、synthetic 数据，并重训 diffusion 与 decoder。
- **依赖版本需要持续维护。** `requirements.lock` 与 `requirements.torch-cpu.lock` 固定当前 Linux CPU 的包版本，但尚非带 wheel hash 的供应链锁；Python、PyTorch、PyG、RDKit 或模型资产升级后必须重新生成依赖锁并在干净环境验收。GPU 部署需要单独验证 CUDA 与驱动兼容性。

## 第三方声明

`formula_design/` 中的 Bamboo 模型骨架保留原始文件头中的 Bytedance 版权和 Apache License 2.0 声明；完整许可证文本位于 `LICENSES/Apache-2.0.txt`。其他数据、模型和依赖的来源及发布范围见 `THIRD_PARTY_NOTICES.md`。如果将该分支改为公开发布，仓库所有方还需确认冻结 checkpoint 和 GDB17 派生目录的对外再分发范围。
