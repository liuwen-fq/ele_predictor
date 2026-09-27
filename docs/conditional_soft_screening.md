# 自动条件化分子软筛选（第一版）

本路径不修改 G0 结果或 MA 训练目标。新版 Bamboo G1/G2 默认关闭固定物性硬门槛；历史实验数据保留，但候选池不再与带门槛版本相同。主比较为相同 MA 流程下 G1 的随机/多样性词表与 G2 的自动条件化词表。

## 新评分的含义

输入为指定盐种及盐内比例、salt_molar_ratio、温度、目标 σ/r 和容差。拟合只接受显式声明 `role=rule_fit, split=train` 的预测导出，拒绝 test/valid 标记及明显的测试文件名。声明不能证明数据划分正确，调用者仍需确保导出来自训练侧，最终测试集不得用于拟合或调参；可以进一步使用训练侧的交叉拟合导出。

1. 按盐身份精确匹配，并对温度、浓度、盐内比例及联合目标距离做核加权。
2. 从解释导出的片段主项读取 Δlnσ、Δlogit(r)。计算将该项从解释预测中移除前后的归一化目标误差差值；目标上方的正贡献可能得到负分。它是解释模型的代理敏感性，不是独立分子的因果效应，也不是普通预测头的严格分解。交互项本轮继续交给 MA 和后续配方评价，未混入分子片段主项评分。
3. 同一配方中的同种片段先合并，重复配方导出只计一次。根据有效样本量、方差和收缩项估计分数，不再使用固定“腈+1、酯+0.7”等表。
4. 从接近目标的配方中按溶剂比例加权，提取**逐分子**物性分布 P10–P90 软窗口；不将配方平均物性范围作为单分子硬门槛。
5. 根据片段匹配、物性软距离和结构多样性选取词表，保留探索名额。局部有效样本不足时，所有性能偏好归零，退回多样性优先，并明确记录 fallback。

仍有算法超参数（带宽、证据阈值、通道权重、多样性及探索比例），它们集中在 task.options 中；“自动化”不是无参数，而是不再人为指定某类化学结构应得几分。独立片段可能有重叠，匹配分数采用均值以减少复杂分子重复累加偏置。

## 数据与适用边界

当前适配器为 `bamboo_prediction_export_v1`，读取现有 predict.py 的联合解释导出。必须同时含 σ/r 解释预测、fragment_attention 中的片段 Δlnσ/Δlogit(r)、盐条件及 molecular_property_profiles。model_id 由使用者声明；实验入口另记录实际 oracle 文件哈希，原始预测 JSON 本身无法证明对应哪个 checkpoint。

当前分子属性预测入口使用 25°C，因此第一版显式限制 25°C，避免错用其他温度。σ/r 目标、浓度、盐内比例可以条件化；换盐种需要匹配的新拟合数据和盐词表。下一步扩展温度时需要温度感知的属性目录/预测适配器。MA pipeline 评价容差目前固定为 σ=2 mS/cm、r=0.1，入口会拒绝不同容差；独立候选池支持其他容差。

规则偏好给定盐内比例，但现有 synthetic 生成器仍按共同的双盐比例先验采样；这不是固定盐比例的生成约束。应在最终配方评价中另审计盐比例偏离。G1/G2 默认不使用 minimum_dc/maximum_vis 硬门槛。G1 从公共有效候选池随机排序并进行多样性选择；G2 使用训练解释拟合的片段分数和物性软窗口，不按窗口硬删除候选。

## 执行顺序

在 GPU 环境导出真实训练侧配方的联合解释，明确提供数据目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash submit/generation/export_conditional_rule_fit.sh \
  /path/to/actual_training_dataset run/rule_fit/train_predictions.json
```

编辑 `config/screening/conditional_g2.example.json`。source.path 相对 task 文件所在目录解析；示例路径指向上一步输出。不要用现有 ft_test_predictions.json 代替。

候选只来自 Bamboo-Mixer 原始分子预训练库，默认路径为 `../bamboo_mixer/run/mono_data.json`（相对仓库根目录）。可通过 MONO_DATA 指定同类预训练库路径；不再读取 GDB17，也不使用 emb_dict/solvent_like_candidates.json 的手工筛选子集，不追加发布版 62 溶剂词表。资格筛选后冻结分子模型预测物性，G1/G2 共用一个缓存目录和相同资格门槛。

G1-MA 与 G2-MA 均在本任务内训练各自的 decoder 与 MA diffusion，二者互不依赖，不再要求先提交 G2-Q。执行单组、单 seed：

```bash
# G1 与 G2 共用同一份 task、目录和实验参数。
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G1G2_MA.sh \
  G1-MA 123 config/screening/conditional_g2.example.json

# G2-MA 也直接训练自己的 decoder。
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G1G2_MA.sh \
  G2-MA 123 config/screening/conditional_g2.example.json
```

456/789 使用相同配置重复上述命令。G1-MA 与 G2-MA 可独立运行。G2-Q 仅作为可选的无 MA 对照；如需 G2-MA-shuffle，先完成同 seed 的 G2-MA，它会共享该 MA 的 decoder。末尾添加 `--plan` 可仅检查输入、计算实验指纹和打印目录，不训练。不会自动申请 GPU 或其他调度资源。

实验写入 `run/generation_conditional_bamboo_ma/<指纹>/`。目标、拟合数据、候选目录、模型、代码或已配置的 pipeline 参数变化会生成独立目录；仅组名/训练 seed 不改变共享实验目录。规则/候选池产物还独立核验输入及输出哈希，拒绝沿用不匹配缓存。请在一套实验结束前冻结代码，避免每次运行进入不同目录。

## 输出及检查

G2 共享候选池新增 `conditional_rules.json`，包含条件、来源哈希、有效样本数、片段分数和标准误、物性窗口及回退原因。`candidate_pool_scores.csv` 给出每个入选分子的分项得分、命中片段、探索标记和前序相似度。`conditional_manifest.json` 保护缓存；输出 solvents.json/salts.json 仍可直接用于旧嵌入、synthetic 和 MA 流程。

先检查是否发生 fallback、词表多样性及规则证据，再比较三个 seed 的 G1-MA/G2-MA 联合目标误差和覆盖率。CPU 合成夹具测试验证程序行为，不能证明真实筛选收益。本次没有用测试集拟合，也没有启动大规模 GPU 实验。

## 为什么旧入口要求 G2-Q

旧 GDB 脚本中 G1-MA 自行训练 decoder，G2 为 Q/MA/shuffle 消融而共用 Q 的 decoder。这是脚本组织方式，不是 MA 算法要求。新 Bamboo 分子库入口采用对称的独立 MA 训练路径；保留旧 GDB 脚本仅用于历史实验复现。

## 无固定物性门槛的新版对照

默认不再要求预测介电常数 ≥ 8、预测黏度 ≤ 4。提交命令不变；为避免继承旧环境配置，提交前可执行：

```bash
unset COMMON_MINIMUM_DC COMMON_MAXIMUM_VIS
```

只有明确需要实验约束时才设置这两个环境变量，并在 G1/G2 全部 seed 中保持一致。例如 `COMMON_MINIMUM_DC=8 COMMON_MAXIMUM_VIS=4` 可显式恢复旧门槛，但不能称为无固定门槛对照。

公共检查仍包含中性单组分、3–40 个重原子、支持的元素、可解析/映射结构，以及当前 Bamboo 适配器所需物性预测完整且有限。这些是当前实现的准入范围，不是自动学习的规则；本次没有实现任意新目标属性的预测适配。迁移到不同预测模型/任务时仍需审核这些范围及属性接口。

此前目录有 189,107 个基础候选；去除两个阈值后是否全部通过，取决于数值有效性。实际数量查看 `candidate_pool_metadata.json` 的 `common_eligibility.eligible_candidates`，默认 `minimum_dc`、`maximum_vis` 应为 `null`，不再预设为 6,000。最终词表仍为 62，G1 带多样性控制，不是均匀无约束抽样。

代码变更会自动生成新实验指纹，重新建立词表、合成数据和训练结果。不要将旧的 6,000 候选池或旧生成器复制进新目录；旧结果留作带门槛对照。G1/G2 应在同一新版指纹下比较。共享 MA 数据的文件检查也已修复，避免已有数据被误判为缺失后重复生成。

## 分组提交入口

两个入口均运行 MA，默认使用 `config/screening/conditional_g2.example.json`，不申请调度资源：

```bash
unset COMMON_MINIMUM_DC COMMON_MAXIMUM_VIS
# 每一行是一个独立任务；GPU 编号按算力池分配设置。
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G1_MA.sh 123
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G1_MA.sh 456
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G1_MA.sh 789
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G2_MA.sh 123
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G2_MA.sh 456
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_conditional_G2_MA.sh 789
```

先完成训练集解释导出，然后提交 G1/G2；两组互不依赖，同组 seed 也没有顺序依赖。在各自分配的计算资源内可并行提交，勿将六个训练任务挤在同一块 GPU 上。可先各提交 seed123，确认公共准备完成后再提交456/789，以减少准备阶段等待。

自定义配置：`bash submit/generation/run_conditional_G2_MA.sh 456 path/to/task.json`。只查看计划：`bash submit/generation/run_conditional_G1_MA.sh 123 --plan`。两组须使用同一 task 和环境设置。

完整标准输出和错误输出保存于 `run/submission_logs/`，日志文件名含组名、seed、时间和进程号，失败保留非零退出码。同指纹下再次提交相同命令会复用完整共享数据，跳过 FINISHED 训练，从已有周期 checkpoint 续跑未完成训练；只有 optimal.pt 而无周期 checkpoint 的未完成训练会明确拒绝继续。相同组/seed 的重复运行由锁拒绝，不同 seed 可并行；等待共享准备锁会打印提示。进程被算力池强制终止时，仍需结合调度器退出状态判断原因。

新脚本及公共脚本的修改会改变实验指纹，首次使用进入新目录；不要将“新实验”误认为旧指纹下的续跑。整个六任务批次应保持代码和环境配置不变。
