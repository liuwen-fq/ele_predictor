# G0 三组工程重构复现

这是保持历史方法不变的第一阶段重构。保留原有 synthetic 标签、MA 残差化、格内 shuffle、训练和采样实现；不包含新条件化分子筛选或异常标签策略变更。共享 oracle 加载实现位于 `formula_design/predictor/oracle_adapter.py`，当前只支持原 Bamboo 加载语义，尚非通用模型插件。

## 配置与运行

编辑 `config/reproduction/g0_legacy.json`：

- `source_q`：已完成历史 G0-Q 的目录，含 generator_data 和 candidate_pool。
- `reference_root`：包含 `G0_seed_<seed>/generation_experiment_summary.csv` 的历史目录。
- `output_root`：独立的新输出目录，不得与历史目录互相包含。
- `seeds`：应与实际拥有的历史 seed 一致；示例为 123、456、789。
- `environment`：原流水线的显式参数。运行机器须有原 bamboo conda 环境与 CUDA。

从仓库根目录执行：

```bash
bash submit/generation/verify_G0_reproduction.sh plan
bash submit/generation/verify_G0_reproduction.sh preflight
bash submit/generation/verify_G0_reproduction.sh run
bash submit/generation/verify_G0_reproduction.sh compare
```

自定义配置：`bash submit/generation/verify_G0_reproduction.sh --config /path/config.json plan`。

run 顺序执行每个 seed 的 Q、MA、shuffle。Q 复用固定历史数据并重训 decoder/diffusion；另两组共享新 Q 的数据来源、词表和 decoder，仅按旧实现增加真实或打乱的辅助标签。每组日志在新 seed 目录。run 不自动提交调度作业；应在分配了 GPU 的计算环境执行。

首次运行记录输入数据、checkpoint、配置及源码 SHA256；恢复时不匹配会拒绝继续，需使用新的 output_root。不追认已有且无 manifest 的运行目录。源 Q 目录只读复用；原 MA 数据准备脚本负责构建新 MA 数据。原始脚本入口继续保留，直接运行它们不获得新入口的缓存保护。

## 验证口径

compare 要求全部配置 seed 和三个组的 CSV 存在，检查配置列出的指标，写入 `reproduction_comparison.json`，失败返回非零退出码。默认 1e-6 绝对、1e-5 相对容差用于严格数值回归，不能据此断言多 seed 的统计等效。CUDA 重训可能不逐位一致；若数值检查失败，应结合原多 seed 波动评估，不能事后放宽容差来宣称复现。

已有 checkpoint 的逐样本预测对比与生成配方组成审计仍应单独执行；当前比较器验证汇总指标，不替代逐样本检查。历史与新结果必须使用相同评价脚本、目标、样本预算和列定义。仅三组优劣排序相同不算复现。

默认历史目录已校正为 `run/generation_mechanism_auxiliary_comparison/G0-Q-comp`。训练配置生成时会写入本次运行的 `configs/dataset.yaml`，将 `save_dir` 绑定到实际数据位置，避免历史 YAML 内旧 `G0-comp` 路径失效；不会修改共享源文件。没有完整 GPU 重训结果前，不应宣称已复现。

## 按组和 seed 独立运行（推荐）

与目录中其他 runner 一样，使用 Bash 激活 conda 并运行 Python。脚本不调用 sbatch，不设置 GPU 数量、CPU、内存或运行时限。在已有 GPU 运行环境中执行，使用 CUDA_VISIBLE_DEVICES 选择可见 GPU；如果集群要求调度分配资源，仍需通过集群自己的方式获得资源。

先配置历史路径，再分别启动 Q：

```bash
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_Q_reproduction.sh 123
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_Q_reproduction.sh 456
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_Q_reproduction.sh 789
```

每个 seed 的 Q 完成后，分别运行它的 MA 与 shuffle：

```bash
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_reproduction.sh 123
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_shuffle_reproduction.sh 123
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_reproduction.sh 456
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_shuffle_reproduction.sh 456
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_reproduction.sh 789
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_shuffle_reproduction.sh 789
```

这些示例逐条前台执行，不会自动把九个任务同时放到 GPU 0。有多张可用 GPU 时可以分配给独立进程；要后台运行可自行加 nohup 和日志重定向。依赖关系仅为：

```text
Q(seed=123) 完成 → MA(123) / MA-shuffle(123)
Q(seed=456) 完成 → MA(456) / MA-shuffle(456)
Q(seed=789) 完成 → MA(789) / MA-shuffle(789)
```

三个 Q 都读取配置中的固定历史 Q 数据，互不依赖；无需等其他 seed 完成。同 seed 的 MA 与 shuffle 无先后依赖，可以在资源充足时同时运行。入口检查同 seed Q 的 FINISHED、checkpoint 和生成结果，不满足则退出，不会自动启动 Q。组锁避免重复启动；MA 运行时对 Q 使用共享锁，防止同时重写 Q。配置与源码指纹仍受校验，运行期间不要修改配置或源码。

每次只运行指定任务，不自动运行其他组或 seed。可选第二个参数指定配置：

```bash
bash submit/generation/run_G0_Q_reproduction.sh 123 /path/task.json
# 等价的通用入口：
bash submit/generation/submit_G0_reproduction.sh G0-Q 123 /path/task.json
```

训练日志在 `<output_root>/G0_seed_<seed>/<group>.log`，每组完成后执行原有 seed 内分析。全部完成后，激活 bamboo 环境，再汇总与比较（示例对应默认 output_root）：

```bash
python3 scripts/generation/aggregate_generation_replicates.py \
  --replicate-root run/reproduction/G0 \
  --groups G0-Q G0-MA G0-MA-shuffle --baseline G0-Q \
  --primary-treatment G0-MA --counterfactual G0-MA-shuffle
bash submit/generation/verify_G0_reproduction.sh compare
```

默认仍为 500 epoch 的历史复现实验，不自动延长到收敛。要延长训练，使用新的配置与独立 output_root。

## 2026-09-22 中断恢复

已确认五个 MA 任务留有 epoch 400 完整 checkpoint；456 的 MA-shuffle 已早停完成训练，仅待生成/评分。多组日志在 2026-09-21 07:41 左右中断，无 OOM traceback，无法仅凭这些日志确认内存耗尽。

修复了 `train_or_resume` 对 optimal.pt 的误判：优先检查完整 epoch checkpoint，使用 --restart 恢复模型、优化器、调度器及训练历史；FINISHED 组继续跳过训练。不改变 batch size、损失或数据。当前 checkpoint 不保存随机数发生器状态，因此中断恢复不保证与连续训练逐位一致。

商汤 GPU 运行环境可串行续跑六组，以降低并发资源需求：

```bash
CUDA_VISIBLE_DEVICES=0 bash submit/generation/resume_G0_MA_reproduction.sh
```

也可以继续用原来的单组入口，例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_reproduction.sh 123
CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_G0_MA_shuffle_reproduction.sh 456
```

本次维护已校验历史配置与输入哈希，保留每个 seed 的 `reproduction_manifest.before_resume_fix_20260922.json`，用 `resume_fix_20260922.json` 记录仅续跑入口修复的指纹迁移；没有取消后续指纹检查。迁移后的 manifest、更新脚本需与算力池使用同一共享文件树。

进一步排查发现模型权重约 65 MB，而 epoch 400 checkpoint 约 13 GB。旧训练循环直接保存带计算图的 loss 张量，历史记录可能持有大块底层 storage。现改为记录 detach 后独立的 CPU 标量，并在恢复旧 checkpoint 时使用 CPU mmap 和历史标量复制，避免把全部旧历史载入 GPU。该缺陷是内存增长的明确风险，但缺少平台退出码，仍不能确认本次外部中断的唯一原因。训练器修复亦已记录在上述迁移审计中，旧 checkpoint 不覆盖。

## 2026-09-22 二次恢复：Bus error 修复

前一次 CPU mmap 优化与原有“恢复后立即原地保存同名 checkpoint”不兼容：写入会截断仍被映射的文件，导致 SIGBUS/退出码 135。已把完整 checkpoint 和 optimal.pt 的保存统一改为同目录临时文件写入、flush/fsync 后 os.replace，避免截断映射源；保存失败时旧文件保持不变。

此次五个 epoch 400 checkpoint 已被截断，不能再用。现有恢复选择器会跳过损坏的 400，回退到 epoch 350；无需手动删除损坏文件。456 / G0-MA-shuffle 已完成生成分析，无需重跑。完整 checkpoint 会恢复优化器、调度器和训练状态，但损失 350 之后的未保留训练进度。

修复通过独立子进程的 mmap→同路径保存回归、写入失败保护及损坏最新 checkpoint 回退测试。每个 seed 的旧 manifest 已备份为 `reproduction_manifest.before_atomic_fix_20260922.json`，迁移记录为 `atomic_checkpoint_fix_20260922.json`。在算力池同步修复后，可使用原单组命令继续；当前本地没有 GPU，未进行 GPU 续训。
