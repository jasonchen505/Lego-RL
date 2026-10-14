# Lego-RL 深入学习文档

- 学习日期：2026-10-14
- 仓库：`jasonchen505/Lego-RL`（fork，上游 `LegoX/Lego-RL`，Apache-2.0）
- 学习时状态：fork 与上游完全同步，main HEAD `0731c95`（2026-10-08 merge PR #24）
- 上游规模：117 stars / 9 forks，2026-07-28 创建；0 open issue，1 个 open PR（#25，eval 支持 DeepSWE 任务）
- Paper：[LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents](https://arxiv.org/abs/2608.17393)（arXiv:2608.17393，v1 2026-08-18，第一作者 Yuxin Jiang）
- 文档：https://lego-rl.pages.dev/docs；训练 Dashboard：https://lego-rl.pages.dev（7 个公开运行）
- 本文只记录学习笔记，不含任何 upstream PR 结论。

## 0. 一句话

Lego-RL 是让 coding agent 在**原生 harness 里零改动**跑在线 RL 的框架：Claude Code、OpenHands SDK、OpenCode、Codex 各自的控制流完全不动，只把 base URL 重定向到一个 in-process proxy；proxy 把请求转给 vLLM 上的策略模型，同时做 token 级轨迹捕获；verl 负责训练；reward 是 Harbor 沙箱里任务自带 verifier 的二值信号（SWE-bench 式，fail-to-pass 全过且 pass-to-pass 全保才给 1）。

## 1. 核心问题：原生 harness 与 RL 训练的三处错位

Paper 摘要把问题说得很清楚，代码和文档把每一处都落成了机制。

**第一，奖励信号不可靠。** 沙箱崩溃（environmental crashes）会产生假失败，agent 也能 hack 奖励（例如把 repo 的 GitHub 历史重新 clone 回来，直接恢复出修复 patch）。Lego-RL 的回答：沙箱 + stage-wise 反作弊防御 + 失败 trial 的过滤（见第 7 节）。

**第二，train-inference 不一致会静默破坏 loss。** 训练算 loss 的 rollout 不是 trainer 自己生成的。重分词（retokenize）、重建 prompt、MoE 路由重采样，都会让梯度指向一个从未运行过的策略，而 loss 看起来依然合理。Lego-RL 的回答：token-faithful 捕获 + rollout fidelity 检查 + R3 routing replay（见第 5 节）。

**第三，harness 的控制流不能被改。** Agent 的上下文管理、工具循环、压缩（compaction）、子 agent 是产品行为，改了就不再是"部署时跑的那个 agent"。Lego-RL 的回答：只改 base URL，harness 本体零改动（见第 3 节）。这也是它和"把 agent 重写成一个 verl agent loop"路线的根本区别。

## 2. 整体架构

数据流自上而下是请求流，自下而上是数据与权重流：

| 层 | 角色 | 关键文件 |
| --- | --- | --- |
| Environment | 沙箱执行 + verifier；scaffold 与后端 | `src/harbor_patch/`，`docs/content/docs/architecture/environment.mdx` |
| AgentLoopWorkers | 并发跑 scaffold，每 trial 捕获一个 rollout | `src/verl_patch/agent_loop/builtin_swe_agent_loop.py` |
| In-Process Proxy | 统一 OpenAI/Anthropic API，捕获 token 级轨迹 | `src/verl_patch/agent_loop/vllm_chat_completion_proxy.py`（2339 行） |
| Global Load Balancer | 把请求分散到推理副本，sticky per-session 路由 | `docs/content/docs/architecture/global-load-balancer.mdx` |
| Session-Level Scheduler | 可选的准入控制，KV 压力大时保护 prefix cache | `docs/content/docs/architecture/session-scheduler.mdx` |
| Rollouter | vLLM 推理服务器池、rollout 数据缓冲、weight sync | `docs/content/docs/architecture/rollouter.mdx` |
| Trainer | verl actor，FSDP / VeOmni / Megatron；PPO / GRPO / GSPO / SAO；sync / async / partial rollout / R3 | `docs/content/docs/architecture/trainer.mdx` |

一次训练 step（sync 模式）：

1. Trainer 采样一批任务，交给 AgentLoopWorkers。
2. 每个 worker 在 Environment 沙箱里启动一个 scaffold，驱动它完成任务；每次模型调用都走 In-Process Proxy。
3. Proxy 经 Global Load Balancer 转到 Rollouter 的某个 vLLM 副本，并记录精确 token。
4. Trial 结束，verifier 打分；完成的 rollout 进入 Data Buffer。
5. Trainer 消费 buffer，算 advantage，更新 actor，经 Weight Sync 把新权重推回 rollouter；下一 step 开始。

术语约定（`core-concepts.mdx`）：**trial** 是一次沙箱执行，**rollout** 是从这次 trial 捕获的 token 序列，一 trial 产一 rollout。一个 step 跑 `TRAIN_BSZ × N_RESP` 个 trial，默认 `64 × 8 = 512`（64 任务 × 8 次尝试）。Turn 数和响应长度是 agent 自身的属性，不是配置项。

## 3. 核心机制一：In-Process Proxy（零改动接入）

这是整个框架最关键的"接缝"（single seam）：**一个进程、一个统一 API、精确 token 级捕获，没有独立 LiteLLM，没有固定端口。**

- 由 verl agent loop 在训练主机上启动，绑定**临时端口**，给每 trial 一个唯一的 **per-session URL**。
- 同一个进程同时服务两种协议（`vllm_chat_completion_proxy.py` 的路由表）：
  - `POST /sess/{session_id}/v1/chat/completions`（OpenAI）
  - `POST /sess/{session_id}/v1/messages`（Anthropic）
  - `GET /v1/models`
- scaffold 零改动，只重定向 base URL：
  - Claude Code：`ANTHROPIC_BASE_URL=http://<host>:<port>/sess/<id>`
  - OpenHands / OpenCode / Terminus：`OPENAI_API_BASE=http://<host>:<port>/sess/<id>/v1`

因为 proxy 转发给 verl 的 `server_manager` 而不是直连 vLLM，vLLM 的 abort/retry 和 fully-async 的 **partial rollout**（权重同步打断的生成可以续写而不是丢弃）都在穿过 proxy 时被透明处理。

## 4. 核心机制二：轨迹捕获与 segment

实现在 `src/verl_patch/agent_loop/trajectory.py`（`TrajectorySegment` / `TrajectoryRecorder`）和 proxy 里。核心思想：**捕获与 harness 可变 transcript 无关的生成 token，只在采样时的原始左上下文下训练。**

- `TrajectorySegment` 存 `prompt_ids`、`response_ids`、`response_mask`、`response_logprobs`、`response_routing`，以及 `generations`（每次真实生成的边界与策略版本）。
- **mask 语义**：只有采样出的 token 才 `response_mask=1`；工具观测（tool observations）、模板尾巴、新 prompt 里的历史 assistant 消息 mask 都是 0。
- **append 延伸当前 segment；改写过的 engine prompt 开启新 segment**，早期的 action 保留在原始上下文之下。例如：
  - segment 0：`Q A1 O1 A2 O2 A3`，训练 A1、A2、A3
  - segment 1：`Q A1 Z A3 O3 A4`，只训练 A4
- 旧的"prefix-logprob 重算"路径已被删除：因为在改写过的前缀下给 A3 打分，无法恢复它原本的采样上下文。精确重试或替换最新 assistant 回复时，只有替换成功才回滚那一次生成；丢弃的生成保留下来供检查。
- 只有 proxy 显式的 last-assistant 重试路径会调用 rollback；任意上下文改写永远不会推断"早期的模型动作被放弃"。
- Trial 结束时 `proxy_trajectory.json` 存所有 `trajectory_segments`、丢弃的重试、最新的 messages/tools。
- **子 agent 隔离**：初始会话身份的改动会路由到一个不参与训练的子会话。多模态轨迹捕获和任意分支身份不在这个路径的范围内。

**Segment 选择**（训练时用哪些 segment，`HARBOR_TRAJECTORY_SELECTION`）：

- `longest`（默认）：选截断后保留模型 token 最多的 segment，平局取后者；一 episode 一行，现有训练路径不变。
- `all`：训练所有带保留模型 token 的 segment。GRPO 里每 episode 的 reward 只计一次，在 query group 内广播 advantage。每个 optimizer minibatch 包含 `ppo_mini_batch_size * rollout.n` 个原始 episode（含它们全部被选 segment）。序列均值 loss 按 episode 计数；GSPO 保持上游序列均值目标并用 episode 归一化，但 ratio 和 clipping 仍按 segment。验证集仍一 episode 一行，验证奖励不受训练截断影响。

## 5. 核心机制三：Rollout fidelity

这是"loss 是否有效"的判定层，三个手段按检查深度递进。

**（1）Logprob 相关性检查。** 对每批数据，比较生成时 serving 引擎给出的 logprob 和训练时重算的 logprob 的相关性。健康值 ≥ 0.99，下降是链路坏掉的第一个症状（`docs/content/docs/troubleshooting/training-startup.mdx`）。要求：同样的 token id、同样的 mask、同样的权重，对稀疏模型还要同样的 expert 路由。

**（2）R3（routing replay）。** MoE 的情况：生成时 router 选了哪些 expert 会被记录下来，训练 forward 时原样 replay，两边对每个 token 的路由完全一致。`ENABLE_R3` 默认由 `MODEL_PATH` 是否为 MoE checkpoint 决定，dense 模型自动关闭；只有需要覆盖默认值时才显式设置。对应论文 Ma et al. 2025《Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers》（arXiv:2510.11370）。

**（3）Importance correction（`ROLLOUT_IS`）。** rollout 和训练 logprob 有结构性差异时（不同引擎、async 下的陈旧权重），做修正而不是假装不存在，分 token 级和序列级。序列权重是每 token 比率的乘积，方差随响应长度增长。实测：`ROLLOUT_IS=sequence` 在这种长 rollout 上有效样本量（ESS）约 0.06，batch 坍缩到少数几个样本，梯度饿死；所以 token 级是默认值。诊断指标就是 ESS：权重实际留下了多少 rollout 在起作用。

## 6. 核心机制四：Harness 解析与混合 harness

实现在 `src/verl_patch/agent_loop/harness.py`，**刻意没有任何 Harbor 或 verl import**，让 harness 配置归一化可以独立于 trial 创建做校验。关键结构：

- `HarnessDefinition`：一个名字 + Harbor 配置 + HTTP ingress 协议（`openai` / `anthropic`）。
- `HarnessResolver.resolve()` 按优先级解析到底用哪个 harness：
  1. **validation**：验证时固定一个 harness（混合运行默认 `openhands_sdk`），保证验证指标跨 step 可比；
  2. **metadata**：数据集元数据的 `agent_harness` / `harness` 字段（直接转发列优先于 `extra_info` 里的嵌套值）；
  3. **weighted_random**：按权重随机选（`granularity=task` 时用 `seed + global_step + task_identity` 的 SHA-256 做稳定抽样；`granularity=trajectory` 时用真随机，因为 fully-async worker 拿不到稳定的 sibling-rollout 身份）；
  4. **default**：回退默认 harness。
- 协议判定：描述符（名字 / agent name / import_path）里含 `claude` 即 Anthropic，否则 OpenAI。
- **render contract**：声明式的 endpoint 模板，把 `{llm_base}`、`{served}`、`{env:XXX}` 等 token 渲染成真实 endpoint / model 名 / kwargs / env。未知 token 直接报错，不静默。
- **env 注入**（`inject_harness_endpoint`）：按协议写不同的 env 变量。Anthropic 协议时 `ANTHROPIC_BASE_URL` 不带 `/v1`（Claude Code 自己会追加 `/v1/messages`）；OpenAI 协议时写 `LLM_BASE_URL`，OpenCode 额外写 `HOSTED_VLLM_BASE_URL`，Codex 额外写 `OPENAI_BASE_URL`。agent pod 在推理前会切换到 egress allowlist，所以把 proxy 主机精确加进 `extra_allowed_hosts`，而不是放开整个子网。
- 校验很严：`harnesses` 和 legacy `harbor_cfg` 二选一，不能同时配；已知 Harbor agent 的 `agent.name` 和 `import_path` 同时出现时报错（Harbor 会忽略 import_path）。

混合 harness 训练（2026-09-11 上线）：一次运行交错 OpenHands SDK、OpenCode、Claude Code，每个轨迹按数据集元数据或加权策略解析一个 harness，验证固定在一个 harness。README 的 mixed run 用的是 "GSPO with R3 router replay"。

## 7. 核心机制五：沙箱与反作弊

基于 [Harbor](https://github.com/Elvin-Yiming-Du/harbor)，`src/harbor_patch/` 里是定制的 agent 与 environment：

| 后端 | 执行单位 | 用途 |
| --- | --- | --- |
| Kubernetes | 一 trial 一 pod | 生产默认，大并行 |
| Local / Remote Docker | 一 trial 一容器 | 最小 setup，无集群 |
| 其他 Harbor 后端（GKE、Daytona、E2B、Modal、Runloop） | 托管集群 / 沙箱服务 | 经同一 import path 继承，未在本栈验证 |

Scaffold（`src/harbor_patch/agents/`）：Claude Code、OpenCode、OpenHands、Terminus 2 是端到端验证过的严格适配；Codex 有独立的 `image_mounted_codex`。scaffold 决定 rollout 的形状和 Harbor 启动哪个运行时镜像，用 `HARBOR_AGENT_IMPORT_PATH` 选择（默认 `harbor_patch.agents.image_mounted_claude_code:ClaudeCode`）。

**让沙箱又快又可信的机制：**

- **Nydus cold start**：懒加载容器镜像，pod 在整个镜像拉完之前就能用。
- **Agent-runtime image mounting**：scaffold/runtime 预烘进 agent 镜像并挂载进去，而不是每 pod 安装（无 egress 的 task pod 上安装会失败）。
- **Image cache**：跨 trial 共享镜像层，减少启动时间。
- **反作弊**：防 agent 用恢复修复 patch 的方式 hack 奖励（例如把 repo 的 GitHub 历史重新 clone 回来）；paper 摘要称之为 stage-wise defenses（本学习未逐项核对代码实现）。
- **基础设施失败过滤**：trial 以基础设施失败结束的 rollout，在进 loss 之前被 trajectory filter 丢掉，所以到达 optimizer 的 batch 比启动的 batch 小。

## 8. 训练算法

`adv_estimator`（advantage）与 `policy_loss_mode`（loss）两个旋钮决定算法。默认 profile 是 `adv_estimator=grpo` + `policy_loss_mode=gspo`。三种算法都优化**完整的跨 turn rollout**，不是单次补全。

### 8.1 GRPO + GSPO（默认）

GRPO（Shao et al. 2024）：每 rollout 的 advantage 是它与同任务其他尝试的对比，不需要 critic：

```
A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)
```

二值 reward 的直接后果：一个任务的所有尝试全解或全败，std=0，**零梯度**，无论烧了多少 GPU 小时。Group 只有落在两个极端之间才携带信号。pool 构成而非 batch size 决定一个 step 有多少训练量，这正是第 9 节任务过滤要筛选的东西。

GSPO（Qwen Team 2025）：在序列级 importance ratio 上 clip，用响应长度归一：

```
s_i = (π_θ(y_i|x) / π_old(y_i|x))^(1/|y_i|)
```

`1/|y_i|` 指数把 clip 界放在"平均每 token log-ratio"的单位上。这就是为什么 `clip_ratio_low` / `clip_ratio_high` 在这里是 1e-4 量级，而且响应从 43k token 涨到 90k token 时不需要重调。

PPO（Schulman et al. 2017）：同样的 clip 形式用在每 token ratio 上，advantage 由 critic 给。

### 8.2 SAO（single-rollout，critic-based）

Paper：[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)。用学到的 value 模型替代 GRPO 的 group baseline，所以每任务每 step 只要 **1 个** rollout；轨迹一完成就进入训练，不等组内最慢的成员，适合 fully-async 下长短不一的 agent episode。

Lego-RL 以模板模块 `scripts/templates/verl/sao.env` + `patches/verl_sao_7aed6b23.patch` 的形式提供。验证过的配方：Qwen3.5-35B-A3B 从 0.640 到 0.682（SWE-bench Verified），180 步。

六件套：

| 件 | 内容 | 旋钮 |
| --- | --- | --- |
| Single rollout | `n=1`，GAE advantage 来自 critic 而非 group mean | `N_RESP=1`，`ADV_ESTIMATOR=gae`，`CRITIC_ENABLE=True` |
| Decoupled GAE λ | policy λ 用 VAPO 的 length-adaptive 形式（`1 - 1/(α·len)`，α=1.5），bootstrap horizon 随轨迹长度伸缩；critic 回归真实 MC return（`lam_critic=1`） | `LENGTH_ADAPTIVE_LAM_ALPHA`，`LAM_CRITIC` |
| DIS | 双边 token 级 importance sampling：bypass 模式（`old_log_prob := rollout_log_prob`），每 token ratio `r_t = π_θ/π_rollout`，trust region [0.2, 4.0] 之外的**权重直接置零**而非 clip，REINFORCE loss `−E[w·A·log π]` | `ROLLOUT_IS=token`，`ROLLOUT_IS_THRESHOLD=0.2_4.0`，`ROLLOUT_CORRECTION_LOSS_TYPE=reinforce`，`POLICY_LOSS_MODE=bypass_mode` |
| 更快的 value 更新 | 每 actor step 做两次 critic optimizer step | `CRITIC_PPO_EPOCHS=2` |
| Skip-observation GAE | 工具输出不是策略的动作，TD 误差只跨模型生成的 token 传播 | verl 的 GAE 按 `response_mask` 处理，无需设置 |
| Frozen-attention critic | value 模型只训练 MLP / expert 投影，attention 冻结 | `CRITIC_FREEZE_PARAM_PATTERNS` |

无参考 KL 项：paper 既不用 `use_kl_loss` 也不用 reward KL；关掉两者意味着不创建 reference-policy worker，省下的内存正好给 critic 用。DIS 限制短期漂移，不是参考 KL 的替代品，加回去是一个新的 ablation，不是修复。

**模板顺序是 load-bearing 的**：`verl/sao.env` 必须在 `TEMPLATE_MODULES` 里排在 `verl/common.env` **之前**。模板用 `: "${VAR:=default}"` 赋值，先赋值者胜；排在 common 之后，SAO 的每个默认值都是静默 no-op，跑出来的是一个挂着 SAO 名字的 GRPO 运行。preflight 规则 10 会拒绝错误顺序。

**Critic 初始化是每轮必须做的决定：**

- Cold start：`CRITIC_MODEL_PATH` 指向 `MODEL_PATH`。只有 value head 是随机的；预期 10–20 步 noisy advantage，`CRITIC_WARMUP ≥ 10`（期间 actor 冻结，让 critic 校准）。paper 称 value cold start 是主要瓶颈。
- Warm start：用 `utils/merge_critic_ckpt.sh` 合并早期的 critic 再指向它。35B 运行上这是最大的稳定器。早于策略病理之前样本上训出的 critic 是"弱但无偏"的，可以安全复用。

**Qwen3.5-MoE 的坑**：Qwen3.5-35B-A3B 是混合架构（10 层全注意力 + 30 层 GatedDeltaNet `linear_attn` + vision tower）。只冻 `[self_attn.]` 只冻住 4% 的张量；验证过的配置用 `[self_attn.,linear_attn.,visual.]`。VeOmni 的模型注册表按子串分发 `<Family>ForTokenClassification`，不打 patch 会回落到 CausalLM 类：critic 被静默建成语言模型，GAE 看到 vocab 维度的"values"。`utils/apply_veomni_valuehead_patch.py` 修注册表并安装 `src/verl_patch/models/` 里的 Qwen3.5-MoE token-classification sidecar。

### 8.3 SAO 的 r6→r7 postmortem（最值得读的一节）

r6 在约 step 130 崩溃，因果链的每个环节都是一个旋钮：

1. **critic 被饿死**：`CRITIC_GRAD_CLIP=1.0`，而 `critic/grad_norm` 在 7–68 之间；每次更新都被缩放 7–70 倍，配置的 critic LR 是虚构的，`vf_explained_var` 在 0–0.3 停留了 130 步。
2. **无 baseline → advantage 失真**：`GAE_WHITEN_ADVANTAGES=False`（这个选择假设 critic 能提供 baseline），batch advantage 均值漂移 ±0.25，出现整批全负的 batch。
3. **触发点**：step 119–121 是连续三个全负 batch。全负 batch 惩罚一切，而与奖励无关的 no-op（再次调用 `think` 工具）被惩罚得**最少**，于是相对被强化。
4. **迷信复合**：`think` 占工具调用的比例 0.07 → 0.18 → 0.29 → step 130 时 **0.73**。策略停止编辑代码。

r7 与 r6 分隔的四个改动，每个切断一条链：

| 旋钮 | r6 | r7 | 原因 |
| --- | --- | --- | --- |
| `GAE_WHITEN_ADVANTAGES` | False | **True** | 结构上杜绝全负更新；critic 还弱时的 fallback baseline。return 在 whitening 之前构建，所以 value 学习不受影响。预期 `actor/grad_norm` 高 2–3 倍，那是尺度不是信号 |
| `CRITIC_GRAD_CLIP` | 1.0 | **10** | 典型梯度不被 clip，critic LR 说的是实话 |
| `CRITIC_MODEL_PATH` | actor（cold） | r6 的 critic @ step 100，合并后 | 弱但无偏，训练样本全部在 spam 之前 |
| `CRITIC_WARMUP` | 0 | **20** | warm critic 在 actor 能动之前先在 base policy 上重新校准 |

前两个现在是 `verl/sao.env` 的默认值。`vf_explained_var > 0.4` 持续约 20 步后，可以把 whitening 交回去。

行为上 r7 的模型改变的是**密度**不是长度：每轨迹 edits 4.7 → 8.0，内联 `python -c` 验证大幅上升，整套 `pytest` 下降，代码执行总量持平约 95%，turns 和响应长度持平。同模型 GRPO 会同时膨胀 turns 和长度（CoT +30%）；SAO 的 per-token GAE 没有长度偏置，所以"更长"不再是免费的。

**SAO 的陷阱清单**（`sao.mdx` 的 Traps 节，逐条都是实测教训）：

- Resume 会固定旧 LR：verl 恢复 optimizer **和** lr-scheduler 状态，配置里的 LR 改动在重启时被静默忽略（r7 有 119 步以为是 2.5e-6 实际跑在 5e-6）。想改 LR 做 resume 时设 `ACTOR_CKPT_LOAD_CONTENTS='[model,optimizer]'` / `CRITIC_CKPT_LOAD_CONTENTS='[model,optimizer]'`。
- 带时间戳的 `EXP_NAME` 破坏 resume：`resume_mode=auto` 找 `checkpoints/<project>/<exp>/`；每次启动一个新名字就从 step 0 重来，多节点下每个节点名字还不一样。
- `TOTAL_ROLLOUT_STEPS` 和 `TOTAL_EPOCHS` 是 AND 关系，rollouter 取 min。
- Token budget × SP 必须带余量覆盖窗口；`common.env` 的默认值正好卡在窗口上；一个 rank 在 `rearrange_micro_batches` 里 assert，剩下的在 `_compute_values` 里死锁到 `NCCL_TIMEOUT`。preflight 规则 10e 检查这个。
- DIS 镜像：`algorithm.rollout_correction.*` 不是 actor loss 读的；`hydra_args.sh` 会把同样的 key 强制写到 `actor_rollout_ref.actor.policy_loss.rollout_correction.*`。没有这个镜像，运行会静默训练 sequence-level TIS + PPO-clip loss，指标看起来还很合理。
- Critic checkpoint 不能直接加载：`global_step_N/critic/` 里是 DTensor shard，且 `config.json` 声称 `ForCausalLM`；`utils/merge_critic_ckpt.sh` 修这两处。流式 merger 拒绝 Qwen3.5-MoE（会损坏 fused expert）。
- Reward 位置：verl patch 把 outcome reward 放在最后一个**模型** token 上，而不是最后一个 attended token（通常是 tool observation，`response_mask` 排除它）。修之前约 0.6% 的 reward 信号到达 GAE。GRPO 不受影响，因为它在 token 上求和 reward。

### 8.4 读 SAO 指标

bypass 模式下 `old_log_probs == rollout_log_probs` 是构造出来的，所以 `training/rollout_actor_probs_pearson_corr` **不产出**（强制算会打印 1.000 但毫无意义）；`actor/entropy` 也不在（算它的 old-actor forward 被跳过了）。Rollout correction 跑在 actor loss 内部，所以指标带 `actor/` 前缀：

```
actor/rollout_corr/kl                          <= ~1e-2
actor/rollout_corr/chi2_token                  <= ~1e-1
actor/rollout_corr/log_ppl_abs_diff            <= ~1e-2
actor/rollout_corr/rollout_is_eff_sample_size  >= ~0.99
actor/rollout_corr/rollout_is_ratio_fraction_{high,low}   被 DIS 置零的 token 份额
```

阈值是 30B smoke 的回归基线，不是普适常数。阅读顺序：log-PPL gap 先，然后 k3 KL、χ²、ESS。ESS 是**相对** ESS，`(Σw)² / (N·Σw²)`，在 (0, 1]：1.0 表示每 token 权重相同。下降可能来自策略陈旧、训练/推理不匹配或 DIS 置零大量 token，它说的是多少梯度存活，不是哪一个原因。Fully-async batch 混合 `lag = 0` 和 `lag ≥ 1` 的轨迹；没有 frozen-weight、同版本的对照，不要把在线上升读成后端不匹配。

`critic/` 命名空间是历史遗留，不是 critic 输出：

| 指标 | 含义 |
| --- | --- |
| `critic/score/mean` | 原始 verifier 分数，过滤前，batch 成功率 |
| `critic/rewards/mean` | 进 GAE 的奖励；除非轨迹被过滤（奖励置零），否则等于 score |
| `critic/returns/*` | critic 的回归目标；γ = lam_critic = 1 且终端 0/1 reward 时，解出轨迹的每个模型 token 的 return 都是 1。**token 加权**，长轨迹占主导，不可与 `score/mean` 直接比 |
| `critic/values/*` | 旧 critic 的预测（本 step 更新前）；无界，cold start 时可低于 0 或高于 1 |
| `critic/vf_explained_var` | `1 − Var(return − value)/Var(return)`；低于 0 = 不如常数；上升 = critic 在学哪些状态会成功。用更新前 value 算，所以本 step 的 critic 更新体现在**后续** step。batch 只有一个 return 值时（全 0）退化为巨大负数，忽略这些 step |
| `critic/vf_loss`，`critic/vf_clipfrac` | clip 后的 value MSE；clipfrac 长期高说明 critic LR / epochs 太激进 |
| `critic/grad_norm` | 30B–35B 上中位数约 10，偶发 30–70 spike；报警条件是**连续三个 step 高于 30**，不是一次 spike |

健康检查五问：信号真实吗（`score/mean` vs `rewards/mean` vs `trajectory_filter/invalid_ratio`，两个都从 step 1 就近 0 是基础设施问题）；有东西可学吗（`returns/min ≈ max` 时本 step 的 explained variance 无意义）；critic 在收敛吗（`vf_explained_var` 应在约 20 步内离开负值区，稳定在 0.2–0.5，最好见过 0.64；50+ 步卡在 ≤0.3 且 grad_norm 远高于 clip 是 r6 特征）；更新合理吗（见上）；DIS 开着吗（四个 `actor/rollout_corr/*` 门限，`actor/pg_clipfrac` 不存在）。

## 9. 数据：任务 index 与任务过滤

数据在运行开始前全部备好。流水线：

```
raw data（repos, issues, patches）
  └─▶ Harbor task instances    一任务一目录：Dockerfile + instruction + tests
        └─▶ sandbox images    预构建，推到集群 registry
              └─▶ task filter  rollout-verified 难度（pass@k 选择）
                    └─▶ task index（.parquet） → TRAIN_INDEX / VAL_INDEX
```

**任务 index 是轻量指针表**，不是任务内容：

```python
{
    "prompt": [{"role": "user", "content": "/data/tasks/astropy__astropy-12907"}],  # 路径，不是文本
    "reward_model": {"style": "rule", "ground_truth": None},  # verifier 评分
    "extra_info": {
        "harbor_task_path": "/data/tasks/astropy__astropy-12907",
        "instance_id": "astropy__astropy-12907",
        "data_source": "harbor",
    },
}
```

任务内容留在共享存储，trial 真正跑时才加载；index 每千任务只有 KB 级，所以组合训练集就是纯 dataframe 工作：按难度带过滤、去掉污染实例、跨数据源 union、按模型做选择，都不需要复制或重写任务内容。转换工具：`utils/convert_swerebench_filtered.py`（SWE-rebench 风格记录 → Harbor 任务目录 + index），`utils/create_task_index.py`（从 Harbor 任务目录建 index）。

**任务过滤**（`Task filter`）：实例进训练池之前按 rollout 验证的难度筛选：用 batch inference（`scripts/infer/`）跑每实例 `N_TRIALS` 次尝试（训练温度），保留目标难度带的实例（例如 4 次解出 1–3 次，即 group-relative advantage 下梯度信号最强的那些）。输出本身也是一个 task index，直接指给 `TRAIN_INDEX` 或继续组合。

**沙箱镜像**：推荐预构建——每个实例的环境镜像离线构建并推到所有沙箱节点能拉到的 registry，设 `HARBOR_OPENSWE_IMAGE_REGISTRY`；镜像名 `<registry>/openswe-<instance_id>:latest`；nydus 这类懒拉格式进一步削冷启动。回退是 inline build（trial 在 pod 内现建镜像），可移植但慢且需要 egress。`INLINE_BUILD` 和 registry 同时配是 preflight fatal。

训练用的是 2,699 任务的 OpenSWE 派生 index（[Lego-X/Lego-RL-2699](https://huggingface.co/datasets/Lego-X/Lego-RL-2699)），验证用 SWE-bench Verified 的 500 任务。

## 10. 部署与运行

### 10.1 训练模式与后端

| `TRAIN_MODE` | 布局 | 用途 |
| --- | --- | --- |
| `sync` | Trainer 与 rollout 引擎共享一个 GPU 池 | 默认，最低成本 |
| `async` | 训练池与 rollout 池分离，并发运行 | 多节点，见 `run-training/scaling.mdx` |

Actor 后端：FSDP（dense 模型，验证过的默认和 fully-async 路径）、VeOmni（大 MoE，如 Qwen3-30B-A3B，设 `MODEL_ENGINE=veomni`）、Megatron（最大模型，经 tensor/pipeline 并行）。

Async 相关旋钮：`staleness_threshold=1`（rollout 最多落后几个参数版本）、`partial_rollout=True`（被权重同步打断的生成续写而非丢弃）。

### 10.2 配置分层与启动流程

所有工作负载（训练 / 评估 / batch inference）都走同一套五步：

```text
1. 建 venv          bash scripts/setup_env.sh              （每机器一次）
2. 描述集群        cp lib/site.example.env lib/site.env   （每集群一次）
3. 复制模板        cp train/templates/X.env train/configs/mine.env
4. 校验            PREFLIGHT_ONLY=1 然后 DRY_RUN=1      （只检查，不启动）
5. 启动            bash scripts/train/train.sh train/configs/mine.env
```

- **集群相关只在 `site.env`**：`MODEL_ROOT`、K8s kubeconfig、镜像 registry。run config 里没有任何集群地址，所以 run config 可以原样搬到同事的集群。
- **模板按轴命名**：`<mode>_<backend>_<scaffold>_<model>.env`（如 `async_k8s_ohsdk_qwen35a3b.env`、`sync_docker_oh_qwen3-30b-a3b.env`）。复制后只改 `CHANGEME` 行（实验名、数据 index、拓扑）。
- **参数分层**：模板没提到的旋钮加一行 `KEY=VALUE` 即可（见 `reference/configuration.mdx` 的 parameter tiers）。
- **setup_env.sh**：用 `uv` 建 venv，clone + editable 安装 `harbor` 和 `verl`（默认在 `<repo>/third_party/`），并做自检（import `harbor` / `verl` / `vllm` 必须解析到刚准备的 checkout）。`veomni` 是默认建模后端，它的 wheel metadata 固定 `datasets<=2.21.0` 与 verl/vllm 冲突，`overrides.txt` 经 `UV_OVERRIDE` 解除该约束；手装时不加 `--overrides overrides.txt` 会解析失败。
- **Preflight**：9 类 fatal 断言，零集群需求。挡下那些否则会浪费数小时运行的错误：tool_parser 与模型 × scaffold 不匹配、节点数对不上、SP_SIZE 不整除训练 world、没新 verl 树就开 R3、无可达镜像源。Agent 等价物是 `/rl:check <config>`（同样的断言 + 实时机器检查：已有运行在飞、陌生 GPU、被占端口）。
- **启动**：runner 拉起 Ray，启动 vLLM，启动 in-process proxy，驱动 PPO/GRPO/GSPO 循环。30B MoE 在 TP=4 时，vLLM CUDA-graph capture 要 10–20 分钟才有第一个 replica 注册，这是预期的。
- **先 smoke**：新模型 / scaffold / 集群先用 smoke 配置（`TRAIN_BSZ=8`，`N_RESP=4`），约 8 倍速走完整链路；梯度噪声大、reward group 退化，所以只是通路检查，不是学习质量信号。
- **监控**：`tail -F logs/<exp>.log`，`bash webui/start_dashboard.sh`（`http://<host>:8090`）。Checkpoint 在 `checkpoints/<project>/<exp>/global_step_N/`。

Agent plugin（`docs/content/docs/agent-plugin.mdx`）：`/rl:*` skills 给 Claude Code 用，校验配置、要求显式确认后再启动同一个 runner（`/rl:check` / `/rl:run` / `/rl:status` / `/rl:dashboard`）。`docs/content/docs/reference/status.mdx` 记录了各模块的状态。

### 10.3 运维文化

这个 repo 的文档最突出的一点是**把故障写成可查的资产**：`troubleshooting/` 里有 `val-false-zero`、`async-val-deadlock`、`offline-eval-garbage`、`rewards-and-throughput`、`sandbox-backends`、`training-startup` 等专题，每个都是实测故障的完整链路；SAO 文档里直接贴了 r6 崩溃的 postmortem 和陷阱清单。`preflight` + dashboard + agent plugin 构成第三个支柱"可观测训练"（observable training）——架构图上没有它的方块，因为没有数据流经它。

## 11. 实验结果

`Qwen3.5-35B-A3B` 训练三 epoch（126 步），2,699 任务 OpenSWE 派生 index，三个原生 harness，评估在 held-out SWE-bench Verified（temperature 0.7，200 turns，200k 上下文预算）：

| Coding agent | 模型 | SWE-bench Verified (%) |
| --- | --- | --- |
| OpenHands SDK | Qwen3.5-35B-A3B | 64.0 |
|  | Qwen3.6-35B-A3B | 67.4 |
|  | KAT-Coder-V2.5-Dev | 67.0 |
|  | **Lego-RL-Qwen3.5-35B-A3B** | **70.4 (+6.4)** |
| Claude Code | Qwen3.5-35B-A3B | 62.4 |
|  | Qwen3.6-35B-A3B | 63.4 |
|  | KAT-Coder-V2.5-Dev | 66.8 |
|  | **Lego-RL-Qwen3.5-35B-A3B** | **68.2 (+5.8)** |
| OpenCode | Qwen3.5-35B-A3B | 57.2 |
|  | Qwen3.6-35B-A3B | 60.6 |
|  | KAT-Coder-V2.5-Dev | 61.2 |
|  | **Lego-RL-Qwen3.5-35B-A3B** | **66.6 (+9.4)** |

README 的解读（照录，未独立验证）：RL 后的 3.5 代策略在三个 harness 下都超过下一代 base（3.6）和从它 post-train 出来的 KAT-Coder-V2.5-Dev。gain 是 harness-specific 的：KAT-Coder 在自己作者报告的 harness（Claude Code）下 +3.4，在 OpenCode 下 +0.6，在 OpenHands SDK 下 −0.4。这正是 Lego-RL 在部署 harness 里训练的论据。

公开 Dashboard（全部训练 Qwen3.5-35B-A3B）：

| Run | Harness | 算法 | Steps | Val（start → best） |
| --- | --- | --- | --- | --- |
| OpenHands SDK | OpenHands SDK | GSPO | 126 | 64.0 → 70.4 |
| OpenCode | OpenCode | GSPO | 127 | 57.2 → 66.6 |
| Claude Code | Claude Code | GSPO | 130 | 62.4 → 68.2 |
| Codex | Codex | GSPO | 125 | 60.0 → 67.6 |
| Mixed harness | 三 harness 混合 | GSPO with R3 router replay | 126 | 62.2 → 68.2 |
| SAO | OpenSWE | SAO（critic-based） | 180 | 0.640 → 0.682 |
| SAO 多语言 | 多语言任务集 | SAO | — | 在 SWE-bench Multilingual 上验证 |

每个 board 都展示完整的 reward / 验证曲线和 per-trial 分析。Codex harness 是 2026-10-08 的新闻（专用 dashboard 需要 Workspace 登录），Mixed harness 2026-09-11，SAO 2026-09-07，首次公开发布 2026-08。

## 12. 代码地图

```
src/
├── verl_patch/                     # 对 verl 的定制（verl 本体在 third_party/，patch 见 patches/）
│   ├── agent_loop/
│   │   ├── vllm_chat_completion_proxy.py   # 2339 行：in-process proxy，统一 API + 轨迹捕获
│   │   ├── trajectory.py                   # TrajectorySegment / TrajectoryRecorder
│   │   ├── harness.py                      # harness 配置归一化 + HarnessResolver（无 Harbor/verl import）
│   │   ├── builtin_swe_agent_loop.py       # 1085 行：OpenHands SDK / OpenHands / OpenCode / Claude Code 的 agent loop
│   │   ├── builtin_cc_agent_loop.py        # 兼容子类，无行为覆盖
│   │   ├── responses_transform.py          # 577 行：响应变换
│   │   └── progress_tracker.py             # 进度追踪
│   └── models/
│       └── qwen3_5_moe_token_classification.py  # VeOmni critic 的 Qwen3.5-MoE value head sidecar
└── harbor_patch/                   # 对 Harbor 的定制
    ├── agents/                     # scaffold 适配：claude_code / opencode / openhands_sdk / codex / terminus_2 / openhands...
    └── environments/               # remote_docker、kubernetes

patches/
├── verl_7aed6b23.patch             # 123 KB：verl 主 patch，钉在 verl commit 7aed6b23
├── verl_sao_7aed6b23.patch         # 143 KB：SAO 所需的 verl 改动
├── verl_segments_7aed6b23.patch    # 30 KB：segment 相关
└── vllm_2a69949b.patch             # 9 KB：vLLM patch，钉在 commit 2a69949b

scripts/
├── train/                          # train.sh + lib + configs（5 个验证过的生产配置）+ templates
├── eval/ / infer/                  # 评估 / batch inference（同五步合同）
├── agent_runtimes/                 # agent 运行时（如 codex）
├── lib/                            # common_env.sh、preflight.sh、ray_bringup.sh、vllm_serve.sh、live_probe.sh…
└── setup_env.sh                    # venv + harbor/verl checkout + 自检

utils/                              # convert_swerebench_filtered.py、create_task_index.py、
                                    # merge_critic_ckpt.sh、apply_veomni_valuehead_patch.py、
                                    # apply_r3_vllm_patches.py、eval_swerebench_filtered.py…
tests/
├── agent_loop/                     # proxy、trajectory segments、harness、episode training、
                                    # anthropic codec、codex runtime/proxy、responses transform、rollout failure 回归
├── harbor/                         # k8s pod 创建调度、文件传输
└── reliability/                    # preflight、dashboard

webui/                              # Live dashboard（函数 + 前端）
docs/content/docs/                  # 文档站源码（architecture / run-training / troubleshooting / reference）
.agents/skills/、.claude/plugins/   # agent plugin（/rl:* skills）
```

依赖关系的关键点：Lego-RL 本身不 fork verl/harbor，而是 **clone 上游 + 打 patch**。`setup_env.sh` clone `harbor` 和 `verl` 到 `third_party/` 并 editable 安装，patch 通过 `patches/*.patch` 应用；`veomni` 是 wheel 无法打 git patch，改用 `utils/apply_veomni_valuehead_patch.py` 直接修 site-packages。版本钉死：verl commit `7aed6b23`，vllm commit `2a69949b`。

## 13. 与我工作的联系

- **verl 生态**：labs-molt 的 length-penalty 工作也是 verl 生态里的 reward shaping；Lego-RL 展示了 verl 在 agentic RL 里的完整用法（agent loop、rollout 捕获、async、partial rollout、R3）。它的 patch 组织方式（主 patch + SAO patch + segment patch，各自钉死上游 commit）是"在不 fork 的前提下深度定制大框架"的一个可参照的范式。
- **Reward shaping vs rollout fidelity**：length-penalty 是在 reward 上做 shaping；Lego-RL 的教训是 shaping 之前先保证 rollout 本身是 faithful 的——mask 错一位、logprob 对不上，shaping 都是在错误信号上做功。两者的检查优先级：先 fidelity（相关性 ≥0.99），再 shaping。
- **Cross-repo port 的对照**：之前从 labs-molt port 到 verl-recipe 时提炼的 checklist（聚合粒度、hook 位置、默认值行为一致、不改动上游原始值）在这里同样适用。Lego-RL 的 `harness.py` 刻意零依赖（无 Harbor/verl import）就是一个"把配置归一化从 trial 创建中解耦"的 hook 位置设计。
- **Postmortem 写作**：r6→r7 的写法值得学——因果链 → 每个环节一个旋钮 → 改动表 → 行为指标验证（密度而非长度）。WWW 论文的 failure analysis 可以借这个结构。

## 14. 值得继续挖的方向

（只列事实，不做"值得提"的结论。）

- 上游唯一 open PR #25：`feat(eval): support native DeepSWE tasks with isolated verification`，eval 侧的工作。
- Codex harness 是 2026-10-08 刚落地的，配套测试 `tests/agent_loop/test_codex_runtime.py`、`test_codex_proxy.py`，以及 Codex 专用的 training dashboard。
- SAO 的 critic 初始化（cold vs warm）和 critic checkpoint 合并（`merge_critic_ckpt.sh`，流式 merger 拒绝 Qwen3.5-MoE）是文档里明确标注的脆弱点。
- Mixed-harness 训练用 "GSPO with R3 router replay"，R3 在多 harness 混合下的路由一致性是这个 repo 特有的机制。
- 文档站的 troubleshooting 专题（`val-false-zero`、`async-val-deadlock`、`offline-eval-garbage`）是实测故障的完整记录，继续读可以直接积累运维判断。

## 15. 参考资料

- Paper：arXiv:2608.17393《LEGO-RL: Harness-Native Reinforcement Learning for Coding Agents》
- SAO：arXiv:2607.07508《Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning》
- GSPO：arXiv:2507.18071（Qwen Team）
- GRPO：arXiv:2402.03300（DeepSeekMath，Shao et al.）
- PPO：arXiv:1707.06347（Schulman et al.）
- HybridFlow / verl：arXiv:2409.19256（Sheng et al.，EuroSys 2025）
- vLLM：arXiv:2309.06180（Kwon et al.，SOSP 2023）
- SWE-bench：arXiv:2310.06770（Jimenez et al.，ICLR 2024）
- MoE 路由对齐：arXiv:2510.11370（Ma et al. 2025）
- Repo 文档：https://lego-rl.pages.dev/docs（本学习主要依据 `docs/content/docs/`，辅以关键源码）
- 训练数据：https://huggingface.co/datasets/Lego-X/Lego-RL-2699
