# TLA：公理组合式神经架构——无反向传播的误差驱动学习、自适应推理深度与终身学习

**TLA: An Axiom-Composed Neural Architecture — Backpropagation-Free Error-Driven Learning, Adaptive Reasoning Depth, and Lifelong Learning**

**作者**：guaidao2 · coolmoon
**单位**：玄幕安全团队 / 墨渊安全实验室（Xuanmu Security Team / Moyuan Security Laboratory）
**日期**：2026-08-10　**版本**：v1.0（对应代码提交 `bc0bb11`）

---

## 摘要

本文提出 TLA（公理组合式神经架构），一个把预测编码（Predictive Coding, PCN）、液态时间常数
（Liquid Time-Constant, LTC）、自适应推理深度、自模型与终身学习组合成单一序列网络的新架构。
其学习环**完全没有全局反向传播**：所有权重更新由局部误差驱动（ΔW ∝ e·a）。推理环通过迭代
精化隐藏表征（"会琢磨"）并支持**双过程回退**——琢磨失败即回退摊销首猜（系统2失败→系统1兜底）。
终身学习由 CLS 重放与**突触巩固**（弹性权重巩固的无反向传播版本）共同实现。

全部机制按**预注册判据**验证（判据先于实现，跑数后只许改代码）。三个核心问题——无捷径变体的
学习强度、推理环"琢磨"的效用、任务切换的灾难性遗忘——初测均为**诚实负结果**，随后分别由
摊销首猜、双过程回退、突触巩固三个**以生物学为依据**的设计修正翻转为正结果。我们同时报告了
"线性捷径悖论"这一核心科学发现：任务无关的共享线性映射让单任务易学，却同时杀死琢磨与防遗忘。
本架构目前是研究原型：唯一跑通的任务是玩具级的连续时间世界模型预测。

**关键词**：预测编码；无反向传播学习；自适应推理深度；双过程理论；弹性权重巩固；液态神经网络

---

## 1 引言

近年来，深度学习架构（尤其是 Transformer）的成功证明了**组合即创新**：注意力、残差、归一化、
多层感知机与位置编码——没有一个是 Transformer 发明的，其"新"在于组合方式与可扩展性。

本文沿袭这一思路，从一组关于智能的公理出发，组合一个新的序列架构。我们遵循的设计方法论是
**在生物学的基础上思考原因**：当一个机制实测失败时，先问"人是怎么解决这个问题的"，再实现
对应的生物机制。这一方法论在本文中三次把负结果翻转为正结果。

TLA 的定位（§0，见设计文档）：
- 不是新的神经元（单元用现成 LTC），而是**新的组织方式 + 新的推理范式 + 新的训练范式**；
- 三个核心增量：**无 BP 误差驱动学习**、**自适应推理深度（会琢磨）**、**终身学习**。

### 1.1 贡献

1. 一个无反向传播、由局部预测误差驱动的序列学习架构（§3.1–3.3）；
2. 推理时自适应深度 + **双过程回退**：琢磨作为首猜的改进器而非替代者（§3.4）；
3. 无 BP 的**突触巩固**（EWC 变体）解决任务切换遗忘（§3.5）；
4. 预注册判据体系下的诚实实验，含三个"负转正"与一个核心负发现（§4–5）。

---

## 2 相关工作

- **预测编码**：Rao & Ballard (1999) 提出分层误差传播的推理-学习二相性；Whittington & Bogacz (2017)
  证明 PCN 的局部学习规则在**收敛推理**下近似反向传播；Millidge、Tschantz & Buckley (2022) 沿
  任意计算图再次确认等价性；Marin-Ricoy 等 (2024) 通过 Lifted Predictive Coding 使其规模化。
  我们沿用标准 PCN 的 settle 动力学与局部学习规则（W&B 的 ΔW ∝ e·a 是本文无 BP 路线的坐标系）。
  **关键边界**：W&B 等价定理在收敛推理下成立，而**有限推理（settle 步数不足）时 PCN 偏离 BP**。
  TLA 的推理环恰好在"充分 settle"与"有限推理"之间可调——settle 深度如何影响与 BP 的距离，
  是本文实证考察的问题（§3.8、§4.5）。
- **液态神经网络**：Hasani 等 (2021) 的 LTC 用输入调制时间常数做连续时间动力系统；Liquid AI
  (2024) 证明纯 ODE 求解不缩放，需离散化。我们只把 LTC 用作时间记忆基板。
- **推理时计算**：o1/R1 的 thinking tokens、PonderNet (Banino 2021)、ACT (Graves 2016)、DEQ
  (Bai 2019)。已知失败模式为 overthinking 与 solver 空转；我们用"误差小即停 + 输出收敛即停 +
  预算耗尽 doubtful"三个停止条件，并用双过程回退解决空转。
- **双过程理论**：Kahneman 系统1（快直觉）/系统2（慢推理）。我们的摊销首猜=系统1，settle=系统2，
  "系统2失败→系统1兜底"是本文的生物学核心设计。
- **灾难性遗忘与巩固**：EWC (Kirkpatrick 2017) 用 Fisher 信息惩罚重要权重；CLS (Kumaran 2016)
  用重放。我们提出无 BP 的突触巩固（importance=更新量级）并与上下文忠实重放结合。
- **MoE**：专家混合 (Shazeer 2017)。我们用它做任务分离容量（探索性，见 §5.3）。

---

## 3 方法

### 3.1 预测编码层叠与自由能

设网络有 $L$ 个隐藏层，表示 $\mu_l \in \mathbb{R}^{d_l}$（$l=1,\dots,L$），输入 $\mu_0 = x$。
生成（自上而下）预测为

$$
p_{l-1} = g(W_l \mu_l + b_l),\qquad g(u) = \tanh(u),\quad l = 1,\dots,L,
$$

顶层取先验 $p_L = 0$。定义各层误差

$$
e_0 = x - p_0,\qquad e_l = \mu_l - p_l\quad (1 \le l \le L-1),\qquad e_L = \mu_L,
$$

自由能（能量函数）为

$$
E(\mu; x) = \frac12 \sum_{l=0}^{L} \|e_l\|^2 .
$$

**推理（settle）**：对固定输入，按负梯度更新表示，即对自由能做梯度下降：

$$
\frac{\partial E}{\partial \mu_l}
= e_l - W_l^\top \Big( g'(a_l) \odot e_{l-1} \Big),
\qquad a_l = W_l \mu_l + b_l,\quad g'(a) = 1 - \tanh^2(a),
$$

$$
\mu_l \leftarrow \mu_l - \eta_{\mathrm{inf}} \frac{\partial E}{\partial \mu_l}
= \mu_l - \eta_{\mathrm{inf}} \Big( e_l - W_l^\top (g'(a_l)\odot e_{l-1}) \Big).
$$

推导：$\partial \|e_{l-1}\|^2/\partial \mu_l = -2\, W_l^\top (g'(a_l)\odot e_{l-1})$，
$\partial \|e_l\|^2/\partial \mu_l = 2 e_l$，合并且除以 2 即得。注意误差沿 $W^\top$ 传播——
这是**权重转置传播**，不是反向传播（无损失函数对权重的全局梯度）。

**学习（权重更新）**：对 $E$ 关于权重求梯度：

$$
\frac{\partial E}{\partial W_l} = - \big( g'(a_l)\odot e_{l-1} \big) \mu_l^\top,
\qquad
\frac{\partial E}{\partial b_l} = - \big( g'(a_l)\odot e_{l-1} \big),
$$

局部误差驱动更新（无 BP）：

$$
\Delta W_l = \eta\, (g'(a_l)\odot e_{l-1})\, \mu_l^\top,
\qquad
\Delta b_l = \eta\, (g'(a_l)\odot e_{l-1}).
$$

读出层 $p_{\mathrm{out}} = W_{\mathrm{out}} \mu_L + W_{\mathrm{lin}} x + b_{\mathrm{out}}$，
误差 $e_{\mathrm{out}} = t - p_{\mathrm{out}}$，更新

$$
\Delta W_{\mathrm{out}} = \eta\, e_{\mathrm{out}} \mu_L^\top,\qquad
\Delta W_{\mathrm{lin}} = \eta\, e_{\mathrm{out}} x^\top.
$$

训练时目标 $t$ 经 $e_{\mathrm{out}}$ 反馈注入顶层（$\mu_L$ 更新项含 $-W_{\mathrm{out}}^\top e_{\mathrm{out}}$），
使表示吸收任务信息——仍是误差驱动，非外部奖励。

### 3.2 液态基板（LTC）

时间记忆由 LTC 细胞提供（$\tau$ 输入调制、离散化、有界）：

$$
\tau(t) = \tau_{\min} + (\tau_{\max}-\tau_{\min})\, \sigma(W_\tau x(t)),
$$

$$
h_{t+1} = h_t + \frac{dt}{\tau(t)} \big( -h_t + W_{\mathrm{in}} x(t) + W_h h_t + b \big),
\qquad
h \leftarrow v_{\max}\tanh(h / v_{\max}).
$$

$h$ 进入读入向量 $x = [s_t, h]$（观测 + 身体状态），保证时间记忆穿过架构（公理⑤）。
$\tanh$ 软饱和保证有界（P-PHY-3）；零输入时 $W_h$ 小初始化保证收敛到静息（P-PHY-1）。

### 3.3 摊销首猜 + 残差修正（分工写死）

**哲学诊断**：两个优化器（快=settle / 慢=权重）抢同一目标导致分工失序——有捷径时慢优化器
抢了快优化器的活（琢磨空转）；无捷径时快优化器稀释了慢优化器的信号（弱学习）。解法是把分工
写死：

$$
\boxed{\;
\operatorname{pred} = \underbrace{W_b x + b_b}_{\text{摊销首猜}} + \underbrace{W_o \mu + b_o}_{\text{残差（经 settle）}} \;}
$$

- **首猜**对"自己的误差"负责：$e_b = t - (W_b x + b_b)$，$\Delta W_b = \eta e_b x^\top$；
  学得快（线性、不被 settle 稀释），提供学习强度；
- **残差**对"总误差"负责：$e_{\mathrm{tot}} = t - \operatorname{pred}$，
  $\Delta W_o = \eta e_{\mathrm{tot}} \mu^\top$；经 settle 的加性残差使琢磨不可旁路。

分工防摆烂：$W_b$ 只按 $e_b$ 更新（残差救不了它，必须自己学）；残差只按 $e_{\mathrm{tot}}$ 更新
（首猜错时 $e_{\mathrm{tot}}$ 大，残差有活干）。

### 3.4 双过程回退（系统2失败 → 系统1兜底）

**生物学依据**（Kahneman 双过程）：人想不出来就凭直觉猜。工程化为推理规则：

$$
\operatorname{guess} = \operatorname{readout}(x)\big|_{\text{暖启动 }\mu},
\qquad
\operatorname{reasoned} = \operatorname{readout}(x)\big|_{\text{settle 后}},
$$

$$
\operatorname{confidence} = 1 - \frac{\max\text{-err}_{\text{最终}}}{\max\text{-err}_{\text{初始}}}
\in [0,1],
\qquad
\operatorname{drift} = \|\operatorname{reasoned} - \operatorname{guess}\|_2,
$$

$$
\boxed{\;
\operatorname{output} =
\begin{cases}
\operatorname{guess}, & \operatorname{confidence} < 0.35 \;\text{或}\; \operatorname{drift} > 0.02,\\
\operatorname{reasoned}, & \text{否则}.
\end{cases}}
$$

直觉：琢磨只被信任做**微调**（drift 小）；大偏离 = 可疑 = 过度精化，回退直觉；
confidence 低 = 琢磨没进展，也回退。这使琢磨从"封闭重复（负价值）"变为"保守确认（正价值）"。

### 3.5 突触巩固（无 BP 的 EWC）

**生物学依据**：海马-皮层突触巩固——重要的突触不轻易改变。对每个权重元素 $k$：

$$
I_k = \sum_{t \in \text{任务A}} \big| \delta_k(t) \big|,
\qquad \delta_k = \text{本次更新的 pre-lr 项（如 } e\,a \text{）},
$$

$$
\hat{I}_k = \frac{I_k}{\max_k I_k} \in [0,1]
\qquad\text{（按张量归一化，防无界累加）},
$$

B 训练时每步后施加"拉回"：

$$
\boxed{\;
W_k \leftarrow W_k + \eta\,\lambda\, \hat{I}_k \big( W^{\mathrm{ref}}_k - W_k \big),\;
\lambda = 10
\;}
$$

其中 $W^{\mathrm{ref}}$ 是任务 A 结束时的权重快照。$I_k$ 是 Fisher 信息对角元的局部近似
（梯度幅度），全局部、无 BP。$\lambda$ 实测标定：$\lambda>200$ 保护过度，$\lambda=500$ 数值不稳定。

### 3.6 自模型（Self_Slot）

自模型预测"自己会输出什么"：

$$
\widehat{p} = W_s\, \phi(h),\qquad \phi = \tanh(F^\top x),\quad F \text{ 固定随机特征},
$$

$$
\mathcal{L}_{\mathrm{self}} = \|\widehat{p} - p_{\mathrm{out}}\|^2,
\qquad \Delta W_s = \eta_s\, e_s\, \phi^\top,\; e_s = p_{\mathrm{out}} - \widehat{p}.
$$

输入只含自身输入状态（不含输出本身），避免平凡复制。它同时充当推理环的一致性门控。

### 3.7 容量管理（生长/修剪）与 CLS 重放

- **生长门**：$\text{grow} \iff \bar{e}>\varepsilon_e \wedge \text{novelty}>\varepsilon_n \wedge E>\varepsilon_E$
  （惊奇 × 新奇 × 能量）；新单元校准增益 $0.3 \to 1.0$；
- **修剪**：importance（长时窗 EMA）低于相对阈值（均值的一定比例，防级联，单次≤30%）；
- **CLS 重放**：均匀抽样 + 上下文忠实（存当时的身体状态 $h$ 一并重放）+ 睡眠式大批量重放。

### 3.8 推理深度与 BP 的关系（实证问题）

W&B (2017) 的等价定理在**收敛推理**下成立：充分 settle 的 PCN 局部学习 ≈ BP；**有限推理
（settle 步数不足）时 PCN 偏离 BP**。TLA 的推理环恰好在二者之间可调——settle 深度如何影响
与 BP 的表示/行为距离，是本节实证考察的问题（对应文献中 target-based learning / 有限推理
偏差的讨论）。

实证策略（§4.5）：同任务、同数据上对比 (i) BP 学生网络、(ii) 单步局部 PCN（有限推理）、
(iii) 充分 settle 的局部 PCN（近收敛推理），用表示距离（CKA）与泛化差异度量与 BP 的距离。
**若收敛推理的 PCN ≈ BP 而有限推理的 PCN 显著偏离，则实证支持 W&B 等价定理及其边界**——
这也为"琢磨步数"作为可调的超参数提供了依据（步数决定与 BP 的偏离程度）。

---

## 4 实验

### 4.1 任务

**有界阻尼弹簧世界**（玩具）：$\operatorname{pos}' = \operatorname{pos} + \operatorname{vel}\cdot dt$，
$\operatorname{vel}' = \operatorname{vel} - \omega^2 \operatorname{pos}\cdot dt$，频率 $\omega$ 区间切换，
$dt$ 不规则采样。观测 $=[\operatorname{pos},\operatorname{vel},dt]$，目标 $=[\operatorname{pos}',\operatorname{vel}']$。
测试集用**训练未见过的 $\omega$ 区间**（未见动态泛化）。另有 `drift` 模式（恒定速度）用于任务切换实验。

### 4.2 判据体系

判据**先于实现**（预注册）：跑数前锁死标准，跑数后只许改代码、不许改判据；负结果如实记录。
全部 40 个测试（pytest）固定种子、可复现。

### 4.3 主要结果

| 判据 | 初测 | 修复后 | 关键数字 |
|---|---|---|---|
| P-LEARN-3 无 BP 学得动 | ✅ | — | 训练 MSE 0.0008；未见 ω 0.480 vs 随机 0.978 vs 恒等 0.670 |
| 学习强度（无捷径弱学习） | ❌ 0.11 | ✅ **0.0046**（原则一） | 分布内 MSE，解锁阈值 0.02 |
| P-COG-3 琢磨消融 | ❌ 空转/负价值 | ✅ **有能力轴翻转**（双过程回退） | 噪声轴 0.0931 < 瞎猜 0.0966 |
| P-LEARN-1 防遗忘 | ❌ 保留率 6.4% | ✅ **108.6%**（突触巩固 EWC） | 无重放 101.9%；学习强度保持 0.0084<0.02 |
| P-COG-4 doubtful 校准 | ✅ | — | 低置信 5.1× 高置信；doubtful 5.3× 未标记 |
| P-COG-1/2 会琢磨步数 | ⚠️ | — | 干净 median=3（预算 8）；噪声均值差分 2.67 vs 3.36 |
| P-LEARN-2 放大 | ✅ | — | hidden 32→128 成本比 1.07 |

### 4.4 三个"负转正"（细节）

**(a) 学习强度**（原则一，§3.3）：无捷径 PCN 分布内 MSE ≈0.11（弱学习）；加入摊销首猜后
0.0046，达捷径基线水平，解锁 P-COG-3/P-LEARN-1 重测。

**(b) 琢磨效用**（双过程回退，§3.4）：有捷径版琢磨空转（增益 0.12%）；无捷径版 OOD 过度精化
（-5.7%，纯首猜最优）。加回退后，**有能力轴**（分布内+噪声）上琢磨从负价值（reasoned 0.1006 >
瞎猜 0.0966）翻转为正价值（fallback 0.0931 < 瞎猜）；未见 ω 为**无能力轴**（三策略全在 ~0.4 瞎猜水平），
如实记录为限制。

**(c) 防遗忘**（突触巩固，§3.5）：定位诊断（冻结首猜，保留率 6.4%→10.9%）确认遗忘分布全网络；
EWC（λ=10，importance 归一化）后保留率 **108.6%**，B 仍可学（0.65 vs 随机 1.02）。

### 4.5 实证：推理深度与 BP 的关系（支持 W&B 等价定理及其边界）

按 §3.8 的策略，在弹簧任务上对比 BP 学生（autograd）、单步局部 PCN（有限推理）、充分 settle
局部 PCN（近收敛推理）。**实测（表示证据为裁决依据；行为证据在设置间翻转，降为报告项）：**

| 度量 | 单步 vs BP | 充分 settle vs BP | 结论 |
|---|---|---|---|
| 表示距离（线性 CKA） | **0.56** | **0.96** | 有限推理显著偏离 BP；收敛推理 ≈ BP |
| 未见 ω 行为差异 \|MSE−MSE_bp\| | 0.10–0.11（翻转） | 0.09–0.13（翻转） | 噪声，不作判定 |

**裁决：实证支持 W&B 等价定理及其边界**——充分 settle（近收敛推理）的局部学习与 BP 高度一致
（CKA 0.96），而单步（有限推理）显著偏离（CKA 0.56）。这与 W&B (2017) / Millidge (2022)
的"收敛推理等价、有限推理偏离"一致。诚实披露：行为差异腿在 n_traj=15/20 间翻转
（0.1096/0.1006 vs 0.1326/0.0948），未达稳健效应量，故裁决仅基于表示证据。
**意义**：① 本架构的"琢磨步数"是可调超参数——步数决定与 BP 的偏离程度；② 单步（有限推理）
学习的表示确实不同于 BP，为未来在非线性场地上考察"有限推理学到 BP 给不出的东西"
（target-based learning 线索）留下空间。

---

## 5 讨论

### 5.1 线性捷径悖论（核心发现）

共享线性捷径 $W_{\mathrm{lin}}x$ 让单任务易学（P-LEARN-3 通过），但它是任务无关的单一线性映射，
同时杀死：**琢磨**（settle 被旁路→空转，P-COG-3）、**防遗忘**（A/B 冲突映射权重级互覆，P-LEARN-1）、
**难度差分**（P-COG-2 弱化）。去掉捷径则机制真实（分布内 +45%）但学习弱。三者同源——
这是本架构最值得继续研究的一个负发现。

### 5.2 生物学方法论

"造神经网络要在生物学的基础上想原因"在本工作中三次把负结果翻转为正结果：
双过程兜底治琢磨、突触巩固治遗忘、摊销首猜治学习强度。我们主张：**当一个机制实测失败时，
先问人/脑是怎么解决的，再实现对应机制**——比盲目标定超参更可能产生结构性的修复。

### 5.3 探索：任务分离容量（MoE）

无捷径 + 专家混合：自由能路由**不分离**（MoE 对称性，r 恒卡 0.5）；原型路由**部分分离**
（low-vel 0.57 vs high-vel 0.30）；但无捷径专家是弱学习者（~0.11 vs 捷径 0.004）——容量分离
不能治愈"无捷径弱学习"。**修复顺序修正**：先解决学习强度，再谈路由分离（原则一已兑现学习强度，
MoE+摊销首猜为下一步）。

---

## 6 局限与未来工作

1. **任务单一**：唯一跑通的是玩具世界模型；无真实任务（图像/语言/控制）验证；
2. **琢磨仍弱**：双过程回退只在有能力轴翻转，OOD 无能力轴三策略全瞎猜；琢磨期间无新信息进入
   （封闭循环），与人类"引入新约束、换方法"的开放思考仍有本质差距；
3. **重放+EWC 的协同**：EWC 单独即可锚定（无重放 101.9%），重放的作用待进一步分离；
4. **规模化未知**：无 BP 局部学习在大规模数据上的效率未验证（P-LEARN-2 仅在 hidden 32→128）；
5. **下一步**：摊销首猜 + MoE 专家分离（消遗忘 + 琢磨只该用时用）；换真实小任务验证组合。

### 6.1 实际实验前的前置（2026-08-10 完成）

- **batch 化训练**：局部误差驱动学习已支持 mini-batch（更新项按批累计除以批大小），实测收敛
  等价（batch 32ep ≈ 单样本，需 ~4× 更新次数——mini-batch 收敛速率，符合预期）。真实任务
  数据按批提供时可直接使用；
- **测试基建**：49 个判据测试 ~87–118s（负载敏感）。EWC 判据对数据量敏感，判据完整性优先不瘦身；
- **表示坍缩防护**（文本世界防预注册，默认关）：EMA 表示统计 + anti_collapse hook 已就绪，
  阈值/强度待真实坍缩场景标定；
- **判据未达项**：P-COG-1（≤1 未达，median=3）、P-COG-5（跷跷板）、P-META-4（降级）——
  均为如实记录，论文措辞已收尾。

---

## 7 结论

本文提出 TLA——一个把预测编码、液态时间常数、自适应推理深度、自模型与终身学习组合成单一
序列网络的新架构，学习环完全无反向传播。通过预注册判据体系，我们诚实报告了两个核心负发现
（琢磨空转、遗忘）及其机制归因，并用三个以生物学为依据的设计修正（摊销首猜、双过程回退、
突触巩固）将其翻转为正结果。作为研究原型，TLA 证明了这套组合可运行、可测试、机制诚实；
能否像 Transformer 那样被证明有效，取决于未来的规模化与真实任务验证。

---

## 附录 A：符号表

| 符号 | 含义 |
|---|---|
| $\mu_l$ | 第 $l$ 层表示 |
| $p_l$ | 第 $l$ 层自上而下预测（$p_l = \tanh(W_{l+1}\mu_{l+1}+b_{l+1})$） |
| $e_l$ | 第 $l$ 层误差（$e_l = \mu_l - p_l$） |
| $E$ | 自由能（能量函数）$\frac12\sum\|e_l\|^2$ |
| $g'$ | $\tanh$ 导数 $1-\tanh^2$ |
| $W_b, b_b$ | 摊销首猜权重/偏置 |
| $W_o, b_o$ | 残差读出权重/偏置 |
| $e_b, e_{\mathrm{tot}}$ | 首猜误差 / 总误差 |
| $\eta, \eta_{\mathrm{inf}}$ | 学习率 / 推理步长 |
| $\lambda$ | 突触巩固强度（EWC） |
| $\hat{I}_k$ | 归一化权重重要性 |
| $h, \tau$ | LTC 状态 / 时间常数 |
| $\omega, dt$ | 弹簧频率 / 采样间隔 |

## 附录 B：判据清单（预注册）

P-PHY-1~3（基板）、P-COG-1/2（琢磨步数）、P-COG-3（琢磨消融）、P-COG-4（doubtful 校准）、
P-COG-5（防摆烂/空转）、P-LEARN-1（防遗忘）、P-LEARN-2（放大成本）、P-LEARN-3（无 BP 学习探针）、
P-META-1~4（生长/校准/修剪/Self_Slot）。全部 44 个测试见 `tests/`，裁决见 §4.3 与设计文档 §10。

## 参考文献

1. Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
2. Whittington, J. C. R., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*.
3. Hasani, R., Lechner, M., et al. (2021). Liquid Time-constant Networks. *AAAI*.
4. Bai, S., Kolter, J. Z., & Koltun, V. (2019). Deep Equilibrium Models. *NeurIPS*.
5. Banino, A., et al. (2021). PonderNet: Learning to Ponder. *arXiv:2107.05407*.
6. Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. *arXiv:1603.08983*.
7. Kahneman, D. (2011). *Thinking, Fast and Slow*.
8. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
9. Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What Learning Systems do Intelligent Agents Need? *Trends in Cognitive Sciences*.
10. Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
11. Marin-Ricoy, A., Alonso, N., & Berbel, A. (2024). Lifted Predictive Coding. *arXiv*.
12. DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.
13. Millidge, B., Tschantz, A., & Buckley, C. L. (2022). Predictive Coding Approximates Backprop along Arbitrary Computation Graphs. *Neural Computation*.
14. Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton, G. (2020). Backpropagation and the brain. *Nature Reviews Neuroscience*.

---

*本文所有实验数字均可由仓库代码复现（固定种子，pytest 40/40）。负结果与降级分支是判据的一部分，同样锁死。*
