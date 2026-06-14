# Delta V2 全面重构方案

> 2026-06-13 v2 | 基于数据审计的修正方案

---

## 🚨 数据审计发现（改变一切）

在对现有数据做完整交叉验证后，发现**论文的核心统计结果比声称的弱得多**：

### 审计结果汇总

| 信号 | 无控制 t | +rel_size | +全控制 t | 论文声称 |
|---|---|---|---|---|
| H_smooth_resid | -2.85*** | -2.73*** | **-1.66** | -3.02*** |
| H_smooth | -1.11 | -1.11 | -0.72 | — |
| H×D (h_d_interaction) | — | — | **+2.25**** | +3.38*** |
| H×D (proper: H+D+H×D) | — | — | **+0.69** | +3.38*** |

### 致命发现

1. **H_smooth_resid 不生存标准控制变量**：加 mom_6m_past 后 t 从 -2.85 降到 -1.66（p=0.10）。论文只报告无控制结果。

2. **H×D 交互效应是统计幻觉**：论文报告 t=3.38 的 H×D，是单独回归 H×D→Y（不含 H 和 D 主效应）。当放入正确的交互模型 Y~H+D+H×D+controls 时，H×D 的 t=+0.69（p=0.49）。这在任何标准下都不显著。

3. **Familiarity bias 有实锤证据**：
   - 知名股（AAPL/MSFT/GOOGL等20只）H_smooth 均值 1.3457 vs 非知名股 1.4224，t=-4.73***
   - 最低熵 Top 10：NVDA, AVGO, META, MSFT, AAPL, AMZN, GOOGL — 全是超级大盘+高训练数据覆盖
   - H_smooth 与 rel_size 相关系数 r=-0.052（p<0.001）

4. **Size 分组中效应全部不显著**：
   - Large: t=-2.72** (但被 familiarity 解释)
   - Medium: t=-2.03*
   - Small: t=-0.41
   - Size quintile 内 double sort：全部不显著

5. **Horizon strengthening 存活**（唯一的好消息）：
   - 1Q: t_NW=-3.02, Cum2Q: t_Hodrick=-4.00, Cum3Q: t_Hodrick=-4.96
   - FM beta 自相关接近零（lag-1=-0.022），所以重叠问题不严重
   - 但注意：horizon 效应主要是累积效应的数学结果，不是新的经济发现

---

## 重新定位：论文还剩什么？

### 不能再声称的

- ❌ "Entropy Premium" 作为独立的截面定价因子（不生存标准控制）
- ❌ "H×D 交互效应"（proper specification 不显著）
- ❌ "Miller (1977) overvaluation correction"（数据不支持负向预测）
- ❌ "分歧度量提供增量信息"（可能只是 momentum 的非线性变换）

### 仍然可以声称的

- ✅ LLM 评分的熵结构与过去回报强相关（t=-8.75）——这本身是一个描述性发现
- ✅ 跨模型分歧是案例难度的信号（这不需要金融回报预测来证明）
- ✅ Horizon 效应在 Hodrick SE 下仍然成立
- ✅ 大盘股效应存在（但需要排除 familiarity 替代解释）
- ✅ 三模型设计比三 prompt 设计更有理论基础

### 根本问题

**Delta 论文试图证明"LLM 分歧度量是新的定价因子"，但数据更支持"LLM 分歧度量是现有因子（尤其是 momentum 和 size）的非线性代理变量"。**

这意味着论文的核心贡献需要从"发现新因子"转向更诚实的定位。

---

## 三条可选路线

### 路线 A：修复论文——降低主张，诚实报告

**定位**：LLM agent 熵与回报的描述性关联，附带 cautionary evidence

**核心修改**：
1. 主表报告完整控制变量结果（t=-1.66）
2. 去掉 H×D 交互效应声称（或放入 Appendix 标注 "not robust"）
3. 匿名化实验 + familiarity 控制（如果通过，可以声称"排除 familiarity 后效应边际显著"）
4. 理论框架降级为"exploratory"而非"confirmatory"
5. 标题改为 "LLM Agent Entropy and Stock Returns: Measurement, Confounds, and Caveats"

**目标期刊**：降档到 JFQA / RFS / Management Science（不再是 JFE）
**优点**：诚实，不会被后续 replication 打脸
**缺点**：贡献大幅缩水，不再是"发现"而是"评估"

### 路线 B：转向——从"定价因子"到"LLM-as-Judge"

**定位**：用金融数据作为 testbed 证明"多模型分歧度量是有信息量的"

**核心重构**：
1. 不再声称"entropy premium"
2. 核心发现变成：当多个 LLM 对同一股票分歧大时，该股票的回报特征确实不同（不是随机噪音）
3. 用三模型设计证明"分歧是结构性的，不是 prompt 工程的产物"
4. 四象限框架不再用来做"alpha 策略"，而是"案例分类器"
5. 连接到 LLM-as-a-Judge：分歧→案例难度→需要人类审查

**目标期刊**：EMNLP / ACL / NeurIPS
**优点**：NLP 社区对"LLM 评估方法论"接受度高，不需要金融理论
**缺点**：需要新数据（Judge benchmark），不能复用已有分析

### 路线 C：系统升级——三模型+微调+知识库，从测量到系统

**定位**：构建一个"多源多模型 Agent 系统"，分歧信号是系统副产品

**核心**：
1. 三模型（Qwen/Phi/Gemma）+ 三微调 + 三知识库 = 真正的 multi-source system
2. 核心贡献不是"entropy premium"，而是"structured disagreement as system diagnostic"
3. 消融实验：E1→E4，每一步信号增强多少
4. 金融回报预测只是 validation 之一，不是唯一目标
5. Judge quality、case difficulty、system calibration 都是并列的贡献

**目标期刊**：EMNLP (系统论文) + NeurIPS Workshop + 金融期刊（后续）
**优点**：贡献最多，和 EWA-Fed / MCP / GraphRAG 技术栈打通
**缺点**：工程量最大，需要 GPU 资源，发表周期长

---

## 我的推荐：路线 B+C 合并

**Phase 1 (2周)**：路线 B 的核心实验——三模型零样本 + 匿名化测试 + familiarity 检验

**Phase 2 (4周)**：路线 C 的系统构建——三微调 + 三知识库 + 消融矩阵

**Phase 3 (2周)**：写成一篇论文，投稿 EMNLP/NeurIPS

**Delta JFE 论文的处理**：不是撤稿，而是诚实修订——降低主张到数据能支持的水平，作为"measurement and evaluation paper"而非"discovery paper"。如果 JFE 拒稿，转投 JFQA。

---

## Phase 1 详细执行计划（路线 B）

### Day 1-2: 基础设施

**API 接入**：
- Qwen2.5-7B-Instruct: DashScope（已有 key）
- Phi-3.5-mini-instruct: OpenRouter（需注册）
- Gemma2-2b-it: OpenRouter 或 Google AI（需注册）
- 备选：DashScope 三规模 Qwen（qwen-plus / qwen2.5-7b / qwen2.5-1.5b）

**统一 API wrapper** (`multi_model_api.py`)：
```python
class MultiModelAPI:
    def __init__(self):
        self.models = {
            'qwen': DashScopeClient(model='qwen2.5-7b-instruct'),
            'phi': OpenRouterClient(model='microsoft/phi-3.5-mini-instruct'),
            'gemma': OpenRouterClient(model='google/gemma-2-2b-it'),
        }
    
    def score(self, model, prompt, temperature=0.3):
        return self.models[model].complete(prompt, temperature)
```

### Day 3-5: 三模型评分

**策略**：先用 High 层 46 只股票做快速验证

| 配置 | 调用数 | 用途 |
|---|---|---|
| 46 stocks × 80 months × 3 models × 3 agents | 33,120 | 主分析 |
| 40 stocks × 80 months × 3 models × 1 agent (anonymized) | 9,600 | familiarity 测试 |

**总调用量**：~43K calls
- DashScope: ~33K（免费额度内）
- OpenRouter: ~22K（约 $2-5）

### Day 6-7: 关键检验

**检验 1：三模型 vs 单模型分歧对比**
- 计算 same-model JS（原版）vs cross-model JS
- 如果 cross-model JS 显著不同 → 证明原版"分歧"是 prompt sensitivity
- 如果 cross-model JS 信号更强 → 支持"架构异质性"论点

**检验 2：匿名化测试**
- 对比匿名 vs 具名评分的熵分布
- 如果匿名后大盘股低熵消失 → familiarity bias 实锤
- 如果匿名后大盘股仍低熵 → 效应来自数值模式（股价特征），不是训练记忆

**检验 3：控制变量全面报告**
- 报告每个信号在 0/1/2/3/4 个控制变量下的 t 值
- 透明展示"信号有多少被 controls 吸收"
- 这不是"weakness"，是"honesty"，NLP 社区更看重这个

**检验 4：Horizon 效应确认**
- Hodrick SE 已验证（t 从 3.02→4.96）
- 加入控制变量后再检验
- 如果带控制后 horizon 仍显著 → 这是论文最可靠的发现

### Day 8-10: 论文重构

**新标题候选**：
1. *"Measuring LLM Agent Disagreement: From Prompt Sensitivity to Architectural Heterogeneity"*
2. *"When Models Disagree: Information-Theoretic Metrics for Multi-LLM Evaluation"*

**新叙事线**：
1. 问题：现有 LLM-as-a-Judge 用 agreement rate 度量一致性，但一个模型多次采样的一致性 ≠ 真正的判断一致性
2. 方法：用信息论度量（JS/H/IC）在 cross-model 设置下度量分歧
3. 实验：金融数据（183 stocks × 80 months × 3 models）
4. 发现：
   - 同模型分歧 ≈ prompt sensitivity（低信息量）
   - 跨模型分歧 ≈ 架构异质性（有结构，有信息）
   - 高分歧案例 = 高难度案例 → 需要 human oversight
   - Inner Confidence 预测 judge 可靠性
5. 消融：从 E1→E2，每步信号质量的变化

---

## Phase 2 详细执行计划（路线 C）

### Day 11-14: 微调准备

**数据收集**：
- FinNLP / Financial PhraseBank → Sentiment 微调数据
- Technical indicator descriptions + price pattern labels → Technical 微调数据
- SEC filing summaries + CFA textbook → Fundamental 微调数据

**微调方案**：
- LoRA (rank=16, alpha=32)
- DashScope 微调 API（如果可用）或租 GPU
- 每个 model ~2-4h on A100

### Day 15-18: 知识库构建

**Sentiment KB**：
- 新闻标题 + 摘要（yfinance news API / Finnhub）
- BGE embedding → FAISS/HNSW 索引

**Technical KB**：
- 技术指标百科 + 历史信号描述
- 纯数值，不需要 RAG

**Fundamental KB**：
- SEC EDGAR 10-K 摘要
- FRED 宏观数据描述

### Day 19-22: 消融实验 E1→E4

| 实验 | 模型 | 微调 | 知识库 | 预期 |
|---|---|---|---|---|
| E1 | Same(Qwen) × 3 prompts | — | — | Baseline (low signal) |
| E2 | Different × 3 prompts | — | — | 架构异质性 |
| E3 | Different × 3 prompts | Domain LoRA | — | +专业训练 |
| E4 | Different × 3 prompts | Domain LoRA | Domain KB | +信息异质性 |

**只对 High 层 46 stocks 做，降低成本**

**预期递进**：E1 的 JS-H 回报预测力 < E2 < E3 < E4

**如果不符合预期**：E4 不比 E2 更好 → 说明微调/知识库没有增加信号，论文需要解释为什么

### Day 23-28: LLM-as-a-Judge 实验

**数据**：
- MT-Bench (80 questions, multi-turn)
- Chatbot Arena (subset with human preference data)
- FairEval (pairwise comparison)

**方法**：
1. 用三模型做 Judge，打分 + 评理
2. 计算 Judge 间 JS/H/IC
3. 与人类标注者一致性对比
4. 建立映射：Judge 分歧 → 案例难度 → 需要人类审查概率

**核心产出**：
- 一个"Disagreement-based Judge Quality Score"
- 当 DJQS > 阈值时，自动标记为"needs human review"
- 在 MT-Bench 上验证：DJQS 高的案例，人类 inter-annotator agreement 也低

### Day 29-35: 论文写作

**结构**：
1. Introduction: Why disagreement matters for LLM evaluation
2. Related Work: LLM-as-a-Judge + information theory + multi-agent systems
3. Method: Cross-model information-theoretic disagreement metrics
4. Experiment 1: Financial data (三模型 × 183 stocks)
5. Experiment 2: Ablation (E1→E4)
6. Experiment 3: LLM-as-a-Judge validation
7. Discussion: From financial signal to evaluation methodology

---

## 对 Delta JFE 论文的处理

### 诚实修订方案（如果仍投金融期刊）

1. **主表**：报告 H_smooth_resid 的完整控制变量结果 (t=-1.66)
2. **交互效应**：报告 proper specification (Y~H+D+H×D)，t=0.69
3. **标题**：从 "The Entropy Premium" 改为 "LLM Agent Entropy and Stock Returns: Evidence and Confounds"
4. **理论框架**：降级为 "motivating framework" 而非 "testable predictions"
5. **核心贡献**：从"发现新因子"改为"评估 LLM 分歧度量的信息含量"
6. **增加**：匿名化实验、familiarity 控制、三模型验证
7. **Horizon**：保留但加 Hodrick SE

**预期**：论文仍可发表，但贡献等级从"发现"降为"方法论评估"

---

## 与技术栈的整合

| 现有资产 | 在新方案中的角色 |
|---|---|
| Delta 40K+ API calls + panel data | Baseline (E1) 对比数据 |
| DashScope API + checkpoint system | Phase 1 评分基础设施 |
| FOMC CB-LM pipeline | Sentiment 微调数据 + 知识库构建模板 |
| EWA-Fed 联邦学习 | Phase 2 多模型聚合方法 |
| Rust HNSW 向量库 | 知识库 RAG 的 similarity search |
| Streamlit dashboard | 可视化四象限 + Judge 分歧 |

---

## 时间线总览

```
Week 1 (Day 1-7):   Phase 1 — 三模型评分 + 匿名化测试 + 关键检验
Week 2 (Day 8-14):  Phase 1 — 论文重构 / Delta JFE 诚实修订
Week 3 (Day 15-21): Phase 2 — 微调 + 知识库构建
Week 4 (Day 22-28): Phase 2 — 消融实验 + Judge 实验
Week 5 (Day 29-35): Phase 2 — 论文写作
Week 6 (Day 36-40): Dashboard + 清理
```

---

## 关键决策点

### Decision 1 (Day 2): API 可用性
- OpenRouter 可用 → Qwen + Phi + Gemma 三家族
- 不可用 → DashScope 三规模 Qwen

### Decision 2 (Day 7): 匿名化结果
- 匿名后效应仍在 → 可以声称"信号来自数值模式"
- 匿名后效应消失 → familiarity 实锤，论文定位降级

### Decision 3 (Day 10): 带控制的 horizon 结果
- 带控制后 horizon 仍显著 → 论文核心发现
- 不显著 → 彻底放弃金融定价主张

### Decision 4 (Day 22): E4 vs E2 对比
- E4 > E2 → 微调+知识库增加信号 → 写系统论文
- E4 ≈ E2 → 零样本足够 → 写方法论文（更简洁）

---

## 底线

现有 Delta 论文的核心统计结果**不能在标准控制变量下生存**。这不是修修补补能解决的问题——是论文基本主张需要重新定位。

最务实的路径：**把 Delta 从"发现新因子"的金融论文，转为"LLM 分歧度量方法论"的 AI 论文**。金融数据成为 testbed 而非 target application。这样，控制变量吃掉信号就不是致命缺陷，而是"methodological finding"——LLM 评分的熵与已知因子高度共线，因此不能声称独立定价能力，但分歧本身的结构信息仍然有方法论价值。
