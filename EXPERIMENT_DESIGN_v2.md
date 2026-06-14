# Delta v2: Experiment Redesign

## 1. 核心定位

**从"发现 anomaly"转向"设计可部署的多 Agent 分歧信号系统，验证其经济价值"**

v1 的问题：同一模型换 prompt → 分歧是语义漂移，不是真正的异质信念
v2 的方案：同一基座 + 三路微调 + 三套知识库 → 分歧来自不同专业视角

## 2. Agent 架构

### 2.1 三路专业 Agent

| Agent | 微调数据 | RAG 知识库 | 专业视角 |
|-------|---------|-----------|---------|
| Sentiment Agent | 金融新闻情绪标注数据 | 新闻/社交媒体/研报摘要 | 市场情绪与叙事 |
| Technical Agent | 技术分析标注数据 | 量价指标/K线形态/交易信号 | 价格动量与趋势 |
| Fundamental Agent | 财报分析标注数据 | 财报/行业报告/宏观指标 | 基本面与估值 |

### 2.2 基座模型选择

**Qwen2.5-7B-Instruct** — 理由：
- 中英双语（A-share 交叉验证需要中文能力）
- 7B 参数量，LoRA 微调单卡可跑
- Apache 2.0 开源，可商用
- vLLM 部署推理速度快

### 2.3 微调方案（LoRA）

```python
# 每个 Agent 用 LoRA 微调
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

**微调数据**（每个 Agent ~5K 条）：
- Sentiment: FinNERTone 标注的新闻 → (新闻, rating) 对
- Technical: 量价特征 + 标注 → (技术指标描述, rating) 对
- Fundamental: 财报指标 + 标注 → (基本面描述, rating) 对

### 2.4 RAG 知识库

每个 Agent 配独立向量库：
- Sentiment KB: 新闻全文、研报摘要、社交媒体帖 → ChromaDB / FAISS
- Technical KB: 技术指标说明、历史形态案例 → ChromaDB / FAISS
- Fundamental KB: 财报原文、行业报告、宏观数据 → ChromaDB / FAISS

检索时：query = 当前股票 + 最近事件 → top-k 上下文 → 注入 prompt

**关键**：知识库设截止时间（cutoff），每个评分月只检索该月之前的信息，杜绝 look-ahead

## 3. 实验设计

### 3.1 分歧来源分解（核心贡献）

| 组 | 基座 | 微调 | RAG | 分歧来源 | 预期 |
|----|------|------|-----|---------|------|
| A | 同一 | 无(prompt) | 无 | 纯语义漂移 | 最弱预测力 |
| B | 同一 | 三路LoRA | 三套KB | 专业视角差异 | **最强预测力** |
| C | 三个不同模型 | 无 | 无 | 模型异质性 | 中等预测力 |

**核心假设**：B > C > A
- B > A：专业视角差异 > 语义漂移
- B > C：定向信息处理 > 无序模型差异
- C > A：任何异质性 > 无异质性

### 3.2 四象限分析框架

替代正交化残差，用直观的 2×2 分组：

```
              D_post (Rating Dispersion)
              Low          High
         ┌────────────┬────────────┐
  Low    │ Concordant │Overconfident│  ← 核心：低熵+高分歧=过度自信
H_smooth │  (一致)    │  (过度自信) │
         ├────────────┼────────────┤
 High    │ Uncertain  │  Noisy     │
         │ (不确定)   │  (噪声)    │
         └────────────┴────────────┘
```

**预测排序**（基于 Daniel et al. 1998 overconfidence 模型）：
- Overconfident > Noisy > Uncertain > Concordant
- 即：Overconfident 组未来收益最低（过度自信→过度反应→长期反转）

### 3.3 Look-ahead Bias 控制

1. **知识库时间截止**：RAG 检索只返回评分月之前的信息
2. **市值分组对比**：大市值（LLM 熟悉）vs 小市值（LLM 不熟悉）
   - 如果熵效应在小市值也显著 → 排除"熟悉度"渠道
   - 如果只在大市值显著 → 熟悉度是混淆变量
3. **子期分析移至正文**：Table 11 升级为主表

### 3.4 标准误

| 方法 | 适用场景 | 本次用途 |
|------|---------|---------|
| Newey-West (4 lags) | 异方差+自相关 | 基准报告 |
| Hodrick (1992) | 重叠回报 | 3Q/4Q horizon 并排报告 |
| Bootstrap | 稳健性 | 补充验证 |

## 4. 代码架构

```
delta/
├── app.py                          # Streamlit dashboard (已有)
├── agents/
│   ├── base_agent.py               # Agent 基类
│   ├── sentiment_agent.py          # 情绪分析 Agent
│   ├── technical_agent.py          # 技术分析 Agent
│   ├── fundamental_agent.py        # 基本面 Agent
│   └── lora_finetune.py            # LoRA 微调脚本
├── rag/
│   ├── build_kb.py                 # 构建知识库
│   ├── retriever.py                # 检索器
│   └── kb_data/                    # 知识库原始数据
│       ├── sentiment/
│       ├── technical/
│       └── fundamental/
├── experiments/
│   ├── step1_data.py               # 数据准备
│   ├── step2_scoring.py            # 多组 Agent 评分
│   ├── step3_analysis.py           # 四象限 + FM 回归
│   ├── step4_robustness.py         # Hodrick SE + 子期 + 市值分组
│   └── step5_comparison.py         # A vs B vs C 对比
├── figures_final/                  # 论文图表
└── requirements.txt
```

## 5. 时间规划

| 阶段 | 内容 | 预计 |
|------|------|------|
| Phase 1 | 数据准备 + Agent 架构搭建 | 1 周 |
| Phase 2 | LoRA 微调 + RAG 知识库构建 | 1 周 |
| Phase 3 | 三组实验评分（A/B/C） | 3-5 天 |
| Phase 4 | 四象限分析 + Hodrick SE | 2-3 天 |
| Phase 5 | 论文重写 | 1 周 |

## 6. 论文叙事（v2）

**Title**: Delta: Disagreement-Preserving Multi-Agent Collaboration

**One-liner**: 我们设计了一个多 Agent 评分系统，每个 Agent 经专业微调并配备独立知识库，发现 Agent 间分歧的信息集中度（熵）预测股票截面收益，且这种预测力来自专业视角的差异，而非模型异质性或语义漂移。

**核心贡献**：
1. **架构贡献**：同一基座 + 三路微调 + 三套知识库 → 可部署的多 Agent 分歧信号系统
2. **方法论贡献**：分歧来源分解（A/B/C 三组）→ 什么类型的分歧最有经济价值
3. **实证贡献**：四象限框架 → overconfidence 渠道的直接证据
4. **稳健性贡献**：Hodrick SE + 市值分组 + 知识库时间截止 → 三重 look-ahead 控制
