# Happy LLM

## 大纲

### 第一章 NLP 基础概念
- 1.1 什么是 NLP
- 1.2 NLP 发展历程
- 1.3 NLP 任务
  - 1.3.1 中文分词
  - 1.3.2 子词切分
  - 1.3.3 词性标注
  - 1.3.4 文本分类
  - 1.3.5 实体识别
  - 1.3.6 关系抽取
  - 1.3.7 文本摘要
  - 1.3.8 机器翻译
  - 1.3.9 自动问答
- 1.4 文本表示的发展历程
  - 1.4.1 词向量
  - 1.4.2 语言模型
  - 1.4.3 Word2Vec
  - 1.4.4 ELMo

### 第二章 Transformer 架构
- 2.1 注意力机制
  - 2.1.1 什么是注意力机制
  - 2.1.2 深入理解注意力机制
  - 2.1.3 注意力机制的实现
  - 2.1.4 自注意力
  - 2.1.5 掩码自注意力
  - 2.1.6 多头注意力
- 2.2 Encoder-Decoder
  - 2.2.1 Seq2Seq 模型
  - 2.2.2 前馈神经网络
  - 2.2.3 层归一化
  - 2.2.4 残差连接
  - 2.2.5 Encoder
  - 2.2.6 Decoder
- 2.3 搭建一个 Transformer
  - 2.3.1 Embeddng 层
  - 2.3.2 位置编码
  - 2.3.3 一个完整的 Transformer

### 第三章 预训练语言模型

- 3.1 Encoder-only PLM
  - 3.1.1 BERT
  - 3.1.2 RoBERTa
  - 3.1.3 ALBERT
- 3.2 Encoder-Decoder PLM
  - 3.2.1 T5
- 3.3 Decoder-Only PLM
  - 3.3.1 GPT
  - 3.3.2 LLaMA
  - 3.3.3 GLM
  - 3.3.4 DeepSeek [WIP]
     
### 第四章 大语言模型

- 4.1 什么是 LLM
  - 4.1.1 LLM 的定义
  - 4.1.2 LLM 的能力
  - 4.1.3 LLM 的特点
- 4.2 如何训练一个 LLM
  - 4.2.2 Pretrain
  - 4.2.3 SFT
  - 4.2.4 RLHF

### 第五章 动手搭建大模型

- 5.1 动手实现一个 LLaMA2 大模型
  - 5.1.1 定义超参数
  - 5.1.2 构建 RMSNorm
  - 5.1.3 构建 LLaMA2 Attention
    - 5.1.3.1 repeat_kv
    - 5.1.3.2 旋转嵌入
    - 5.1.3.3 组装 LLaMA2 Attention
  - 5.1.4 构建 LLaMA2 MLP模块
  - 5.1.5 LLaMA2 Decoder Layer
  - 5.1.6 构建 LLaMA2 模型
- 5.2 训练 Tokenizer
  - 5.3.1 Word-based Tokenizer
  - 5.2.2 Character-based Tokenizer
  - 5.2.3 Subword Tokenizer
  - 5.2.4 训练一个 Tokenizer
- 5.3 预训练一个小型LLM
  - 5.3.0 数据下载
- 1 处理预训练数据
- 2 处理SFT数据
  - 5.3.1 训练Tokenize
  - 5.3.2 Dataset
  - 5.3.3 预训练
  - 5.3.4 SFT 训练
  - 5.3.4 使用模型生成文本

### 第六章 大模型训练流程实践

- 6.1 模型预训练
  - 6.1.1 框架介绍
  - 6.1.2 初始化 LLM
  - 6.1.3 预训练数据处理
  - 6.1.4 使用 Trainer 进行训练
  - 6.1.5 使用 DeepSpeed 实现分布式训练
- 6.2 模型有监督微调
  - 6.2.1 Pretrain VS SFT
  - 6.2.2 微调数据处理
- 6.3 高效微调
  - 6.3.1 高效微调方案
  - 6.3.2 LoRA 微调
  - 6.3.3 LoRA 微调的原理
  - 6.3.4 LoRA 的代码实现
  - 6.3.5 使用 peft 实现 LoRA 微调

### 第七章 大模型应用

- 7.1 LLM 的评测
  - 7.1.1 LLM 的评测数据集
  - 7.1.2 主流的评测榜单
  - 7.1.3 特定的评测榜单
- 7.2 RAG
  - 7.2.1 RAG 的基本原理
  - 7.2.2 搭建一个 RAG 框架
- 7.3 Agent
  - 7.3.1 什么是 LLM Agent？
  - 7.3.2 LLM Agent 的类型
  - 7.3.3 动手构造一个 Tiny-Agent