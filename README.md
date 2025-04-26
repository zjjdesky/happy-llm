# Happy LLM

## 大纲

### 第一章 NLP 基础概念 志学 Done
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

### 第二章 基础架构-Transformer 雨衡 Done
  - 2.1 注意力机制 
    - 2.1.1 注意力机制详解
    - 2.1.2 自注意力与多头注意力
    - 2.1.3 注意力掩码与因果注意力
  - 2.2 Encoder-Decoder
    - 2.2.1 Seq2Seq 模型
    - 2.2.2 Encoder
    - 2.2.3 Decoder
  - 2.3 Transformer 
    - 2.3.1 Transformer 结构总览
    - 2.3.2 Tokenizer 与 Embedding 层
    - 2.3.3 位置编码
    - 2.3.4 Transformer 中的其他结构

### 第三章 预训练语言模型 Partly Done
  - 3.1 Encoder-Only PLM
    - 3.1.1 BERT
      - （1）模型架构：Encoder Only
      - （2）预训练任务
      - （3）针对下游任务微调
    - 3.1.2 RoBERTa
    - 3.1.3 ALBERT
  - 3.2 Encoder-Decoder PLM
    - 3.2.1 T5
      - （1）模型架构：Encoder-Decoder
      - （2）预训练任务
      - （3）大一统思想
    - 3.2.2 BART
    - 3.2.3 XLNet
  - 3.3 Decoder-Only PLM
    - 3.3.1 GPT
      - （1）模型架构：Decoder Only
      - （2）预训练任务
      - （3）GPT 的发展历程
    - 3.3.2 LLaMA
      - （1）模型架构优化
      - （2）预训练数据
      - （3）LLaMA1 到 LLaMA2
    - 3.3.3 ChatGLM
      - （1）模型架构：Prefix-Decoder
      - （2）预训练数据
      - （3）ChatGLM 的发展历程
     
### 第四章 大语言模型 雨衡 Done
  - 4.1 什么是 LLM 
    - 4.1.1 LLM 的定义
    - 4.1.2 LLM 的能力
    - 4.1.3 LLM 的特点
  - 4.2 训练 LLM 的三个阶段
    - 4.2.1 Pretrain
    - 4.2.2 SFT
    - 4.2.3 RLHF

### 第五章 动手搭建大模型
  - 5.1 模型架构-LLaMA Done
    - 5.1.1 LLaMA Attentœion
    - 5.1.2 LLaMA Decoder Layer
    - 5.1.3 LLaMA MLP
    - 5.1.4 LLaMA RMSNorm
    - 5.1.5 A Whole LLaMA
  - 5.2 训练 Tokenizer
    - 5.2.1 Word-based Tokenizer
    - 5.2.2 Character-based Tokenzier
    - 5.2.3 Subword Tokenizer
      - （1）BPE
      - （2）Word Piece
      - （3）Unigram
    - 5.2.4 训练一个 Tokenizer
  - 5.3 训练一个小型LLM
    - 5.3.1 训练Tokenizer
    - 5.3.2 数据预处理
    - 5.3.3 训练模型
    - 5.3.4 使用模型生成文本

### 第六章 训练 LLM
  - 6.1 框架介绍
    - 6.1.1 transformers
    - 6.1.2 deepspeed
    - 6.1.3 peft
    - 6.1.4 trl
  - 6.2 LLM Pretrain
    - 6.2.1 初始化 LLM
    - 6.2.2 预训练数据处理
    - 6.2.3 使用 Trainer 进行预训练
  - 6.3 LLM SFT
    - 6.3.1 加载预训练模型
    - 6.3.2 微调数据处理
    - 6.3.3 使用 Trainer 进行微调
  - 6.4 基于强化学习的偏好对齐
    - 6.4.1 DPO 训练
    - 6.4.2 KTO 训练
    - 6.4.3 GRPO 训练
  - 6.5 高效微调-LoRA
    - 6.5.1 LoRA 原理
    - 6.5.2 使用 peft 进行 LoRA 微调

### 第七章 大模型应用
  - 7.1 LLM 的评测
    - 7.1.1 LLM 的评测方法
    - 7.1.2 主流的评测榜单
    - 7.1.3 特定的评测榜单
  - 7.2 RAG
    - 7.2.1 RAG 的基本原理
    - 7.2.2 搭建一个 RAG 框架
  - 7.3 Agent
    - 7.3.1 Agent 的基本原理
    - 7.3.2 搭建一个 Multi-Agent 框架