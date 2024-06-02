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

### 第三章 预训练语言模型 
  - 3.1 Encoder-Only PLM 雨衡 Done
    - 3.1.1 BERT
      - （1）模型架构：Encoder Only
      - （2）预训练任务
      - （3）针对下游任务微调
    - 3.1.2 RoBERTa
    - 3.1.3 ALBERT
  - 3.2 Encoder-Decoder PLM 志学 
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
    - 3.3.4 BaiChuan
    - 3.3.5 Qwen
    - 3.3.6 Mistral
    - 3.3.7 MiniCPM
    - 3.3.8 Mixtral-8*7B
      - （1）模型架构：MoE
      - （2）MoE 架构的核心优势
     
### 第四章 大语言模型 雨衡 Done
  - 4.1 什么是 LLM 
    - 4.1.1 LLM 的定义
    - 4.1.2 LLM 的能力
    - 4.1.3 LLM 的特点
  - 4.2 训练 LLM 的三个阶段
    - 4.2.1 Pretrain
    - 4.2.2 SFT
    - 4.2.3 RLHF

### 第五章 预训练一个 LLM 志学
  - 5.1 模型架构-LLaMA Done
    - 5.1.1 LLaMA Attention
    - 5.1.2 LLaMA Decoder Layer
    - 5.1.3 LLaMA MLP
    - 5.1.4 LLaMA RMSNorm
    - 5.1.5 A Whole LLaMA
  - 5.2 预训练数据
    - 5.2.1 预训练数据集
    - 5.2.2 预训练数据处理
  - 5.3 训练 Tokenizer
    - 5.3.1 Word-based Tokenizer
    - 5.3.2 Character-based Tokenzier
    - 5.3.3 Subword Tokenizer
      - （1）BPE
      - （2）Word Piece
      - （3）Unigram
    - 5.3.4 训练一个 Tokenizer
  - 5.4 预训练

### 第六章 微调 LLM
  - 6.1 微调数据
    - 6.1.1 指令数据集
    - 6.1.2 微调数据处理
  - 6.2 SFT
  - 6.3 微调其他 LLM 的通用流程
  - 6.4 高效微调-LoRA
    - 6.4.1 LoRA 原理（注：深入浅出 LoRA）
    - 6.4.2 实践 LoRA 微调
   
### 第七章 RLHF
  - 7.1 RM 训练
  - 7.2 PPO 训练
  - 7.3 RLHF 的平替版本-DPO

### 第八章 LLM 应用
  - 8.1 LLM 的评测
    - 8.1.1 LLM 的评测方法
    - 8.1.2 主流的评测榜单
    - 8.1.3 特定的评测榜单
  - 8.2 Prompt Engineering（注：吴恩达课程）
    - 8.2.1 上下文学习
    - 8.2.2 思维链
    - 8.2.3 Prompt 的迭代优化
  - 8.3 RAG （注：志学-TinyRAG）
    - 8.3.1 RAG 的基本原理
    - 8.3.2 搭建一个 RAG 框架
  - 8.4 Agent
    - 8.4.1 Agent 的基本原理
    - 8.4.2 搭建一个 Multi-Agent 框架
