import json
import random
import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation.utils import GenerationConfig

st.set_page_config(page_title="K-Model-215M LLM")
st.title("K-Model-215M LLM")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")


with st.sidebar:
    st.markdown("## K-Model-215M LLM")
    "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
    # 创建一个滑块，用于选择最大长度，范围在 0 到 8192 之间，默认值为 512（Qwen2.5 支持 128K 上下文，并能生成最多 8K tokens）
    st.sidebar.title("设定调整")
    st.session_state.max_new_tokens = st.sidebar.slider("最大输入/生成长度", 128, 512, 512, step=1)
    st.session_state.temperature = st.sidebar.slider("temperature", 0.1, 1.2, 0.75, step=0.01)


model_id = "./k-model-215M/"

# 定义一个函数，用于获取模型和 tokenizer
@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto").eval()
    return tokenizer, model


tokenizer, model = get_model()

# 如果 session_state 中没有 "messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "有什么可以帮您的？"}]

# 遍历 session_state 中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 将用户输入添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 将对话输入模型，获得返回
    input_ids = tokenizer.apply_chat_template(st.session_state.messages,tokenize=False,add_generation_prompt=True)
    input_ids = tokenizer(input_ids).data['input_ids']
    x = (torch.tensor(input_ids, dtype=torch.long)[None, ...])

    with torch.no_grad():
        y = model.generate(x, tokenizer.eos_token_id, st.max_new_tokens, temperature=st.temperature)
        response = tokenizer.decode(y[0].tolist())
        
    # 将模型的输出添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
    # print(st.session_state) # 打印 session_state 调试

