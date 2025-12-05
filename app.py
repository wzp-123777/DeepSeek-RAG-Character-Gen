import streamlit as st
import os
import tempfile
import json
import re

# 设置 HuggingFace 镜像，解决国内连接问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llm_client import LLMClient
from rag_engine import RAGEngine
from history_utils import save_history_item, load_history, delete_history_item

RAG_CONFIG_FILE = "rag_config.json"

def save_rag_config(config):
    try:
        with open(RAG_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except Exception as e:
        print(f"保存配置失败: {e}")

def load_rag_config():
    if os.path.exists(RAG_CONFIG_FILE):
        try:
            with open(RAG_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return None

# 页面配置
st.set_page_config(page_title="DeepSeek RAG 角色生成器", layout="wide")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gen_messages" not in st.session_state:
    st.session_state.gen_messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "llm_client" not in st.session_state:
    st.session_state.llm_client = None
if "vector_db_ready" not in st.session_state:
    st.session_state.vector_db_ready = False

def init_rag(embedding_type, model_name, api_key=None, base_url=None):
    try:
        return RAGEngine(
            embedding_type=embedding_type, 
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        st.error(f"初始化 RAG 引擎失败: {e}")
        return None

# 尝试自动加载本地知识库配置
if not st.session_state.rag_engine and os.path.exists("./chroma_db") and os.path.exists(RAG_CONFIG_FILE):
    config = load_rag_config()
    if config:
        e_type = config.get("embedding_type")
        api_key_to_use = config.get("api_key")
        
        # 如果是 API 模式且没有保存 Key，尝试从用户配置读取（假设复用）
        if e_type == "api" and not api_key_to_use:
             # 这里需要临时加载一下 user config
             if os.path.exists("user_config.json"):
                 try:
                     with open("user_config.json", "r", encoding="utf-8") as f:
                         user_conf = json.load(f)
                         api_key_to_use = user_conf.get("api_key")
                 except:
                     pass

        if e_type == "local" or (e_type == "api" and api_key_to_use):
            with st.spinner("正在自动加载上次的知识库..."):
                st.session_state.rag_engine = init_rag(
                    embedding_type=e_type,
                    model_name=config["model_name"],
                    api_key=api_key_to_use,
                    base_url=config.get("base_url")
                )
                if st.session_state.rag_engine:
                    # 简单的验证一下是否真的有数据
                    # 这里不进行深层检查，假设 chroma_db 存在即有效
                    st.session_state.vector_db_ready = True
                    # st.toast("已自动加载上次的知识库") # toast 在这里可能显示不出来，因为还没渲染页面

USER_CONFIG_FILE = "user_config.json"

def save_user_config(config):
    try:
        with open(USER_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except Exception as e:
        print(f"保存用户配置失败: {e}")

def load_user_config():
    if os.path.exists(USER_CONFIG_FILE):
        try:
            with open(USER_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

def main():
    st.title("🤖 DeepSeek RAG 角色提示词生成器")
    
    # 加载用户配置
    user_config = load_user_config()
    
    # --- 侧边栏配置 ---
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # API 配置
        default_provider_index = 0
        if user_config.get("api_provider") == "siliconflow":
            default_provider_index = 1
            
        api_provider = st.selectbox("选择 LLM 提供商", ["deepseek", "siliconflow"], index=default_provider_index)
        
        default_api_key = user_config.get("api_key", "")
        api_key = st.text_input("API Key", value=default_api_key, type="password", help="输入对应的 API Key")
        
        # 初始化 LLM Client
        if api_key:
            try:
                st.session_state.llm_client = LLMClient(provider=api_provider, api_key=api_key)
                models = st.session_state.llm_client.get_available_models()
                
                default_model_index = 0
                saved_model = user_config.get("model_name")
                if saved_model and saved_model in models:
                    default_model_index = models.index(saved_model)
                
                selected_model = st.selectbox("选择对话模型", models, index=default_model_index)
                st.success(f"已连接到 {api_provider}")
                
                # 保存配置（当连接成功时）
                if api_key != user_config.get("api_key") or api_provider != user_config.get("api_provider") or selected_model != user_config.get("model_name"):
                    save_user_config({
                        "api_provider": api_provider,
                        "api_key": api_key,
                        "model_name": selected_model
                    })
                    
            except Exception as e:
                st.error(f"连接失败: {e}")
        else:
            st.warning("请输入 API Key 以继续")

        st.divider()

        # RAG 配置
        st.subheader("📚 知识库构建")
        
        rag_config = load_rag_config() or {}
        
        # 默认选中上次的模式
        default_mode_index = 0
        if rag_config.get("embedding_type") == "api":
            default_mode_index = 1
            
        rag_mode = st.radio("Embedding 模式", ["本地 (HuggingFace)", "云端 API (SiliconFlow)"], index=default_mode_index)
        
        embedding_model_name = ""
        rag_api_key = None
        rag_base_url = None
        
        if rag_mode == "本地 (HuggingFace)":
            default_model = rag_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            if rag_config.get("embedding_type") != "local": # 如果上次不是 local，就用默认值
                 default_model = "sentence-transformers/all-MiniLM-L6-v2"
                 
            embedding_model_name = st.text_input("模型名称", value=default_model)
            st.caption("提示：首次运行会自动下载模型。已配置国内镜像加速。")
        else:
            # 尝试找到上次使用的模型 index
            model_options = [
                    "BAAI/bge-m3", 
                    "BAAI/bge-large-zh-v1.5", 
                    "Qwen/Qwen3-Embedding-8B", # 用户指定
                    "netease-youdao/bce-embedding-base_v1"
                ]
            default_emb_index = 0
            if rag_config.get("embedding_type") == "api" and rag_config.get("model_name") in model_options:
                default_emb_index = model_options.index(rag_config.get("model_name"))
                
            embedding_model_name = st.selectbox(
                "选择 Embedding 模型", 
                model_options,
                index=default_emb_index
            )
            st.caption("推荐使用 BAAI/bge-m3 或 Qwen/Qwen3-Embedding-8B")
            # 默认复用上面的 API Key，如果用户需要单独设置也可以
            use_same_key = st.checkbox("使用上方相同的 API Key", value=True)
            if use_same_key:
                rag_api_key = api_key
            else:
                rag_api_key = st.text_input("Embedding API Key", type="password")
            
            rag_base_url = "https://api.siliconflow.cn/v1"

        uploaded_files = st.file_uploader("上传大文本 (txt, pdf, docx)", accept_multiple_files=True)
        kb_name = st.text_input("目标知识库名称 (仅限字母、数字、下划线)", value="default_kb", help="将文件存入指定的知识库分组中。注意：不支持中文，长度3-63字符。")
        
        if st.button("构建/更新 知识库"):
            # 校验知识库名称
            if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$', kb_name):
                 st.error("知识库名称格式错误！只能包含字母、数字、下划线、连字符和点，且长度在3-63之间，首尾必须是字母或数字。")
            elif not uploaded_files:
                st.error("请先上传文件")
            else:
                if rag_mode == "云端 API (SiliconFlow)" and not rag_api_key:
                    st.error("请提供 API Key")
                else:
                    with st.spinner("正在处理文档..."):
                        # 确定参数
                        e_type = "local" if rag_mode == "本地 (HuggingFace)" else "api"
                        
                        # 初始化 RAG 引擎
                        st.session_state.rag_engine = init_rag(
                            embedding_type=e_type,
                            model_name=embedding_model_name,
                            api_key=rag_api_key,
                            base_url=rag_base_url
                        )
                    
                        if st.session_state.rag_engine:
                            # 保存上传的文件到临时目录
                            temp_dir = tempfile.mkdtemp()
                            file_paths = []
                            for uploaded_file in uploaded_files:
                                file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                file_paths.append(file_path)
                            
                            # 加载和切分
                            docs = st.session_state.rag_engine.load_documents(file_paths)
                            if isinstance(docs, str): # Error message
                                st.error(docs)
                            else:
                                # 构建向量库
                                # 使用用户指定的 collection name，如果为空则使用默认
                                target_collection = kb_name.strip() if kb_name.strip() else "character_data"
                                msg = st.session_state.rag_engine.build_vector_store(docs, collection_name=target_collection)
                                st.success(msg)
                                st.session_state.vector_db_ready = True
                                
                                # 保存配置
                                save_rag_config({
                                    "embedding_type": e_type,
                                    "model_name": embedding_model_name,
                                    "base_url": rag_base_url,
                                    "api_key": rag_api_key # 保存 Key
                                })
                        
                        # 清理临时文件
                        # shutil.rmtree(temp_dir) # 可以在适当时候清理

        if st.button("清空知识库"):
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_database()
                st.session_state.vector_db_ready = False
                st.success("知识库已清空")
                # 删除配置文件
                if os.path.exists(RAG_CONFIG_FILE):
                    try:
                        os.remove(RAG_CONFIG_FILE)
                    except:
                        pass
                st.rerun()

        # 显示已有知识库内容
        if st.session_state.vector_db_ready and st.session_state.rag_engine:
            st.divider()
            with st.expander("📂 已收录文档列表", expanded=True):
                summary = st.session_state.rag_engine.get_documents_summary()
                if summary:
                    for kb, info in summary.items():
                        files = info['files']
                        count = info['count']
                        st.markdown(f"**📦 {kb}** (共 {count} 个片段)")
                        for f in files:
                            st.text(f"  └─ 📄 {f}")
                else:
                    st.caption("暂无文件信息")
    # --- 主界面 ---
    tab1, tab2, tab3 = st.tabs(["🎭 角色提示词生成", "💬 自由对话", "📜 历史记录"])

    # Tab 1: 角色生成
    with tab1:
        st.markdown("### 基于 RAG 生成角色 Prompt")
        if not st.session_state.vector_db_ready:
            st.info("请先在左侧上传文件并构建知识库。")
        
        # 知识库选择
        selected_kbs = []
        if st.session_state.vector_db_ready and st.session_state.rag_engine:
            available_kbs = st.session_state.rag_engine.get_available_collections()
            if available_kbs:
                selected_kbs = st.multiselect("选择检索范围（知识库）", available_kbs, default=available_kbs)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            char_name = st.text_input("角色名称", placeholder="例如：孙悟空")
        with col2:
            char_style = st.selectbox("提示词风格", ["详细设定版", "简短对话版", "JSON格式"])
        with col3:
            retrieve_k = st.number_input("检索片段数", min_value=1, max_value=20, value=8, help="增加此数值可以读取更多原文内容，但会消耗更多 Token")

        extra_req = st.text_area("额外要求 (可选)", placeholder="例如：重点描述他的战斗经历，或者他和某人的关系...")

        if st.button("生成角色提示词", disabled=not (st.session_state.vector_db_ready and st.session_state.llm_client)):
            if not char_name:
                st.warning("请输入角色名称")
            else:
                with st.spinner(f"正在检索关于 {char_name} 的信息并生成..."):
                    # 1. RAG 检索
                    query = f"关于角色 {char_name} 的外貌、性格、身世、说话风格、重要经历、人际关系。"
                    if extra_req:
                        query += f" 额外关注：{extra_req}"
                    
                    # 使用选中的知识库进行检索
                    retrieved_docs = st.session_state.rag_engine.query(query, k=retrieve_k, collection_names=selected_kbs) 
                    
                    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # 显示检索到的内容 (用于调试/确认)
                    with st.expander(f"查看检索到的原文片段 (共 {len(retrieved_docs)} 个片段)"):
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**片段 {i+1}** (Source: {doc.metadata.get('source', 'unknown')}):")
                            # 显示完整内容，不再截断
                            st.text(doc.page_content)
                            st.divider()

                    # 2. 构建 Prompt
                    # 定义系统角色：Prompt 专家
                    sys_instruction = "你是一个专业的角色设定专家。你的任务是根据用户提供的资料，撰写或修改大语言模型的角色扮演提示词（System Prompt）。请始终保持客观、专业的态度，直接输出优化后的 Prompt，不要进行角色扮演，也不要输出无关的闲聊。"
                    
                    user_task = f"""目标角色：{char_name}
风格要求：{char_style}

请从以下原文片段中提取信息：
1. 外貌特征
2. 性格特点（包括优点和缺点）
3. 说话风格（口癖、语气、常用词）
4. 背景故事和重要经历
5. 人际关系

原文片段：
{context_text}

用户额外要求：{extra_req}

请输出一个结构清晰、可以直接复制使用的 System Prompt。如果原文信息不足，请根据角色设定进行合理的逻辑推断，但不要捏造与原文冲突的事实。
"""
                    # 重置对话历史
                    st.session_state.gen_messages = [
                        {"role": "system", "content": sys_instruction},
                        {"role": "user", "content": user_task}
                    ]

                    # 3. 调用 LLM
                    full_response = ""
                    
                    with st.chat_message("assistant"):
                        # 流式输出
                        stream = st.session_state.llm_client.chat(st.session_state.gen_messages, model=selected_model, stream=True)
                        
                        if isinstance(stream, str): # Error
                            st.error(stream)
                        else:
                            response_placeholder = st.empty()
                            for chunk in stream:
                                if chunk.choices[0].delta.content:
                                    content = chunk.choices[0].delta.content
                                    full_response += content
                                    response_placeholder.markdown(full_response)
                            
                            st.session_state.gen_messages.append({"role": "assistant", "content": full_response})
                            st.rerun()

        # 显示生成历史和对话
        for msg in st.session_state.gen_messages:
            if msg["role"] == "system":
                continue # 不显示系统指令
            if msg["role"] == "user":
                # 隐藏初始的大段 Prompt，只显示后续的修改意见
                if msg["content"].startswith("目标角色："):
                    with st.expander("查看初始 Prompt 请求"):
                        st.text(msg["content"])
                else:
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

        # 保存按钮
        if st.session_state.gen_messages and st.session_state.gen_messages[-1]["role"] == "assistant":
            if st.button("💾 保存当前 Prompt 到历史记录"):
                last_response = st.session_state.gen_messages[-1]["content"]
                save_history_item(char_name, last_response)
                st.success("已保存！")

        # 修改意见输入框
        if prompt := st.chat_input("对结果不满意？请输入修改意见...", key="gen_chat"):
            if not st.session_state.gen_messages:
                st.warning("请先生成角色提示词")
            else:
                st.session_state.gen_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    stream = st.session_state.llm_client.chat(st.session_state.gen_messages, model=selected_model, stream=True)
                    if isinstance(stream, str):
                        st.error(stream)
                    else:
                        full_response = ""
                        response_placeholder = st.empty()
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                response_placeholder.markdown(full_response)
                        st.session_state.gen_messages.append({"role": "assistant", "content": full_response})

    # Tab 2: 自由对话
    with tab2:
        st.markdown("### 与模型对话 (可选 RAG)")
        enable_rag = st.checkbox("启用 RAG (引用知识库)", value=True, disabled=not st.session_state.vector_db_ready)
        
        # 显示历史消息
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("输入你的问题..."):
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 生成回复
            with st.chat_message("assistant"):
                if not st.session_state.llm_client:
                    st.error("请先配置 API Key")
                else:
                    context_str = ""
                    if enable_rag and st.session_state.vector_db_ready:
                        with st.spinner("检索中..."):
                            docs = st.session_state.rag_engine.query(prompt, k=3)
                            context_str = "\n\n".join([doc.page_content for doc in docs])
                            with st.expander("参考上下文"):
                                st.text(context_str)
                    
                    # 构建消息
                    messages_payload = []
                    # 如果有 RAG 上下文，插入到 System Prompt 或 User Prompt 中
                    if context_str:
                        system_msg = f"你是一个助手。请基于以下上下文回答用户的问题。\n\n上下文：\n{context_str}"
                        messages_payload.append({"role": "system", "content": system_msg})
                    
                    # 添加历史记录 (简单处理，只取最近几轮以节省 token)
                    for m in st.session_state.messages[-5:]:
                        messages_payload.append(m)
                    
                    # 如果没有 RAG 且没有历史 system prompt，可以加一个默认的
                    if not context_str and not any(m['role'] == 'system' for m in messages_payload):
                         messages_payload.insert(0, {"role": "system", "content": "你是一个乐于助人的助手。"})

                    # 调用 LLM
                    response_placeholder = st.empty()
                    full_response = ""
                    stream = st.session_state.llm_client.chat(messages_payload, model=selected_model, stream=True)
                    
                    if isinstance(stream, str):
                        st.error(stream)
                    else:
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                response_placeholder.markdown(full_response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Tab 3: 历史记录
    with tab3:
        st.markdown("### 📜 历史 Prompt 记录")
        history = load_history()
        if not history:
            st.info("暂无历史记录。")
        else:
            for i, item in enumerate(history):
                with st.expander(f"{item['timestamp']} - {item['char_name']}"):
                    st.code(item['content'], language="markdown")
                    if st.button("删除", key=f"del_{i}"):
                        delete_history_item(i)
                        st.rerun()

if __name__ == "__main__":
    main()
