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
if "qq_dialogue_messages" not in st.session_state:
    st.session_state.qq_dialogue_messages = []
if "qq_prompt_data" not in st.session_state:
    st.session_state.qq_prompt_data = {
        "character_info": "",
        "background": "",
        "chat_requirements": "",
        "dialogue_examples": [
            {"user": "你好", "character": "你好！有什么可以帮助你的吗？"},
            {"user": "今天天气怎么样？", "character": "今天天气不错，阳光明媚。"},
            {"user": "你最喜欢做什么？", "character": "我喜欢聊天和帮助别人。"},
            {"user": "讲个笑话吧", "character": "为什么鸡过马路？因为对面有肯德基！"},
            {"user": "再见", "character": "再见，有事随时找我！"}
        ]
    }
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

def update_qq_prompt_data():
    """更新QQ prompt数据"""
    st.session_state.qq_prompt_data["character_info"] = st.session_state.edit_character_info
    st.session_state.qq_prompt_data["background"] = st.session_state.edit_background
    st.session_state.qq_prompt_data["chat_requirements"] = st.session_state.edit_chat_requirements

def update_example(index, field):
    """更新对话示例"""
    key = f"{'user' if field == 'user' else 'char'}_msg_{index}"
    if key in st.session_state:
        st.session_state.qq_prompt_data["dialogue_examples"][index][field] = st.session_state[key]

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
        
        # 新增：网页 URL 输入
        st.markdown("或者")
        input_urls = st.text_area("输入网页链接 (每行一个)", height=100, help="支持直接读取网页小说章节内容")
        is_crawl_mode = st.checkbox("这是一个目录页 (自动抓取页面内的章节链接)", value=False, help="勾选后，系统会尝试分析页面中的链接，并抓取所有章节内容。")
        
        kb_name = st.text_input("目标知识库名称 (仅限字母、数字、下划线)", value="default_kb", help="将文件存入指定的知识库分组中。注意：不支持中文，长度3-63字符。")
        
        if st.button("构建/更新 知识库"):
            # 校验知识库名称
            if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$', kb_name):
                 st.error("知识库名称格式错误！只能包含字母、数字、下划线、连字符和点，且长度在3-63之间，首尾必须是字母或数字。")
            elif not uploaded_files and not input_urls.strip():
                st.error("请先上传文件或输入网页链接")
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
                            all_docs = []
                            
                            # 1. 处理上传的文件
                            if uploaded_files:
                                temp_dir = tempfile.mkdtemp()
                                file_paths = []
                                for uploaded_file in uploaded_files:
                                    file_path = os.path.join(temp_dir, uploaded_file.name)
                                    with open(file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    file_paths.append(file_path)
                                
                                file_docs = st.session_state.rag_engine.load_documents(file_paths)
                                if isinstance(file_docs, str):
                                    st.error(f"文件处理错误: {file_docs}")
                                else:
                                    all_docs.extend(file_docs)
                                    
                            # 2. 处理网页链接
                            if input_urls.strip():
                                url_list = [url.strip() for url in input_urls.split('\n') if url.strip()]
                                if url_list:
                                    web_docs = st.session_state.rag_engine.load_urls(url_list, fetch_links=is_crawl_mode)
                                    if isinstance(web_docs, str):
                                        st.error(f"网页处理错误: {web_docs}")
                                    else:
                                        all_docs.extend(web_docs)

                            if not all_docs:
                                st.warning("未能提取到任何有效内容。")
                            else:
                                # 构建向量库
                                # 使用用户指定的 collection name，如果为空则使用默认
                                target_collection = kb_name.strip() if kb_name.strip() else "character_data"
                                msg = st.session_state.rag_engine.build_vector_store(all_docs, collection_name=target_collection)
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

        # --- 知识库管理区域 ---
        st.divider()
        st.subheader("🗑️ 知识库管理")
        
        if st.session_state.vector_db_ready and st.session_state.rag_engine:
            available_kbs = st.session_state.rag_engine.get_available_collections()
            
            # 1. 删除单个知识库
            if available_kbs:
                col_del1, col_del2 = st.columns([3, 1])
                with col_del1:
                    kb_to_delete = st.selectbox("选择要删除的知识库", [""] + available_kbs, key="del_kb_select")
                with col_del2:
                    if kb_to_delete and st.button("删除", key="del_kb_btn"):
                        success, msg = st.session_state.rag_engine.delete_collection(kb_to_delete)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            
            # 2. 清空所有
            if st.button("⚠️ 清空所有知识库", type="primary"):
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.clear_database()
                    st.session_state.vector_db_ready = False
                    st.success("知识库已全部清空")
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
            with st.expander("📂 已收录文档列表", expanded=False):
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
    tab1, tab2, tab3, tab4 = st.tabs(["🎭 角色提示词生成", "💬 自由对话", "🤖 QQ角色生成", "📜 历史记录"])

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
                # 简单的分类逻辑：尝试提取前缀（按 _ 或 - 分割）
                prefixes = set()
                for kb in available_kbs:
                    if "_" in kb:
                        prefixes.add(kb.split("_")[0])
                    elif "-" in kb:
                        prefixes.add(kb.split("-")[0])
                
                # 如果有前缀分类，显示快速筛选
                default_selection = available_kbs
                if prefixes:
                    cols_filter = st.columns([1, 2])
                    with cols_filter[0]:
                        filter_category = st.selectbox("📂 按大类快速筛选", ["全部"] + sorted(list(prefixes)), help="根据知识库名称的前缀（如 '小说A_第一卷' 中的 '小说A'）进行筛选")
                    
                    if filter_category != "全部":
                        default_selection = [kb for kb in available_kbs if kb.startswith(filter_category)]
                
                selected_kbs = st.multiselect("选择检索范围（知识库）", available_kbs, default=default_selection)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            char_name = st.text_input("角色名称", placeholder="例如：孙悟空")
        with col2:
            char_style = st.selectbox("提示词风格", ["详细设定版", "简短对话版", "JSON格式"])
        with col3:
            retrieve_k = st.number_input("检索片段数", min_value=1, max_value=100, value=15, help="增加此数值可以读取更多原文内容，但会消耗更多 Token")

        extra_req = st.text_area("额外要求 (可选)", placeholder="例如：重点描述他的战斗经历，或者他和某人的关系...")

        if st.button("生成角色提示词", disabled=not (st.session_state.vector_db_ready and st.session_state.llm_client)):
            if not char_name:
                st.warning("请输入角色名称")
            else:
                with st.spinner(f"正在多角度检索关于 {char_name} 的信息..."):
                    # 1. RAG 多路检索 (Multi-Query Retrieval)
                    # 定义不同的检索角度，以提取更丰富的信息
                    queries = [
                        f"关于角色 {char_name} 的外貌描写、性格特征、身世背景",
                        f"{char_name} 的说话风格、口头禅、经典台词、语气",
                        f"{char_name} 的重要经历、关键剧情、人际关系、对其他人的态度"
                    ]
                    if extra_req:
                        queries.append(f"{char_name} {extra_req}")
                    
                    all_retrieved_docs = []
                    seen_contents = set()
                    
                    # 执行多次检索
                    for q in queries:
                        docs = st.session_state.rag_engine.query(q, k=retrieve_k, collection_names=selected_kbs)
                        for doc in docs:
                            if doc.page_content not in seen_contents:
                                seen_contents.add(doc.page_content)
                                all_retrieved_docs.append(doc)
                    
                    # 截取用户指定的数量 (如果多路检索结果太多)
                    # 这里的逻辑是：优先保留前面的结果（通常相关性更高），但因为我们是多路合并，
                    # 简单的截断可能不够完美，但对于 RAG 来说，去重后的并集通常是最好的。
                    # 如果数量实在太多超过 retrieve_k * 2，可以适当截断，防止 Token 爆炸
                    if len(all_retrieved_docs) > retrieve_k:
                         # 这里我们稍微放宽一点，允许比用户设定的多一点，因为是多路合并的
                         all_retrieved_docs = all_retrieved_docs[:retrieve_k]

                    context_text = "\n\n".join([doc.page_content for doc in all_retrieved_docs])
                    
                    # 显示检索到的内容 (用于调试/确认)
                    with st.expander(f"查看检索到的原文片段 (共 {len(all_retrieved_docs)} 个片段)"):
                        st.info("已启用多角度混合检索（外貌性格 + 语言风格 + 经历关系 + 额外要求）")
                        for i, doc in enumerate(all_retrieved_docs):
                            st.markdown(f"**片段 {i+1}** (Source: {doc.metadata.get('source', 'unknown')}):")
                            # 显示完整内容，不再截断
                            st.text(doc.page_content)
                            st.divider()

                    # 2. 构建 Prompt (第一阶段：生成)
                    gen_prompt = f"""你是一个专业的角色设定专家。请根据提供的原文片段，为角色【{char_name}】撰写一份高级的角色扮演 System Prompt。

【任务要求】
1. **Prompt结构**：请使用动态Prompt结构，包含以下模块：
   - [角色详情]：姓名、年龄、身份等。
   - [性格特质]：深层性格、行事逻辑、优缺点。
   - [语言风格]：口癖、语气、常用词、句式特点。
   - [经历背景]：关键身世、重要剧情节点。
   - [人际关系]：与关键人物的关系及态度。
2. **对话生成**：请生成一段包含 **5个来回** 的对话示例（User与{char_name}的互动）。对话内容需紧扣剧情逻辑，展现角色的语气和性格。
3. **行文风格提取**：**必须**在所有输出的最后，单独列出一个章节叫“【提取的原文本行文风格】”，描述原文的描写手法、修辞风格和氛围感。

【原文片段】
{context_text}

【用户额外要求】
{extra_req}

请直接输出结果。
"""
                    
                    # 第一阶段调用
                    first_stage_response = ""
                    with st.status("正在进行深度生成...", expanded=True) as status:
                        st.write("📝 正在生成初始角色设定与对话...")
                        messages_gen = [{"role": "user", "content": gen_prompt}]
                        stream_gen = st.session_state.llm_client.chat(messages_gen, model=selected_model, stream=True)
                        
                        gen_placeholder = st.empty()
                        if isinstance(stream_gen, str):
                            st.error(stream_gen)
                            st.stop()
                        
                        for chunk in stream_gen:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                first_stage_response += content
                                gen_placeholder.markdown(first_stage_response + "▌")
                        gen_placeholder.markdown(first_stage_response)
                        
                        # 3. 构建 Prompt (第二阶段：判别与修正)
                        st.write("⚖️ 正在进行剧情逻辑与人设校验...")
                        judge_prompt = f"""你是一个剧情逻辑审核员。请评估以下生成的角色Prompt和对话是否符合原文的剧情逻辑和人设。

【原文片段】
{context_text}

【待评估生成的设定】
{first_stage_response}

【审核要求】
1. **判断标准**：重点判断是否符合“剧情逻辑”和“人设还原度”。**削弱逻辑判断**，不要过分纠结严密的现实逻辑，只要符合故事内部的剧情逻辑即可。
2. **输出处理**：
   - 如果内容合格，请直接输出原内容。
   - 如果有偏差（如OOC、语气不对、剧情冲突），请修正并输出优化后的完整版本。
3. **保留项**：确保输出的最后依然包含“【提取的原文本行文风格】”。

请输出最终确定的版本。
"""
                        messages_judge = [{"role": "user", "content": judge_prompt}]
                        stream_judge = st.session_state.llm_client.chat(messages_judge, model=selected_model, stream=True)
                        
                        final_response = ""
                        # Clear previous placeholder to show final result cleanly
                        gen_placeholder.empty() 
                        final_placeholder = st.empty()
                        
                        if isinstance(stream_judge, str):
                            st.error(stream_judge)
                        else:
                            for chunk in stream_judge:
                                if chunk.choices[0].delta.content:
                                    content = chunk.choices[0].delta.content
                                    final_response += content
                                    final_placeholder.markdown(final_response + "▌")
                            final_placeholder.markdown(final_response)
                        
                        status.update(label="生成完成", state="complete", expanded=False)
                        
                        # 重置对话历史，存入最终结果
                        st.session_state.gen_messages = [
                            {"role": "user", "content": gen_prompt}, # 保存初始请求
                            {"role": "assistant", "content": final_response}
                        ]
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

    # Tab 3: QQ角色生成
    with tab3:
        st.markdown("### 🤖 QQ聊天角色Prompt生成")
        st.markdown("通过与AI对话来创建适合QQ聊天的角色设定")

        # 第一步：对话收集
        st.subheader("📝 第一步：与AI自由对话")
        st.markdown("与AI进行自由对话，帮助AI了解你想要的角色特点")

        # 显示对话历史
        for msg in st.session_state.qq_dialogue_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 对话输入
        if qq_prompt := st.chat_input("输入你的问题或对话内容...", key="qq_dialogue"):
            if not st.session_state.llm_client:
                st.error("请先配置API Key")
            else:
                # 添加用户消息
                st.session_state.qq_dialogue_messages.append({"role": "user", "content": qq_prompt})
                with st.chat_message("user"):
                    st.markdown(qq_prompt)

                # 生成AI回复
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    # 构建消息
                    messages_payload = [{"role": "system", "content": "你是一个友好的AI助手，请与用户进行自然、流畅的对话。通过对话了解用户的喜好、性格特点，为后续生成QQ聊天角色设定做准备。"}]
                    for m in st.session_state.qq_dialogue_messages[-10:]:  # 只保留最近10轮对话
                        messages_payload.append(m)

                    stream = st.session_state.llm_client.chat(messages_payload, model=selected_model, stream=True)

                    if isinstance(stream, str):
                        st.error(stream)
                    else:
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                response_placeholder.markdown(full_response + "▌")

                        response_placeholder.markdown(full_response)
                        st.session_state.qq_dialogue_messages.append({"role": "assistant", "content": full_response})

        # 第二步：生成QQ角色Prompt
        st.divider()
        st.subheader("🎯 第二步：生成QQ聊天Prompt")

        col_gen1, col_gen2 = st.columns([1, 1])
        with col_gen1:
            if st.button("📝 生成角色Prompt", disabled=not st.session_state.llm_client or not st.session_state.qq_dialogue_messages):
                if not st.session_state.qq_dialogue_messages:
                    st.warning("请先进行一些对话来帮助AI了解角色特点")
                else:
                    with st.spinner("正在分析对话并生成角色设定..."):
                        # 构建生成prompt
                        dialogue_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.qq_dialogue_messages])

                        gen_prompt = f"""基于以下对话内容，请为QQ聊天生成一个角色Prompt。

【对话记录】
{dialogue_text}

请生成一个完整的QQ聊天角色Prompt，包含以下部分：

1. 【人设基本信息】：角色的姓名、年龄、性别、职业等基本信息
2. 【人物背景】：角色的身世背景、经历、性格特点等
3. 【聊天对话要求】：角色的说话风格、语气、常用表情、聊天习惯等
4. 【对话示例】：请提供5个具体的对话示例，格式如下：
   用户：消息内容
   角色：回复内容

请确保对话示例贴合角色的性格和说话风格。
"""

                        messages_gen = [{"role": "user", "content": gen_prompt}]
                        response = st.session_state.llm_client.chat(messages_gen, model=selected_model, stream=False)

                        if isinstance(response, str):
                            st.error(response)
                        else:
                            full_response = response.choices[0].message.content

                            # 解析生成的prompt
                            try:
                                # 简单的文本解析
                                lines = full_response.split('\n')
                                character_info = ""
                                background = ""
                                chat_requirements = ""
                                examples = []

                                current_section = ""
                                example_lines = []

                                for line in lines:
                                    line = line.strip()
                                    if line.startswith("【人设基本信息】"):
                                        current_section = "character_info"
                                        character_info = line.replace("【人设基本信息】", "").strip()
                                    elif line.startswith("【人物背景】"):
                                        current_section = "background"
                                        background = line.replace("【人物背景】", "").strip()
                                    elif line.startswith("【聊天对话要求】"):
                                        current_section = "chat_requirements"
                                        chat_requirements = line.replace("【聊天对话要求】", "").strip()
                                    elif line.startswith("【对话示例】"):
                                        current_section = "examples"
                                    elif current_section == "character_info" and not line.startswith("【"):
                                        character_info += "\n" + line
                                    elif current_section == "background" and not line.startswith("【"):
                                        background += "\n" + line
                                    elif current_section == "chat_requirements" and not line.startswith("【"):
                                        chat_requirements += "\n" + line
                                    elif current_section == "examples":
                                        if "：" in line:
                                            parts = line.split("：", 1)
                                            if len(parts) == 2:
                                                role = parts[0].strip()
                                                content = parts[1].strip()
                                                if role == "用户" or role == "角色":
                                                    example_lines.append({"role": "user" if role == "用户" else "character", "content": content})
                                                if len(example_lines) >= 2:
                                                    examples.append({
                                                        "user": example_lines[-2]["content"] if example_lines[-2]["role"] == "user" else "",
                                                        "character": example_lines[-1]["content"] if example_lines[-1]["role"] == "character" else ""
                                                    })
                                                    example_lines = []

                                # 更新session state
                                st.session_state.qq_prompt_data = {
                                    "character_info": character_info.strip(),
                                    "background": background.strip(),
                                    "chat_requirements": chat_requirements.strip(),
                                    "dialogue_examples": examples[:5] if examples else st.session_state.qq_prompt_data["dialogue_examples"]
                                }

                                st.success("角色Prompt生成完成！")
                                st.rerun()

                            except Exception as e:
                                st.error(f"解析生成结果失败: {e}")
                                st.text_area("生成的完整内容", full_response, height=300)

        with col_gen2:
            if st.button("🗑️ 清空对话", disabled=not st.session_state.qq_dialogue_messages):
                st.session_state.qq_dialogue_messages = []
                st.success("对话已清空")
                st.rerun()

        # 第三步：编辑和调整
        if st.session_state.qq_prompt_data["character_info"] or st.session_state.qq_prompt_data["background"]:
            st.divider()
            st.subheader("✏️ 第三步：编辑和调整")

            # 显示当前Prompt
            with st.expander("📋 当前角色Prompt", expanded=True):
                st.markdown("**人设基本信息：**")
                st.text_area("人设基本信息", st.session_state.qq_prompt_data["character_info"], height=100, key="edit_character_info", on_change=lambda: update_qq_prompt_data())

                st.markdown("**人物背景：**")
                st.text_area("人物背景", st.session_state.qq_prompt_data["background"], height=150, key="edit_background", on_change=lambda: update_qq_prompt_data())

                st.markdown("**聊天对话要求：**")
                st.text_area("聊天对话要求", st.session_state.qq_prompt_data["chat_requirements"], height=150, key="edit_chat_requirements", on_change=lambda: update_qq_prompt_data())

                st.markdown("**对话示例：**")
                for i, example in enumerate(st.session_state.qq_prompt_data["dialogue_examples"]):
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        st.text_input(f"用户消息 {i+1}", example["user"], key=f"user_msg_{i}", on_change=lambda idx=i: update_example(idx, "user"))
                    with col_e2:
                        st.text_input(f"角色回复 {i+1}", example["character"], key=f"char_msg_{i}", on_change=lambda idx=i: update_example(idx, "character"))

            # 调整按钮
            if st.button("🔄 根据示例调整对话要求", disabled=not st.session_state.llm_client):
                with st.spinner("正在根据对话示例调整对话要求..."):
                    examples_text = "\n".join([f"用户：{ex['user']}\n角色：{ex['character']}" for ex in st.session_state.qq_prompt_data["dialogue_examples"]])

                    adjust_prompt = f"""基于以下对话示例，请优化聊天对话要求部分：

【当前对话要求】
{st.session_state.qq_prompt_data["chat_requirements"]}

【对话示例】
{examples_text}

请根据这些对话示例，重新优化"聊天对话要求"部分，使其更准确地反映角色的说话风格、语气和聊天习惯。
只输出优化后的对话要求内容，不要包含其他说明。
"""

                    messages_adjust = [{"role": "user", "content": adjust_prompt}]
                    response = st.session_state.llm_client.chat(messages_adjust, model=selected_model, stream=False)

                    if isinstance(response, str):
                        st.error(response)
                    else:
                        new_requirements = response.choices[0].message.content.strip()
                        st.session_state.qq_prompt_data["chat_requirements"] = new_requirements
                        st.success("对话要求已更新！")
                        st.rerun()

            # 保存按钮
            if st.button("💾 保存到历史记录"):
                prompt_content = f"""【人设基本信息】
{st.session_state.qq_prompt_data["character_info"]}

【人物背景】
{st.session_state.qq_prompt_data["background"]}

【聊天对话要求】
{st.session_state.qq_prompt_data["chat_requirements"]}

【对话示例】
""" + "\n".join([f"用户：{ex['user']}\n角色：{ex['character']}\n" for ex in st.session_state.qq_prompt_data["dialogue_examples"]])

                char_name = "QQ角色"
                if st.session_state.qq_prompt_data["character_info"]:
                    # 尝试提取角色名称
                    first_line = st.session_state.qq_prompt_data["character_info"].split('\n')[0]
                    if "：" in first_line:
                        char_name = first_line.split("：")[1].strip()

                save_history_item(char_name, prompt_content)
                st.success("已保存到历史记录！")

    # Tab 4: 历史记录
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
