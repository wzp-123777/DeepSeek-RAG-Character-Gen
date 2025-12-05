# DeepSeek RAG 角色提示词生成器

这是一个基于 Streamlit 的 Web 应用，允许用户上传大文本文件（如小说、剧本），利用 RAG（检索增强生成）技术，结合 DeepSeek 或 硅基流动（SiliconFlow）的大模型 API，自动生成高质量的角色提示词（System Prompt）。

## 功能特点

*   **多 API 支持**：支持 DeepSeek 官方 API 和 硅基流动 API。
*   **本地 RAG 引擎**：使用 LangChain 和 ChromaDB 构建本地知识库，数据不出域（除了发送给 LLM 的片段）。
*   **嵌入式模型**：支持加载 HuggingFace 的 Embedding 模型（默认 `sentence-transformers/all-MiniLM-L6-v2`，支持中文）。
*   **角色生成**：专门优化的 Prompt，用于从原文中提取角色的外貌、性格、经历等信息。
*   **自由对话**：支持基于知识库的自由问答。

## 安装步骤

1.  **环境要求**：Python 3.8+
2.  **安装依赖**：
    打开终端，进入项目目录，运行：
    ```bash
    pip install -r requirements.txt
    ```

## 运行方法

在终端中运行以下命令启动应用：

```bash
streamlit run app.py
```

应用启动后，浏览器会自动打开 `http://localhost:8501`。

## 使用指南

1.  **配置 API**：
    *   在左侧侧边栏选择 API 提供商（DeepSeek 或 SiliconFlow）。
    *   输入你的 API Key。
    *   选择模型（如 `deepseek-chat`）。

2.  **构建知识库**：
    *   在侧边栏上传你的大文本文件（支持 .txt, .pdf, .docx）。
    *   点击“构建/更新 知识库”按钮。
    *   *注意：首次运行时，程序会自动下载 Embedding 模型（约几百 MB），请耐心等待。*

3.  **生成角色提示词**：
    *   切换到“角色提示词生成”标签页。
    *   输入你想生成的角色名称（如“林黛玉”）。
    *   （可选）输入额外要求。
    *   点击“生成角色提示词”。
    *   程序会检索相关片段，并生成详细的 System Prompt。

4.  **自由对话**：
    *   切换到“自由对话”标签页，可以针对上传的文档进行提问。

## 注意事项

*   **Embedding 模型**：默认使用的是轻量级模型。如果你需要更好的中文效果，可以在代码或界面中将模型更改为 `shibing624/text2vec-base-chinese`。
*   **API 费用**：使用 DeepSeek 或 SiliconFlow API 会产生相应的 token 费用，请确保账户余额充足。
