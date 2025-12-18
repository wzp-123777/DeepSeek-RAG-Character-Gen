# DeepSeek RAG 角色提示词生成器

这是一个基于 Streamlit 的 Web 应用，专为角色扮演（Roleplay）爱好者和创作者设计。它允许你上传小说、剧本等大文本文件，利用 RAG（检索增强生成）技术，结合 DeepSeek、Kimi 等大模型，自动从原文中提取角色的外貌、性格、经历等关键信息，并生成高质量的 System Prompt。

## ✨ 核心功能

*   **🧠 多模型支持**：
    *   **DeepSeek 官方**：支持 `deepseek-chat`, `deepseek-reasoner`。
    *   **硅基流动 (SiliconFlow)**：支持 `DeepSeek-V3`, `DeepSeek-R1`, `DeepSeek-V3.2`, `Kimi-K2-Thinking` 等多种高性能模型。
*   **📚 强大的知识库管理**：
    *   **多知识库分组**：支持将不同来源的文件存入不同的知识库（如“红楼梦”、“三国演义”）。
    *   **灵活检索**：生成 Prompt 时可自由勾选一个或多个知识库作为检索源。
    *   **可视化管理**：侧边栏实时显示已收录的文件列表及片段数量。
    *   **双模 Embedding**：支持 **本地模型** (HuggingFace, 免费, 隐私好) 和 **云端 API** (SiliconFlow, 高性能, 无需显卡)。
*   **💬 交互式 Prompt 优化**：
    *   生成初始 Prompt 后，可以通过对话框与"专家 AI"进行多轮沟通。
    *   支持提出修改意见（如"让性格更傲娇一点"），模型会实时调整 Prompt。
    *   支持查看完整的 RAG 检索原文片段，确保信息准确。
*   **🤖 QQ角色生成**：
    *   **自由对话收集**：与AI进行自由对话，帮助AI了解你想要的角色特点。
    *   **智能Prompt生成**：基于对话内容自动生成包含人设、背景、对话要求和示例的QQ聊天Prompt。
    *   **可编辑示例**：提供5个对话示例，支持用户手动编辑和调整。
    *   **智能调整**：修改对话示例后，AI会自动优化聊天对话要求，确保一致性。
*   **💾 历史记录与配置记忆**：
    *   **自动保存配置**：API Key、模型选择、知识库设置等自动保存，下次打开即用。
    *   **Prompt 历史**：一键保存满意的 Prompt，随时在“历史记录”页查看或删除。
*   **🚀 极简启动**：提供 Windows 一键启动脚本，无需懂代码也能轻松使用。

## 🚀 部署指南

### 📋 第一次部署清单

在开始使用本项目前，请确保你已准备好以下内容：

#### 1. 环境准备
- ✅ **Python 3.8+** 已安装
- ✅ **Git** 已安装（用于克隆项目）
- ✅ **稳定的网络连接**（用于下载依赖和API调用）

#### 2. API服务准备
- ✅ **DeepSeek API** 或 **SiliconFlow API** 账户
- ✅ **有效的API Key**（DeepSeek或SiliconFlow）
- ✅ **账户余额充足**（API调用会产生费用）

#### 3. 硬件要求
- ✅ **至少4GB RAM**（推荐8GB+）
- ✅ **网络连接稳定**（用于API调用和模型下载）

### 🛠️ 详细部署步骤

#### 步骤1：获取项目代码
```bash
# 克隆项目到本地
git clone https://github.com/wzp-123777/DeepSeek-RAG-Character-Gen.git

# 进入项目目录
cd DeepSeek-RAG-Character-Gen
```

#### 步骤2：创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate
```

#### 步骤3：安装依赖
```bash
# 安装所有必需的Python包
pip install -r requirements.txt
```

#### 步骤4：获取API Key
1. **DeepSeek API**：
   - 访问 [DeepSeek 官网](https://platform.deepseek.com/)
   - 注册账户并获取 API Key

2. **SiliconFlow API**（推荐）：
   - 访问 [SiliconFlow 官网](https://siliconflow.cn/)
   - 注册账户并获取 API Key
   - 推荐充值少量金额（¥10-20）用于测试

#### 步骤5：首次运行测试
```bash
# 运行应用
streamlit run app.py
```

#### 步骤6：配置API
1. 在浏览器中打开 `http://localhost:8501`
2. 在左侧边栏选择 LLM 提供商（DeepSeek 或 SiliconFlow）
3. 输入你的 API Key
4. 选择合适的对话模型
5. 测试连接是否成功

#### 步骤7：构建知识库（可选）
如果要使用RAG功能：
1. 上传小说、剧本等文本文件
2. 选择Embedding模式（推荐使用SiliconFlow的云端API）
3. 点击"构建/更新知识库"

#### 步骤8：开始使用
- 🎭 **角色提示词生成**：基于知识库生成角色设定
- 🤖 **QQ角色生成**：通过对话创建QQ聊天角色
- 💬 **自由对话**：与AI进行一般对话

### 🐛 常见问题解决

#### 问题1：依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 问题2：Streamlit启动失败
```bash
# 检查端口是否被占用
netstat -ano | findstr :8501

# 使用不同端口启动
streamlit run app.py --server.port 8502
```

#### 问题3：API连接失败
- 检查API Key是否正确
- 确认账户余额充足
- 检查网络连接是否正常
- 尝试更换API提供商

#### 问题4：内存不足
- 关闭其他程序
- 使用云端Embedding模式而非本地模式
- 减少同时处理的文件数量

## 🛠️ 安装与运行

### 方法一：Windows 一键启动（推荐）

1.  双击运行目录下的 **`install_dependencies.bat`** （首次运行或更新时使用）。
2.  双击运行 **`run.bat`** 即可启动应用。

### 方法二：命令行启动

1.  **环境要求**：Python 3.8+
2.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```
3.  **运行应用**：
    ```bash
    streamlit run app.py
    ```

应用启动后，浏览器会自动打开 `http://localhost:8501`。

## 📖 使用指南

### 1. 配置 API
*   在左侧侧边栏选择 **LLM 提供商**（DeepSeek 或 SiliconFlow）。
*   输入对应的 **API Key**。
*   选择你喜欢的 **对话模型**（推荐 DeepSeek-V3 或 Kimi-K2）。

### 2. 构建知识库
*   **Embedding 模式**：
    *   **本地**：默认使用 `sentence-transformers`，首次运行会自动下载模型。
    *   **云端 API**：推荐使用 `Qwen/Qwen3-Embedding-8B` (SiliconFlow)，速度快且效果好。
*   **上传文件**：支持 `.txt`, `.pdf`, `.docx` 格式。
*   **设置名称**：输入“目标知识库名称”（如 `book_1`），点击构建。

### 3. 生成角色 Prompt
1.  切换到 **“🎭 角色提示词生成”** 标签页。
2.  **选择检索范围**：勾选你需要参考的知识库。
3.  输入 **角色名称**（如“林黛玉”）和 **提示词风格**。
4.  （可选）调整 **检索片段数**，获取更多上下文。
5.  点击 **“生成角色提示词”**。

### 4. 优化与保存
*   如果不满意，在下方的对话框输入修改意见，AI 会重新生成。
*   满意后，点击 **"💾 保存当前 Prompt 到历史记录"**。
*   在 **"📜 历史记录"** 标签页可以找回所有保存的内容。

### 5. QQ角色生成
1.  切换到 **"🤖 QQ角色生成"** 标签页。
2.  **第一步**：与AI进行自由对话，描述你想要的角色特点、性格、说话风格等。
3.  **第二步**：点击 **"📝 生成角色Prompt"**，AI会基于对话内容生成完整的QQ聊天Prompt。
4.  **第三步**：在展开的编辑区域中，可以修改人设信息、人物背景、对话要求。
5.  还可以编辑5个对话示例，修改后点击 **"🔄 根据示例调整对话要求"** 让AI优化对话风格。
6.  满意后点击 **"💾 保存到历史记录"**。

## ⚠️ 注意事项

*   **API 费用**：使用 DeepSeek 或 SiliconFlow API 会产生相应的 Token 费用，请确保账户余额充足。
*   **隐私安全**：你的 API Key 和知识库数据仅保存在本地（`user_config.json`, `chroma_db/`），**不会**被上传到任何第三方服务器（除了发送给 LLM 进行生成的必要片段）。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进这个项目！
