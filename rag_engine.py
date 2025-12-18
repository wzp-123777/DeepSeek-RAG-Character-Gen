import os
import shutil
import chromadb
import warnings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# 忽略 tiktoken 的模型警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*model not found. Using cl100k_base encoding.*")

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class RAGEngine:
    def __init__(self, persist_directory="./chroma_db", embedding_type="local", model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=None, base_url=None):
        self.persist_directory = persist_directory
        self.embedding_model_name = model_name
        self.embeddings = None
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        print(f"正在初始化 Embedding 模型: {model_name} ({embedding_type}) ...")
        try:
            if embedding_type == "local":
                # 注意：这会下载模型到本地缓存
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            elif embedding_type == "api":
                if not api_key or not base_url:
                    raise ValueError("使用 API Embedding 需要提供 API Key 和 Base URL")
                self.embeddings = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=api_key,
                    openai_api_base=base_url,
                    chunk_size=16, # 减小批处理大小，防止 API 500 错误
                    tiktoken_model_name="cl100k_base" # 显式指定编码，避免警告
                )
                
                # 立即测试连接，确保 API Key 和模型名称有效
                # 注意：这会消耗极少量的 token，但能保证后续流程顺畅
                print(f"正在测试 API 连接 ({model_name})...")
                try:
                    self.embeddings.embed_query("test")
                    print("API 连接测试成功！")
                except Exception as api_error:
                    raise ValueError(f"API 连接失败: {api_error}")
                    
        except Exception as e:
            print(f"加载 Embedding 模型失败: {e}")
            raise e

    def load_documents(self, file_paths):
        """
        加载并切分文档
        """
        documents = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            loader = None
            
            try:
                if ext == ".txt":
                    # 针对大文件优化：优先尝试常见编码，避免自动检测超时
                    success = False
                    for enc in ["utf-8", "gb18030", "gbk"]:
                        try:
                            loader = TextLoader(file_path, encoding=enc)
                            docs = loader.load()
                            documents.extend(docs)
                            success = True
                            break
                        except Exception:
                            continue
                    
                    if not success:
                        # 最后尝试自动检测
                        loader = TextLoader(file_path, autodetect_encoding=True)
                        docs = loader.load()
                        documents.extend(docs)
                        
                elif ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    if loader:
                        docs = loader.load()
                        documents.extend(docs)
                elif ext in [".docx", ".doc"]:
                    loader = Docx2txtLoader(file_path)
                    if loader:
                        docs = loader.load()
                        documents.extend(docs)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                return f"Error loading {file_path}: {str(e)}"
        
        if not documents:
            return "没有成功加载任何文档。"

        # 切分文档
        split_docs = self.text_splitter.split_documents(documents)
        return split_docs

    def load_urls(self, urls, fetch_links=False):
        """
        加载并切分网页内容
        """
        target_urls = []
        if fetch_links:
            # 如果是目录页，先抓取链接
            for url in urls:
                try:
                    print(f"正在分析目录页: {url}")
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                    response = requests.get(url, headers=headers, timeout=10)
                    response.encoding = response.apparent_encoding # Fix encoding
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 提取所有链接
                    links = soup.find_all('a')
                    chapter_links = []
                    
                    base_domain = urlparse(url).netloc
                    
                    for link in links:
                        href = link.get('href')
                        text = link.get_text().strip()
                        
                        if not href or href.startswith('javascript') or href.startswith('#'):
                            continue
                            
                        full_url = urljoin(url, href)
                        
                        # 简单的过滤规则：
                        # 1. 必须是同域名
                        # 2. 文本长度适中 (章节名通常不会太长)
                        if urlparse(full_url).netloc == base_domain:
                            # 关键词过滤 (可选，但为了通用性先不做太死)
                            if 2 < len(text) < 50: 
                                chapter_links.append(full_url)
                    
                    # 去重
                    chapter_links = list(set(chapter_links))
                    print(f"找到 {len(chapter_links)} 个潜在章节链接")
                    target_urls.extend(chapter_links)
                    
                except Exception as e:
                    print(f"解析目录页 {url} 失败: {e}")
                    # 如果解析失败，至少把目录页本身加进去
                    target_urls.append(url)
        else:
            target_urls = urls

        if not target_urls:
            return "没有找到有效的网页链接。"

        documents = []
        try:
            print(f"准备抓取 {len(target_urls)} 个页面...")
            # 使用 WebBaseLoader 批量抓取
            # 注意：如果链接非常多，可能会很慢
            loader = WebBaseLoader(target_urls)
            loader.requests_kwargs = {'headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}}
            loader.requests_per_second = 5 # 限制并发
            
            documents = loader.load()
        except Exception as e:
            print(f"处理网页时出错: {e}")
            return f"Error loading URLs: {str(e)}"
        
        if not documents:
            return "没有成功加载任何网页内容。"

        # 切分文档
        split_docs = self.text_splitter.split_documents(documents)
        return split_docs

    def build_vector_store(self, documents, collection_name="character_data"):
        """
        建立向量数据库 (带速率限制保护)
        """
        if not documents:
            return "没有文档可用于构建向量库。"

        try:
            import time
            
            # 初始化 vector_store
            vector_store = Chroma(
                embedding_function=self.embeddings,
                client=self.client,
                collection_name=collection_name
            )
            
            # 分批处理，避免触发 API 速率限制 (429)
            batch_size = 10  # 每次处理 10 个片段
            total_docs = len(documents)
            
            print(f"开始构建向量库，共 {total_docs} 个片段，分批处理中...")
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i : i + batch_size]
                try:
                    vector_store.add_documents(batch)
                    # 简单的速率限制：每批处理完暂停 0.5 秒
                    # 如果是 API 模式，这个暂停很重要
                    time.sleep(0.5) 
                except Exception as batch_error:
                    print(f"批次 {i} 处理失败: {batch_error}")
                    # 遇到 429 错误时，尝试等待更久后重试一次
                    if "429" in str(batch_error):
                        print("触发速率限制，等待 5 秒后重试...")
                        time.sleep(5)
                        vector_store.add_documents(batch)
            
            return f"成功构建知识库 '{collection_name}'，包含 {len(documents)} 个片段。"
        except Exception as e:
            return f"构建向量库失败: {str(e)}"

    def query(self, query_text, k=5, collection_names=None):
        """
        检索相关文档，支持多知识库
        """
        if collection_names is None:
            collection_names = ["character_data"]
        
        if isinstance(collection_names, str):
            collection_names = [collection_names]

        all_results = []
        
        for col_name in collection_names:
            try:
                vector_store = Chroma(
                    client=self.client,
                    embedding_function=self.embeddings,
                    collection_name=col_name
                )
                # 简单的检索
                results = vector_store.similarity_search(query_text, k=k)
                all_results.extend(results)
            except Exception as e:
                print(f"检索知识库 {col_name} 失败: {e}")
                continue

        # 如果有多个知识库，结果可能会很多，这里简单去重（基于内容）并截取
        # 注意：简单的 extend 会导致结果排序混乱（不同库的分数可能不可比），
        # 但 Chroma 的 similarity_search 返回的是 Document 对象，不带分数。
        # 如果需要更精确的排序，应该用 similarity_search_with_score
        
        # 简单去重
        seen_content = set()
        unique_results = []
        for doc in all_results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_results.append(doc)
        
        return unique_results[:k] # 返回前 k 个（这里其实不太准确，因为没有全局排序）

    def delete_collection(self, collection_name):
        """
        删除指定的知识库
        """
        try:
            self.client.delete_collection(collection_name)
            return True, f"已删除知识库: {collection_name}"
        except Exception as e:
            return False, f"删除失败: {str(e)}"

    def clear_database(self):
        if os.path.exists(self.persist_directory):
            try:
                # 关闭 client 连接可能比较麻烦，直接删文件最暴力有效
                # 但由于 client 保持着连接，可能无法删除。
                # 尝试使用 client.reset() 如果允许，或者删除所有 collections
                collections = self.client.list_collections()
                for col in collections:
                    self.client.delete_collection(col.name)
                return True
            except Exception as e:
                print(f"清理数据库失败: {e}")
                return False
        return True

    def get_available_collections(self):
        """获取所有可用的知识库名称"""
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except:
            return []

    def get_documents_summary(self):
        """
        获取知识库中的文档摘要（文件名列表和片段数），按知识库分组
        """
        summary = {}
        try:
            collections = self.client.list_collections()
            for col in collections:
                try:
                    # 获取 metadata
                    data = col.get()
                    count = len(data['ids']) if data['ids'] else 0
                    metadatas = data['metadatas']
                    files = set()
                    if metadatas:
                        for meta in metadatas:
                            if meta and 'source' in meta:
                                files.add(os.path.basename(meta['source']))
                    
                    summary[col.name] = {
                        "files": list(files),
                        "count": count
                    }
                except:
                    continue
        except Exception as e:
            print(f"获取摘要失败: {e}")
        
        return summary
