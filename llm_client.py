import os
from openai import OpenAI

class LLMClient:
    def __init__(self, provider="deepseek", api_key=None):
        self.provider = provider
        self.api_key = api_key
        self.client = None
        self.model_name = ""
        
        self._setup_client()

    def _setup_client(self):
        if not self.api_key:
            # 尝试从环境变量获取
            if self.provider == "deepseek":
                self.api_key = os.getenv("DEEPSEEK_API_KEY")
            elif self.provider == "siliconflow":
                self.api_key = os.getenv("SILICONFLOW_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"未提供 {self.provider} 的 API Key")

        if self.provider == "deepseek":
            self.base_url = "https://api.deepseek.com"
            self.model_name = "deepseek-chat" # 或者 deepseek-reasoner
        elif self.provider == "siliconflow":
            self.base_url = "https://api.siliconflow.cn/v1"
            # 硅基流动支持多个模型，这里默认设为一个常用的 deepseek 模型，实际调用时可覆盖
            self.model_name = "deepseek-ai/DeepSeek-V3" 
        else:
            raise ValueError("不支持的提供商")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages, model=None, temperature=0.7, stream=True):
        """
        发送对话请求
        """
        use_model = model if model else self.model_name
        
        try:
            response = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def get_available_models(self):
        if self.provider == "deepseek":
            return ["deepseek-chat", "deepseek-reasoner"]
        elif self.provider == "siliconflow":
            # 硅基流动模型列表较多，这里列出几个常用的 DeepSeek 相关模型
            return [
                "deepseek-ai/DeepSeek-V3",
                "deepseek-ai/DeepSeek-R1",
                "deepseek-ai/DeepSeek-V2.5",
                "Pro/deepseek-ai/DeepSeek-V3.2", # 新增 V3.2 Pro
                "deepseek-ai/DeepSeek-V3.2-Exp", # 新增 V3.2 Exp
                "moonshotai/Kimi-K2-Thinking", 
            ]
        return []
