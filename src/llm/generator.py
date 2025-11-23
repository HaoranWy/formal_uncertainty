import os
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 引入 OpenAI 异步客户端
from openai import AsyncOpenAI

# 引入 vLLM (仅在本地模式下需要，使用 try-import 防止 API 模式下报错)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class BaseModelInterface(ABC):
    """模型调用抽象基类"""
    @abstractmethod
    async def generate(self, messages: List[Dict], n: int = 1, temperature: float = 0.7, max_tokens: int = 1024) -> List[str]:
        pass

class APIInterface(BaseModelInterface):
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        # 确保 api_key 不为 None，否则 OpenAI 库可能会报错或尝试读取环境变量
        if not api_key:
            api_key = "EMPTY"
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate(self, messages: List[Dict], n: int = 1, temperature: float = 0.7, max_tokens: int = 1024) -> List[str]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            err_str = str(e).lower()
            
            # --- FIX START: 修复误判 "Unauthorized" 为 "n" 参数错误的问题 ---
            # 只有明确是 BadRequestError 且包含 parameter 相关描述才降级
            is_param_error = (
                "parameter" in err_str and 
                ("n" in err_str or "not supported" in err_str)
            )
            # 或者 DeepSeek 特有的错误信息
            if is_param_error:
                print(f"⚠️ API may not support n={n}, falling back to parallel requests...")
                return await self._generate_parallel(messages, n, temperature, max_tokens)
            
            # 如果是 401 (Unauthorized) 或其他错误，直接抛出，不要降级重试
            if "unauthorized" in err_str or "401" in err_str:
                raise e
            # --- FIX END ---
            
            raise e

    async def _generate_parallel(self, messages, n, temperature, max_tokens):
        """降级方案：并发发送 n 个请求"""
        tasks = []
        for _ in range(n):
            tasks.append(self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens
            ))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs = []
        for res in results:
            if not isinstance(res, Exception):
                outputs.append(res.choices[0].message.content)
            else:
                print(f"❌ One request failed: {res}")
                outputs.append("") # 失败占位
        return outputs

class LocalVLLMInterface(BaseModelInterface):
    """
    适用于本地显卡运行 vLLM (Offline Inference Mode)。
    注意：vLLM 通常是同步阻塞的，且独占 GPU。不要在 asyncio loop 中直接混用。
    """
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Please pip install vllm.")
        
        print(f"🚀 Loading vLLM model: {model_path}...")
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)

    async def generate(self, messages: List[Dict], n: int = 1, temperature: float = 0.7, max_tokens: int = 1024) -> List[str]:
        # vLLM 的离线推理通常接收 prompt string 或 tokens
        # 这里我们需要简单的将 messages 转换为 string (或者使用 tokenizer.apply_chat_template)
        # 为简化，假设 messages 已经处理好，或者我们直接拼接 prompt
        # 警告：vLLM 的 Chat 模板处理比较复杂，这里简化为取 user content
        # 实际复现建议使用 tokenizer 的 chat template
        
        prompt_text = ""
        for msg in messages:
            prompt_text += f"{msg['role']}: {msg['content']}\n"
        prompt_text += "Assistant:"

        sampling_params = SamplingParams(
            n=n, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        # vLLM 是同步的，为了适配 async 接口，这里其实是假异步
        # 在批量脚本中，我们会攒一批 prompt 一起发给 vLLM，而不是单条调
        outputs = self.llm.generate([prompt_text], sampling_params)
        return [output.text for output in outputs[0].outputs]

def get_model_interface(backend: str, **kwargs) -> BaseModelInterface:
    if backend == "api":
        return APIInterface(
            model_name=kwargs.get("model_name"),
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url")
        )
    elif backend == "vllm":
        return LocalVLLMInterface(
            model_path=kwargs.get("model_name"),
            tensor_parallel_size=kwargs.get("tp", 1)
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")