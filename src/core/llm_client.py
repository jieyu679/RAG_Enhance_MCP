import openai
from typing import List, Dict, Optional
import time
import json

class Qwen3Client:
    """Qwen3客户端封装"""
    
    def __init__(self, base_url: str, model_name: str, temperature: float = 0.7, max_tokens: int = 2048):
        self.client = openai.OpenAI(
            api_key="EMPTY",  # 本地部署不需要API key
            base_url=base_url
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> Dict:
        """调用Qwen3聊天接口"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens": response.usage.total_tokens,
                "success": True
            }
        except Exception as e:
            return {
                "content": "",
                "tokens": 0,
                "success": False,
                "error": str(e)
            }
    
    def generate_json(self, messages: List[Dict[str, str]]) -> Optional[Dict]:
        """生成JSON格式的响应"""
        response = self.chat(messages)
        if not response["success"]:
            return None
        
        try:
            # 尝试提取JSON（可能被包裹在```json```中）
            content = response["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始输出: {response['content']}")
            return None