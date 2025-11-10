from typing import List, Dict, Optional
from src.core.data_structures import Task, MCPMetadata, ExecutionResult
from src.core.llm_client import Qwen3Client
from collections import defaultdict
import uuid

class RawMCPPool:
    """原始MCP池（存储成功的任务执行）"""
    
    def __init__(self):
        self.successful_tasks = []
    
    def add(self, task: Task, mcp: MCPMetadata, result: ExecutionResult):
        """添加成功的任务"""
        self.successful_tasks.append({
            "task": task,
            "mcp": mcp,
            "result": result
        })
    
    def find_similar_clusters(self, threshold: int = 3) -> List[List[dict]]:
        """找到相似任务簇（简化版：基于MCP ID分组）"""
        # 按MCP ID分组
        mcp_groups = defaultdict(list)
        for item in self.successful_tasks:
            mcp_groups[item["mcp"].id].append(item)
        
        # 筛选出数量>=threshold的组
        clusters = []
        for mcp_id, items in mcp_groups.items():
            if len(items) >= threshold:
                clusters.append(items)
        
        return clusters

class MCPAbstractionPipeline:
    """MCP抽象流水线"""
    
    def __init__(self, llm_client: Qwen3Client, success_rate_threshold: float = 0.8):
        self.llm = llm_client
        self.success_rate_threshold = success_rate_threshold
    
    def abstract(self, similar_tasks: List[dict]) -> Optional[MCPMetadata]:
        """
        从相似任务中抽象出新MCP
        
        Args:
            similar_tasks: 相似任务列表（包含task, mcp, result）
        
        Returns:
            新的MCP或None
        """
        if len(similar_tasks) < 3:
            return None
        
        # 计算成功率
        success_count = sum(1 for item in similar_tasks if item["result"].success)
        success_rate = success_count / len(similar_tasks)
        
        if success_rate < self.success_rate_threshold:
            print(f"⚠️ 成功率 {success_rate:.2f} 低于阈值 {self.success_rate_threshold}，跳过抽象")
            return None
        
        # 阶段1: 参数泛化
        generalized = self._generalize_parameters(similar_tasks)
        if generalized is None:
            return None
        
        # 阶段2: 生成文档
        documented = self._generate_documentation(generalized, similar_tasks)
        
        return documented
    
    def _generalize_parameters(self, similar_tasks: List[dict]) -> Optional[Dict]:
        """参数泛化（使用LLM）"""
        task_descriptions = "\n".join([
            f"- {item['task'].description}"
            for item in similar_tasks[:5]  # 最多5个示例
        ])
        
        prompt = f"""
以下是几个成功执行的相似任务：
{task_descriptions}

请提取出通用模式，将具体值替换为参数。

示例：
任务1: "查询用户张三的订单历史"
任务2: "查询用户李四的订单历史"
抽象: "查询用户{{user_name}}的订单历史"

输出JSON格式:
{{
  "template": "泛化后的任务描述模板",
  "parameters": {{
    "param_name": "参数描述"
  }},
  "capability": "这个MCP的能力描述（一句话）"
}}

只输出JSON，不要其他文字。
"""
        
        messages = [
            {"role": "system", "content": "你是MCP抽象专家。"},
            {"role": "user", "content": prompt}
        ]
        
        result = self.llm.generate_json(messages)
        return result
    
    def _generate_documentation(self, generalized: Dict, examples: List[dict]) -> MCPMetadata:
        """生成文档"""
        mcp_id = f"mcp_{uuid.uuid4().hex[:8]}"
        
        # 提取示例
        example_list = []
        for item in examples[:3]:
            example_list.append({
                "task": item["task"].description,
                "output": str(item["result"].output)[:200]  # 截断
            })
        
        mcp = MCPMetadata(
            id=mcp_id,
            name=generalized.get("template", "未命名MCP"),
            capability_description=generalized.get("capability", ""),
            parameters=generalized.get("parameters", {}),
            success_count=sum(1 for item in examples if item["result"].success),
            failure_count=len(examples) - sum(1 for item in examples if item["result"].success),
            examples=example_list
        )
        
        return mcp