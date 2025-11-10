from typing import List, Dict
from src.core.data_structures import Task, MCPMetadata, PlanningResult
from src.core.llm_client import Qwen3Client
import uuid

class Decomposer:
    """任务分解智能体"""
    
    def __init__(self, llm_client: Qwen3Client):
        self.llm = llm_client
    
    def decompose(self, query: str, mcp_box: List[MCPMetadata]) -> PlanningResult:
        """
        将用户查询分解为任务DAG
        
        Args:
            query: 用户查询
            mcp_box: 可用的MCP列表（只包含元数据）
        
        Returns:
            PlanningResult包含任务列表
        """
        # 构建提示词
        prompt = self._build_prompt(query, mcp_box)
        
        messages = [
            {"role": "system", "content": "你是一个专业的任务分解专家。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM
        result = self.llm.generate_json(messages)
        
        if result is None:
            return PlanningResult(tasks=[], is_valid=False, feedback="LLM返回格式错误")
        
        # 解析任务
        tasks = []
        for task_data in result.get("tasks", []):
            task = Task(
                id=task_data.get("id", f"task_{uuid.uuid4().hex[:8]}"),
                description=task_data["description"],
                candidate_mcps=task_data.get("candidate_mcps", []),
                depends_on=task_data.get("depends_on", [])
            )
            tasks.append(task)
        
        return PlanningResult(tasks=tasks, is_valid=True)
    
    def _build_prompt(self, query: str, mcp_box: List[MCPMetadata]) -> str:
        """构建分解提示词"""
        mcp_descriptions = "\n".join([
            f"- {mcp.id}: {mcp.name} - {mcp.capability_description}"
            for mcp in mcp_box
        ])
        
        prompt = f"""
你是一个任务分解专家。请将用户的复杂查询分解为可执行的子任务序列。

可用工具(MCP):
{mcp_descriptions}

用户查询: {query}

请将查询分解为3-7个子任务。每个任务需要:
1. 清晰的描述（一句话）
2. 推荐的MCP工具（从上述列表选择1-3个候选）
3. 依赖的前置任务（task_id列表，如果没有则为空）

分解原则:
- 每个任务应该是原子性的（不可再分）
- 任务之间的依赖关系要明确
- 选择的MCP要与任务描述匹配
- 考虑任务之间的并行执行可能性

输出严格的JSON格式:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "任务描述",
      "candidate_mcps": ["mcp_id_1", "mcp_id_2"],
      "depends_on": [],
      "rationale": "为什么需要这个任务的简短说明"
    }}
  ],
  "reasoning": "整体分解思路"
}}

请只输出JSON，不要有其他文字。
"""
        return prompt