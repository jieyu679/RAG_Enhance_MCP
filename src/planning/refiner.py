from typing import List
from src.core.data_structures import Task, MCPMetadata, PlanningResult
from src.core.llm_client import Qwen3Client
import networkx as nx

class Refiner:
    """任务验证智能体"""
    
    def __init__(self, llm_client: Qwen3Client):
        self.llm = llm_client
    
    def validate(self, query: str, planning_result: PlanningResult, mcp_box: List[MCPMetadata]) -> PlanningResult:
        """
        验证任务分解的质量
        
        Args:
            query: 原始用户查询
            planning_result: Decomposer的输出
            mcp_box: 可用MCP列表
        
        Returns:
            更新后的PlanningResult（is_valid + feedback）
        """
        tasks = planning_result.tasks
        
        # 检查1: 结构性检查（快速，基于规则）
        structure_valid, structure_feedback = self._check_structure(tasks, mcp_box)
        if not structure_valid:
            planning_result.is_valid = False
            planning_result.feedback = structure_feedback
            return planning_result
        
        # 检查2: 语义检查（使用LLM）
        semantic_valid, semantic_feedback = self._check_semantics(query, tasks, mcp_box)
        if not semantic_valid:
            planning_result.is_valid = False
            planning_result.feedback = semantic_feedback
            return planning_result
        
        planning_result.is_valid = True
        planning_result.feedback = "验证通过"
        return planning_result
    
    def _check_structure(self, tasks: List[Task], mcp_box: List[MCPMetadata]) -> tuple[bool, str]:
        """结构性检查"""
        if len(tasks) == 0:
            return False, "任务列表为空"
        
        if len(tasks) > 10:
            return False, "任务数量过多（>10），请简化"
        
        # 检查MCP是否存在
        mcp_ids = {mcp.id for mcp in mcp_box}
        for task in tasks:
            if not task.candidate_mcps:
                return False, f"任务 {task.id} 没有候选MCP"
            for mcp_id in task.candidate_mcps:
                if mcp_id not in mcp_ids:
                    return False, f"任务 {task.id} 引用了不存在的MCP: {mcp_id}"
        
        # 检查依赖关系（无循环依赖）
        G = nx.DiGraph()
        for task in tasks:
            G.add_node(task.id)
            for dep in task.depends_on:
                G.add_edge(dep, task.id)
        
        if not nx.is_directed_acyclic_graph(G):
            return False, "任务依赖关系存在循环"
        
        return True, ""
    
    def _check_semantics(self, query: str, tasks: List[Task], mcp_box: List[MCPMetadata]) -> tuple[bool, str]:
        """语义检查（使用LLM）"""
        prompt = self._build_validation_prompt(query, tasks, mcp_box)
        
        messages = [
            {"role": "system", "content": "你是一个任务规划质量检查专家。"},
            {"role": "user", "content": prompt}
        ]
        
        result = self.llm.generate_json(messages)
        
        if result is None:
            return True, ""  # LLM失败时默认通过（避免阻塞）
        
        is_valid = result.get("is_valid", True)
        issues = result.get("issues", [])
        
        if not is_valid:
            feedback = "; ".join([issue["description"] for issue in issues])
            return False, feedback
        
        return True, ""
    
    def _build_validation_prompt(self, query: str, tasks: List[Task], mcp_box: List[MCPMetadata]) -> str:
        """构建验证提示词"""
        tasks_str = "\n".join([
            f"- {task.id}: {task.description} (使用: {', '.join(task.candidate_mcps)}, 依赖: {task.depends_on})"
            for task in tasks
        ])
        
        mcp_str = "\n".join([f"- {mcp.id}: {mcp.capability_description}" for mcp in mcp_box])
        
        prompt = f"""
请评估以下任务分解的质量。

原始查询: {query}

分解的任务:
{tasks_str}

可用MCP:
{mcp_str}

评估维度:
1. 完整性: 所有子任务是否覆盖原查询的需求？
2. 可行性: 每个任务选择的MCP是否合适？
3. 逻辑性: 任务顺序和依赖关系是否合理？

输出JSON格式:
{{
  "is_valid": true/false,
  "issues": [
    {{
      "type": "completeness/feasibility/logic",
      "description": "具体问题描述",
      "affected_tasks": ["task_id"]
    }}
  ],
  "score": 0-10的评分
}}

只输出JSON，不要其他文字。
"""
        return prompt