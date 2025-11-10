from typing import List
from src.core.data_structures import Task, MCPMetadata, PlanningResult
from src.core.llm_client import Qwen3Client
import networkx as nx

class Refiner:
    """Task Validation Agent任务验证智能体"""
    
    def __init__(self, llm_client: Qwen3Client):
        self.llm = llm_client
    
    def validate(self, query: str, planning_result: PlanningResult, mcp_box: List[MCPMetadata]) -> PlanningResult:
        """
        Validate the quality of task decomposition
        
        Args:
            query: Original user query
            planning_result: Output from Decomposer
            mcp_box: Available MCP list
        
        Returns:
            Updated PlanningResult (with is_valid + feedback)
        """
        tasks = planning_result.tasks
        
        # Check 1: Structural validation (fast, rule-based)
        structure_valid, structure_feedback = self._check_structure(tasks, mcp_box)
        if not structure_valid:
            planning_result.is_valid = False
            planning_result.feedback = structure_feedback
            return planning_result
        
        # Check 2: Semantic validation (using LLM)
        semantic_valid, semantic_feedback = self._check_semantics(query, tasks, mcp_box)
        if not semantic_valid:
            planning_result.is_valid = False
            planning_result.feedback = semantic_feedback
            return planning_result
        
        planning_result.is_valid = True
        planning_result.feedback = "Validation passed"
        return planning_result
    
    def _check_structure(self, tasks: List[Task], mcp_box: List[MCPMetadata]) -> tuple[bool, str]:
        """Structural validation"""
        if len(tasks) == 0:
            return False, "Task list is empty"
        
        if len(tasks) > 10:
            return False, "Too many tasks (>10), please simplify"
        
        # Check if MCPs exist
        mcp_ids = {mcp.id for mcp in mcp_box}
        for task in tasks:
            if not task.candidate_mcps:
                return False, f"Task {task.id} has no candidate MCPs"
            for mcp_id in task.candidate_mcps:
                if mcp_id not in mcp_ids:
                    return False, f"Task {task.id} references non-existent MCP: {mcp_id}"
        
        # Check dependencies (no circular dependencies)
        G = nx.DiGraph()
        for task in tasks:
            G.add_node(task.id)
            for dep in task.depends_on:
                G.add_edge(dep, task.id)
        
        if not nx.is_directed_acyclic_graph(G):
            return False, "Task dependencies contain cycles"
        
        return True, ""
    
    def _check_semantics(self, query: str, tasks: List[Task], mcp_box: List[MCPMetadata]) -> tuple[bool, str]:
        """Semantic validation (using LLM)"""
        prompt = self._build_validation_prompt(query, tasks, mcp_box)
        
        messages = [
            {"role": "system", "content": "You are an expert in task planning quality assessment."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.llm.generate_json(messages)
        
        if result is None:
            return True, ""  # Default to pass if LLM fails (avoid blocking)
        
        is_valid = result.get("is_valid", True)
        issues = result.get("issues", [])
        
        if not is_valid:
            feedback = "; ".join([issue["description"] for issue in issues])
            return False, feedback
        
        return True, ""
    
    def _build_validation_prompt(self, query: str, tasks: List[Task], mcp_box: List[MCPMetadata]) -> str:
        """Build validation prompt"""
        tasks_str = "\n".join([
            f"- {task.id}: {task.description} (Uses: {', '.join(task.candidate_mcps)}, Depends on: {task.depends_on})"
            for task in tasks
        ])
        
        mcp_str = "\n".join([f"- {mcp.id}: {mcp.capability_description}" for mcp in mcp_box])
        
        prompt = f"""
Please evaluate the quality of the following task decomposition.

Original Query: {query}

Decomposed Tasks:
{tasks_str}

Available MCPs:
{mcp_str}

Evaluation Dimensions:
1. Completeness: Do all sub-tasks cover the requirements of the original query?
2. Feasibility: Is the selected MCP appropriate for each task?
3. Logic: Are the task order and dependencies reasonable?

Output JSON format:
{{
  "is_valid": true/false,
  "issues": [
    {{
      "type": "completeness/feasibility/logic",
      "description": "Specific issue description",
      "affected_tasks": ["task_id"]
    }}
  ],
  "score": 0-10 rating
}}

Output only JSON, no other text.
"""
        return prompt