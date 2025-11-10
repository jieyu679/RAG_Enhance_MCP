from typing import List, Dict
from src.core.data_structures import Task, MCPMetadata, PlanningResult
from src.core.llm_client import Qwen3Client
import uuid

class Decomposer:
    """Task Decomposition Agent"""
    
    def __init__(self, llm_client: Qwen3Client):
        self.llm = llm_client
    
    def decompose(self, query: str, mcp_box: List[MCPMetadata]) -> PlanningResult:
        """
        Decompose user query into a task DAG
        
        Args:
            query: User query
            mcp_box: Available MCP list (metadata only)
        
        Returns:
            PlanningResult containing task list
        """
        # Build prompt
        prompt = self._build_prompt(query, mcp_box)
        
        messages = [
            {"role": "system", "content": "You are an expert in task decomposition."},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM
        result = self.llm.generate_json(messages)
        
        if result is None:
            return PlanningResult(tasks=[], is_valid=False, feedback="LLM returned invalid format")
        
        # Parse tasks
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
        """Build decomposition prompt"""
        mcp_descriptions = "\n".join([
            f"- {mcp.id}: {mcp.name} - {mcp.capability_description}"
            for mcp in mcp_box
        ])
        
        prompt = f"""
You are a task decomposition expert. Please decompose the user's complex query into executable sub-tasks.

Available Tools (MCPs):
{mcp_descriptions}

User Query: {query}

Please decompose the query into 3-7 sub-tasks. For each task, specify:
1. Clear description (one sentence)
2. Recommended MCP tools (select 1-3 candidates from the list above)
3. Dependencies on prerequisite tasks (list of task_ids, empty if none)

Decomposition Principles:
- Each task should be atomic (cannot be further divided)
- Dependencies between tasks must be clear
- Selected MCPs should match the task description
- Consider parallel execution possibilities between tasks

Output in strict JSON format:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "Task description",
      "candidate_mcps": ["mcp_id_1", "mcp_id_2"],
      "depends_on": [],
      "rationale": "Brief explanation of why this task is needed"
    }}
  ],
  "reasoning": "Overall decomposition reasoning"
}}

Output only JSON, no other text.
"""
        return prompt