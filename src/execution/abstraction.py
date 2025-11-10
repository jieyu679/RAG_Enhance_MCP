from typing import List, Dict, Optional
from src.core.data_structures import Task, MCPMetadata, ExecutionResult
from src.core.llm_client import Qwen3Client
from collections import defaultdict
import uuid

class RawMCPPool:
    """Raw MCP Pool (stores successful task executions)"""
    
    def __init__(self):
        self.successful_tasks = []
    
    def add(self, task: Task, mcp: MCPMetadata, result: ExecutionResult):
        """Add successful task"""
        self.successful_tasks.append({
            "task": task,
            "mcp": mcp,
            "result": result
        })
    
    def find_similar_clusters(self, threshold: int = 3) -> List[List[dict]]:
        """Find similar task clusters (simplified: group by MCP ID)"""
        # Group by MCP ID
        mcp_groups = defaultdict(list)
        for item in self.successful_tasks:
            mcp_groups[item["mcp"].id].append(item)
        
        # Filter groups with count >= threshold
        clusters = []
        for mcp_id, items in mcp_groups.items():
            if len(items) >= threshold:
                clusters.append(items)
        
        return clusters

class MCPAbstractionPipeline:
    """MCP Abstraction Pipeline"""
    
    def __init__(self, llm_client: Qwen3Client, success_rate_threshold: float = 0.8):
        self.llm = llm_client
        self.success_rate_threshold = success_rate_threshold
    
    def abstract(self, similar_tasks: List[dict]) -> Optional[MCPMetadata]:
        """
        Abstract new MCP from similar tasks
        
        Args:
            similar_tasks: List of similar tasks (including task, mcp, result)
        
        Returns:
            New MCP or None
        """
        if len(similar_tasks) < 3:
            return None
        
        # Calculate success rate
        success_count = sum(1 for item in similar_tasks if item["result"].success)
        success_rate = success_count / len(similar_tasks)
        
        if success_rate < self.success_rate_threshold:
            print(f"⚠️ Success rate {success_rate:.2f} below threshold {self.success_rate_threshold}, skipping abstraction")
            return None
        
        # Stage 1: Parameter generalization
        generalized = self._generalize_parameters(similar_tasks)
        if generalized is None:
            return None
        
        # Stage 2: Generate documentation
        documented = self._generate_documentation(generalized, similar_tasks)
        
        return documented
    
    def _generalize_parameters(self, similar_tasks: List[dict]) -> Optional[Dict]:
        """Parameter generalization (using LLM)"""
        task_descriptions = "\n".join([
            f"- {item['task'].description}"
            for item in similar_tasks[:5]  # Max 5 examples
        ])
        
        prompt = f"""
Here are several successfully executed similar tasks:
{task_descriptions}

Please extract the common pattern and replace specific values with parameters.

Example:
Task 1: "Query order history for user Zhang San"
Task 2: "Query order history for user Li Si"
Abstraction: "Query order history for user {{user_name}}"

Output JSON format:
{{
  "template": "Generalized task description template",
  "parameters": {{
    "param_name": "Parameter description"
  }},
  "capability": "Capability description of this MCP (one sentence)"
}}

Output only JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are an MCP abstraction expert."},
            {"role": "user", "content": prompt}
        ]
        
        result = self.llm.generate_json(messages)
        return result
    
    def _generate_documentation(self, generalized: Dict, examples: List[dict]) -> MCPMetadata:
        """Generate documentation"""
        mcp_id = f"mcp_{uuid.uuid4().hex[:8]}"
        
        # Extract examples
        example_list = []
        for item in examples[:3]:
            example_list.append({
                "task": item["task"].description,
                "output": str(item["result"].output)[:200]  # Truncate
            })
        
        mcp = MCPMetadata(
            id=mcp_id,
            name=generalized.get("template", "Unnamed MCP"),
            capability_description=generalized.get("capability", ""),
            parameters=generalized.get("parameters", {}),
            success_count=sum(1 for item in examples if item["result"].success),
            failure_count=len(examples) - sum(1 for item in examples if item["result"].success),
            examples=example_list
        )
        
        return mcp