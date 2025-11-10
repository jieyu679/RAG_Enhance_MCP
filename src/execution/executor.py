from src.core.data_structures import Task, MCPMetadata, ExecutionResult
from src.core.llm_client import Qwen3Client
import time

class MCPExecutor:
    """MCP Executor"""
    
    def __init__(self, llm_client: Qwen3Client):
        self.llm = llm_client
    
    def execute(self, task: Task, mcp: MCPMetadata, context: dict = None) -> ExecutionResult:
        """
        Execute task
        
        Args:
            task: Task to execute
            mcp: Selected MCP
            context: Context information (e.g., results from prerequisite tasks)
        
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        
        # Build execution prompt
        prompt = self._build_execution_prompt(task, mcp, context)
        
        messages = [
            {"role": "system", "content": f"You are a {mcp.name} tool."},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM for execution
        response = self.llm.chat(messages)
        
        execution_time = time.time() - start_time
        
        if not response["success"]:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=response.get("error", "Unknown error"),
                token_count=0,
                execution_time=execution_time
            )
        
        return ExecutionResult(
            success=True,
            output=response["content"],
            token_count=response["tokens"],
            execution_time=execution_time,
            quality_score=0.8  # Simplified: fixed score
        )
    
    def _build_execution_prompt(self, task: Task, mcp: MCPMetadata, context: dict) -> str:
        """Build execution prompt"""
        context_str = ""
        if context and context.get("previous_results"):
            context_str = "\nAvailable results from prerequisite tasks:\n"
            for task_id, result in context["previous_results"].items():
                context_str += f"- {task_id}: {result}\n"
        
        prompt = f"""
You need to complete the following task: {task.description}

{context_str}

Please output the result directly without unnecessary explanations.
"""
        return prompt