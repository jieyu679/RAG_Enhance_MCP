from src.core.data_structures import Task, MCPMetadata, ExecutionResult
from src.core.llm_client import Qwen3Client
import time

class MCPExecutor:
    """MCP执行器"""
    
    def __init__(self, llm_client: Qwen3Client):
        self.llm = llm_client
    
    def execute(self, task: Task, mcp: MCPMetadata, context: dict = None) -> ExecutionResult:
        """
        执行任务
        
        Args:
            task: 待执行的任务
            mcp: 选定的MCP
            context: 上下文信息（如前置任务的结果）
        
        Returns:
            ExecutionResult
        """
        start_time = time.time()
        
        # 构建执行提示词
        prompt = self._build_execution_prompt(task, mcp, context)
        
        messages = [
            {"role": "system", "content": f"你是一个{mcp.name}工具。"},
            {"role": "user", "content": prompt}
        ]
        
        # 调用LLM执行
        response = self.llm.chat(messages)
        
        execution_time = time.time() - start_time
        
        if not response["success"]:
            return ExecutionResult(
                success=False,
                output=None,
                error_message=response.get("error", "未知错误"),
                token_count=0,
                execution_time=execution_time
            )
        
        return ExecutionResult(
            success=True,
            output=response["content"],
            token_count=response["tokens"],
            execution_time=execution_time,
            quality_score=0.8  # 简化版：固定评分
        )
    
    def _build_execution_prompt(self, task: Task, mcp: MCPMetadata, context: dict) -> str:
        """构建执行提示词"""
        context_str = ""
        if context and context.get("previous_results"):
            context_str = "\n可用的前置任务结果:\n"
            for task_id, result in context["previous_results"].items():
                context_str += f"- {task_id}: {result}\n"
        
        prompt = f"""
你需要完成以下任务: {task.description}

{context_str}

请直接输出结果，不要有多余的解释。
"""
        return prompt