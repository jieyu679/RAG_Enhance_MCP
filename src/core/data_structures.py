from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class Task:
    """任务数据结构"""
    id: str
    description: str
    candidate_mcps: List[str]
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    selected_mcp: Optional[str] = None
    execution_time: float = 0.0
    token_count: int = 0
    
    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "candidate_mcps": self.candidate_mcps,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "selected_mcp": self.selected_mcp
        }

@dataclass
class MCPMetadata:
    """MCP元数据（Planners看到的信息）"""
    id: str
    name: str
    capability_description: str
    parameters: Dict[str, str]
    success_count: int = 0
    failure_count: int = 0
    total_tokens: int = 0
    examples: List[Dict] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "capability_description": self.capability_description,
            "parameters": self.parameters,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "examples": self.examples
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            name=data["name"],
            capability_description=data["capability_description"],
            parameters=data["parameters"],
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            total_tokens=data.get("total_tokens", 0),
            examples=data.get("examples", [])
        )

@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    output: Any
    error_message: Optional[str] = None
    token_count: int = 0
    execution_time: float = 0.0
    quality_score: float = 0.0  # 0-1，由LLM评估

@dataclass
class PlanningResult:
    """规划结果"""
    tasks: List[Task]
    is_valid: bool
    feedback: Optional[str] = None
    execution_plan: Optional[List[List[str]]] = None  # 按批次组织的task_ids

@dataclass
class Experience:
    """强化学习经验"""
    state: Any
    action: str  # MCP ID
    reward: float
    next_state: Any
    done: bool