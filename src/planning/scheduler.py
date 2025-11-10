from typing import List
from src.core.data_structures import Task, PlanningResult
import networkx as nx

class Scheduler:
    """任务调度智能体"""
    
    def schedule(self, planning_result: PlanningResult) -> PlanningResult:
        """
        生成任务执行计划（按批次组织，支持并行）
        
        Args:
            planning_result: 包含任务列表的规划结果
        
        Returns:
            更新后的PlanningResult（添加execution_plan）
        """
        tasks = planning_result.tasks
        
        # 构建依赖图
        G = nx.DiGraph()
        task_dict = {task.id: task for task in tasks}
        
        for task in tasks:
            G.add_node(task.id)
            for dep in task.depends_on:
                if dep in task_dict:
                    G.add_edge(dep, task.id)
        
        # 拓扑排序分层
        execution_plan = []
        remaining_nodes = set(G.nodes())
        
        while remaining_nodes:
            # 找到当前没有依赖的任务（可以并行执行）
            ready_tasks = [
                node for node in remaining_nodes
                if all(dep not in remaining_nodes for dep in G.predecessors(node))
            ]
            
            if not ready_tasks:
                # 如果没有ready的任务但还有剩余，说明有循环依赖（理论上Refiner应该已经检查）
                break
            
            execution_plan.append(ready_tasks)
            remaining_nodes -= set(ready_tasks)
        
        planning_result.execution_plan = execution_plan
        return planning_result