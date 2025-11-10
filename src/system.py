from typing import List, Dict
from src.core.data_structures import Task, ExecutionResult, TaskStatus
from src.core.llm_client import Qwen3Client
from src.planning.decomposer import Decomposer
from src.planning.refiner import Refiner
from src.planning.scheduler import Scheduler
from src.execution.mcp_box import DynamicMCPBox
from src.execution.retriever import MCPRetriever
from src.execution.dynamic_dqn import DynamicDQNAgent
from src.execution.executor import MCPExecutor
from src.execution.abstraction import RawMCPPool, MCPAbstractionPipeline
import time

class MultiAgentCoEvolutionSystem:
    """多智能体共同进化系统"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # LLM客户端
        self.llm = Qwen3Client(
            base_url=config['llm']['base_url'],
            model_name=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens']
        )
        
        # 规划层：多智能体
        self.decomposer = Decomposer(self.llm)
        self.refiner = Refiner(self.llm)
        self.scheduler = Scheduler()
        
        # 执行层：动态空间与策略
        self.mcp_box = DynamicMCPBox()
        self.mcp_box.load()
        
        self.retriever = MCPRetriever(config['retrieval']['model_name'])
        self.dqn_agent = DynamicDQNAgent(config['dqn'], self.retriever)
        self.executor = MCPExecutor(self.llm)
        
        # 学习层：MCP抽象
        self.raw_mcp_pool = RawMCPPool()
        self.abstraction_pipeline = MCPAbstractionPipeline(
            self.llm,
            config['system']['success_rate_threshold']
        )
        
        # 统计
        self.query_count = 0
        self.total_rewards = []
    
    def process_query(self, query: str) -> Dict:
        """
        处理用户查询（完整流程）
        
        Returns:
            结果字典
        """
        self.query_count += 1
        print(f"\n{'='*60}")
        print(f"查询 #{self.query_count}: {query}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # ===== 阶段1: 多智能体协作规划 =====
        print("\n[阶段1] 多智能体规划...")
        
        # Decomposer分解
        mcps = self.mcp_box.get_all_mcps()
        planning_result = self.decomposer.decompose(query, mcps)
        
        if not planning_result.is_valid:
            return {"success": False, "error": "分解失败"}
        
        print(f"  ✓ Decomposer: 分解为 {len(planning_result.tasks)} 个任务")
        
        # Refiner验证
        planning_result = self.refiner.validate(query, planning_result, mcps)
        
        if not planning_result.is_valid:
            print(f"  ✗ Refiner: 验证失败 - {planning_result.feedback}")
            # 简化版：不重新分解，直接返回失败
            return {"success": False, "error": planning_result.feedback}
        
        print(f"  ✓ Refiner: 验证通过")
        
        # Scheduler调度
        planning_result = self.scheduler.schedule(planning_result)
        print(f"  ✓ Scheduler: 生成 {len(planning_result.execution_plan)} 个批次")
        
        # ===== 阶段2: 动态MCP选择与执行 =====
        print("\n[阶段2] 动态MCP选择与执行...")
        
        execution_history = []
        task_results = {}
        episode_reward = 0
        
        for batch_idx, batch in enumerate(planning_result.execution_plan):
            print(f"\n  批次 {batch_idx + 1}/{len(planning_result.execution_plan)}:")
            
            for task_id in batch:
                task = next(t for t in planning_result.tasks if t.id == task_id)
                
                # 检索候选MCP
                candidate_mcps = self.retriever.retrieve(
                    task,
                    mcps,
                    top_k=self.config['retrieval']['top_k']
                )
                
                if not candidate_mcps:
                    print(f"    ✗ {task.id}: 无可用MCP")
                    continue
                
                # DQN选择MCP
                state = self.dqn_agent.get_state(task, execution_history)
                selected_mcp = self.dqn_agent.select_action(state, candidate_mcps)
                
                print(f"    → {task.id}: 选择 {selected_mcp.name} (ε={self.dqn_agent.epsilon:.3f})")
                
                # 执行任务
                context = {"previous_results": task_results}
                result = self.executor.execute(task, selected_mcp, context)
                
                # 更新任务状态
                task.status = TaskStatus.SUCCESS if result.success else TaskStatus.FAILED
                task.selected_mcp = selected_mcp.id
                task.result = result.output
                task.execution_time = result.execution_time
                task.token_count = result.token_count
                
                task_results[task.id] = result.output
                
                # 计算奖励
                reward = self._compute_reward(task, selected_mcp, result)
                episode_reward += reward
                
                # 存储经验
                next_state = state  # 简化版
                self.dqn_agent.store_experience(state, selected_mcp, reward, next_state, False)
                
                # 更新MCP统计
                self.mcp_box.update_stats(selected_mcp.id, result.success, result.token_count)
                
                # 添加到Raw MCP Pool
                if result.success:
                    self.raw_mcp_pool.add(task, selected_mcp, result)
                
                execution_history.append({
                    "task_id": task.id,
                    "mcp_id": selected_mcp.id,
                    "success": result.success,
                    "reward": reward
                })
                
                print(f"      {'✓' if result.success else '✗'} 执行{'成功' if result.success else '失败'} (奖励: {reward:.1f})")
        
        # ===== 阶段3: 策略学习 =====
        print("\n[阶段3] 策略学习...")
        loss = self.dqn_agent.train_step()
        if loss is not None:
            print(f"  ✓ DQN训练: loss={loss:.4f}")
        
        # ===== 阶段4: MCP抽象（定期触发） =====
        if self.query_count % self.config['system']['mcp_abstraction_threshold'] == 0:
            print("\n[阶段4] MCP抽象...")
            self._trigger_mcp_abstraction()
        
        # 统计
        total_time = time.time() - start_time
        success_count = sum(1 for t in planning_result.tasks if t.status == TaskStatus.SUCCESS)
        total_tasks = len(planning_result.tasks)
        
        self.total_rewards.append(episode_reward)
        
        print(f"\n{'='*60}")
        print(f"完成: {success_count}/{total_tasks} 任务成功")
        print(f"总奖励: {episode_reward:.1f} | 耗时: {total_time:.2f}s")
        print(f"MCP Box: {len(self.mcp_box.mcps)} 个MCP")
        print(f"{'='*60}")
        
        return {
            "success": success_count == total_tasks,
            "tasks": [t.to_dict() for t in planning_result.tasks],
            "episode_reward": episode_reward,
            "execution_time": total_time,
            "mcp_box_size": len(self.mcp_box.mcps)
        }
    
    def _compute_reward(self, task: Task, mcp, result: ExecutionResult) -> float:
        """计算奖励"""
        if not result.success:
            return -5.0
        
        reward = 10.0  # 基础成功奖励
        
        # 效率奖励（token少 = 奖励高）
        if result.token_count < 200:
            reward += 2.0
        
        # 质量奖励
        reward += result.quality_score * 3.0
        
        return reward
    
    def _trigger_mcp_abstraction(self):
        """触发MCP抽象"""
        clusters = self.raw_mcp_pool.find_similar_clusters(
            threshold=self.config['system']['mcp_abstraction_threshold']
        )
        
        if not clusters:
            print("  → 没有发现可抽象的模式")
            return
        
        print(f"  → 发现 {len(clusters)} 个候选簇")
        
        for cluster in clusters:
            new_mcp = self.abstraction_pipeline.abstract(cluster)
            if new_mcp:
                added = self.mcp_box.add_mcp(new_mcp)
                if added:
                    print(f"  ✓ 抽象出新MCP: {new_mcp.name}")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        self.mcp_box.save()
        self.dqn_agent.save(f"{path}/dqn_agent.pth")
        print(f"💾 保存检查点到 {path}")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        mcp_stats = self.mcp_box.get_stats()
        
        return {
            "total_queries": self.query_count,
            "avg_reward": sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 0,
            "mcp_box_stats": mcp_stats,
            "dqn_epsilon": self.dqn_agent.epsilon,
            "dqn_training_steps": self.dqn_agent.training_steps
        }