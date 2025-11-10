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
    """Multi-Agent Co-Evolution System"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # LLM client
        self.llm = Qwen3Client(
            base_url=config['llm']['base_url'],
            model_name=config['llm']['model_name'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens']
        )
        
        # Planning layer: Multi-agent
        self.decomposer = Decomposer(self.llm)
        self.refiner = Refiner(self.llm)
        self.scheduler = Scheduler()
        
        # Execution layer: Dynamic space and policy
        self.mcp_box = DynamicMCPBox()
        self.mcp_box.load()
        
        self.retriever = MCPRetriever(config['retrieval']['model_name'])
        
        # 🔧 Auto-detect embedding dimension and update config
        actual_dim = self.retriever.embedding_dim
        if actual_dim != config['dqn']['state_dim']:
            print(f"⚠️ Detected embedding dimension mismatch: config={config['dqn']['state_dim']}, actual={actual_dim}")
            print(f"✅ Auto-updated to: {actual_dim}")
            config['dqn']['state_dim'] = actual_dim
            config['dqn']['action_emb_dim'] = actual_dim
        
        self.dqn_agent = DynamicDQNAgent(config['dqn'], self.retriever)
        self.executor = MCPExecutor(self.llm)
        
        # Learning layer: MCP abstraction
        self.raw_mcp_pool = RawMCPPool()
        self.abstraction_pipeline = MCPAbstractionPipeline(
            self.llm,
            config['system']['success_rate_threshold']
        )
        
        # Statistics
        self.query_count = 0
        self.total_rewards = []
    
    def process_query(self, query: str) -> Dict:
        """
        Process user query (complete pipeline)
        
        Returns:
            Result dictionary
        """
        self.query_count += 1
        print(f"\n{'='*60}")
        print(f"Query #{self.query_count}: {query}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # ===== Stage 1: Multi-Agent Collaborative Planning =====
        print("\n[Stage 1] Multi-Agent Planning...")
        
        # Decomposer
        mcps = self.mcp_box.get_all_mcps()
        planning_result = self.decomposer.decompose(query, mcps)
        
        if not planning_result.is_valid:
            return {"success": False, "error": "Decomposition failed"}
        
        print(f"  ✓ Decomposer: Decomposed into {len(planning_result.tasks)} tasks")
        
        # Refiner
        planning_result = self.refiner.validate(query, planning_result, mcps)
        
        if not planning_result.is_valid:
            print(f"  ✗ Refiner: Validation failed - {planning_result.feedback}")
            # Simplified: don't re-decompose, just return failure
            return {"success": False, "error": planning_result.feedback}
        
        print(f"  ✓ Refiner: Validation passed")
        
        # Scheduler
        planning_result = self.scheduler.schedule(planning_result)
        print(f"  ✓ Scheduler: Generated {len(planning_result.execution_plan)} batches")
        
        # ===== Stage 2: Dynamic MCP Selection & Execution =====
        print("\n[Stage 2] Dynamic MCP Selection & Execution...")
        
        execution_history = []
        task_results = {}
        episode_reward = 0
        
        for batch_idx, batch in enumerate(planning_result.execution_plan):
            print(f"\n  Batch {batch_idx + 1}/{len(planning_result.execution_plan)}:")
            
            for task_id in batch:
                task = next(t for t in planning_result.tasks if t.id == task_id)
                
                # Retrieve candidate MCPs
                candidate_mcps = self.retriever.retrieve(
                    task,
                    mcps,
                    top_k=self.config['retrieval']['top_k']
                )
                
                if not candidate_mcps:
                    print(f"    ✗ {task.id}: No available MCPs")
                    continue
                
                # DQN selects MCP
                state = self.dqn_agent.get_state(task, execution_history)
                selected_mcp = self.dqn_agent.select_action(state, candidate_mcps)
                
                print(f"    → {task.id}: Selected {selected_mcp.name} (ε={self.dqn_agent.epsilon:.3f})")
                
                # Execute task
                context = {"previous_results": task_results}
                result = self.executor.execute(task, selected_mcp, context)
                
                # Update task status
                task.status = TaskStatus.SUCCESS if result.success else TaskStatus.FAILED
                task.selected_mcp = selected_mcp.id
                task.result = result.output
                task.execution_time = result.execution_time
                task.token_count = result.token_count
                
                task_results[task.id] = result.output
                
                # Compute reward
                reward = self._compute_reward(task, selected_mcp, result)
                episode_reward += reward
                
                # Store experience
                next_state = state  # Simplified
                self.dqn_agent.store_experience(state, selected_mcp, reward, next_state, False)
                
                # Update MCP statistics
                self.mcp_box.update_stats(selected_mcp.id, result.success, result.token_count)
                
                # Add to Raw MCP Pool
                if result.success:
                    self.raw_mcp_pool.add(task, selected_mcp, result)
                
                execution_history.append({
                    "task_id": task.id,
                    "mcp_id": selected_mcp.id,
                    "success": result.success,
                    "reward": reward
                })
                
                print(f"      {'✓' if result.success else '✗'} {'Success' if result.success else 'Failed'} (reward: {reward:.1f})")
        
        # ===== Stage 3: Policy Learning =====
        print("\n[Stage 3] Policy Learning...")
        loss = self.dqn_agent.train_step()
        if loss is not None:
            print(f"  ✓ DQN Training: loss={loss:.4f}")
        
        # ===== Stage 4: MCP Abstraction (triggered periodically) =====
        if self.query_count % self.config['system']['mcp_abstraction_threshold'] == 0:
            print("\n[Stage 4] MCP Abstraction...")
            self._trigger_mcp_abstraction()
        
        # Statistics
        total_time = time.time() - start_time
        success_count = sum(1 for t in planning_result.tasks if t.status == TaskStatus.SUCCESS)
        total_tasks = len(planning_result.tasks)
        
        self.total_rewards.append(episode_reward)
        
        print(f"\n{'='*60}")
        print(f"Completed: {success_count}/{total_tasks} tasks succeeded")
        print(f"Total Reward: {episode_reward:.1f} | Time: {total_time:.2f}s")
        print(f"MCP Box: {len(self.mcp_box.mcps)} MCPs")
        print(f"{'='*60}")
        
        return {
            "success": success_count == total_tasks,
            "tasks": [t.to_dict() for t in planning_result.tasks],
            "episode_reward": episode_reward,
            "execution_time": total_time,
            "mcp_box_size": len(self.mcp_box.mcps)
        }
    
    def _compute_reward(self, task: Task, mcp, result: ExecutionResult) -> float:
        """Compute reward"""
        if not result.success:
            return -5.0
        
        reward = 10.0  # Base success reward
        
        # Efficiency reward (fewer tokens = higher reward)
        if result.token_count < 200:
            reward += 2.0
        
        # Quality reward
        reward += result.quality_score * 3.0
        
        return reward
    
    def _trigger_mcp_abstraction(self):
        """Trigger MCP abstraction"""
        clusters = self.raw_mcp_pool.find_similar_clusters(
            threshold=self.config['system']['mcp_abstraction_threshold']
        )
        
        if not clusters:
            print("  → No abstractable patterns found")
            return
        
        print(f"  → Found {len(clusters)} candidate clusters")
        
        for cluster in clusters:
            new_mcp = self.abstraction_pipeline.abstract(cluster)
            if new_mcp:
                added = self.mcp_box.add_mcp(new_mcp)
                if added:
                    print(f"  ✓ Abstracted new MCP: {new_mcp.name}")
    
    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        self.mcp_box.save()
        self.dqn_agent.save(f"{path}/dqn_agent.pth")
        print(f"💾 Checkpoint saved to {path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        mcp_stats = self.mcp_box.get_stats()
        
        return {
            "total_queries": self.query_count,
            "avg_reward": sum(self.total_rewards) / len(self.total_rewards) if self.total_rewards else 0,
            "mcp_box_stats": mcp_stats,
            "dqn_epsilon": self.dqn_agent.epsilon,
            "dqn_training_steps": self.dqn_agent.training_steps
        }