import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Optional
from src.core.data_structures import MCPMetadata, Task, Experience

class DynamicQNetwork(nn.Module):
    """动态Q网络（接受action embedding）"""
    
    def __init__(self, state_dim: int, action_emb_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + action_emb_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        print(f"[DQN] 网络输入维度: {input_dim} (state:{state_dim} + action:{action_emb_dim})")
    
    def forward(self, state, action_embedding):
        """
        Args:
            state: [batch_size, state_dim]
            action_embedding: [batch_size, action_emb_dim]
        Returns:
            q_values: [batch_size, 1]
        """
        # 确保维度正确
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action_embedding.shape) == 1:
            action_embedding = action_embedding.unsqueeze(0)
        
        combined = torch.cat([state, action_embedding], dim=-1)
        return self.network(combined)

class DynamicDQNAgent:
    """动态DQN智能体（核心创新：处理动态动作空间）"""
    
    def __init__(self, config: dict, retriever):
        self.config = config
        self.retriever = retriever
        
        # 网络
        self.q_network = DynamicQNetwork(
            config['state_dim'],
            config['action_emb_dim'],
            config['hidden_dim']
        )
        self.target_network = DynamicQNetwork(
            config['state_dim'],
            config['action_emb_dim'],
            config['hidden_dim']
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['learning_rate'])
        
        # 经验回放
        self.replay_buffer = deque(maxlen=config['replay_buffer_size'])
        
        # 探索策略
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        
        # 训练参数
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        
        # 统计
        self.training_steps = 0
        self.episode_rewards = []
    
    def get_state(self, task: Task, execution_history: List[dict]) -> np.ndarray:
        """
        构建状态表示
        
        Args:
            task: 当前任务
            execution_history: 最近的执行历史
        
        Returns:
            state向量 [state_dim]
        """
        # 任务嵌入
        task_emb = self.retriever.encode_task(task)
        
        # 确保维度正确
        if len(task_emb.shape) == 0:
            task_emb = np.array([task_emb])
        
        # 打印调试信息（首次运行）
        if self.training_steps == 0:
            print(f"[DQN] State维度: {task_emb.shape}")
        
        return task_emb
    
    def encode_action(self, mcp: MCPMetadata) -> np.ndarray:
        """编码动作（MCP）"""
        action_emb = self.retriever.encode_mcp(mcp)
        
        # 确保维度正确
        if len(action_emb.shape) == 0:
            action_emb = np.array([action_emb])
        
        # 打印调试信息（首次运行）
        if self.training_steps == 0:
            print(f"[DQN] Action维度: {action_emb.shape}")
        
        return action_emb
    
    def select_action(self, state: np.ndarray, candidates: List[MCPMetadata]) -> MCPMetadata:
        """
        选择动作（MCP）
        
        核心创新：即使MCP Box增长，也能处理
        """
        if not candidates:
            raise ValueError("候选MCP列表为空")
        
        # Epsilon-greedy探索
        if random.random() < self.epsilon:
            return random.choice(candidates)
        
        # 利用：选择Q值最高的MCP
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            q_values = []
            for mcp in candidates:
                action_emb = self.encode_action(mcp)
                action_tensor = torch.FloatTensor(action_emb)
                if len(action_tensor.shape) == 1:
                    action_tensor = action_tensor.unsqueeze(0)
                
                # 打印维度调试信息（仅第一次）
                if len(q_values) == 0 and self.training_steps == 0:
                    print(f"[DQN] state_tensor: {state_tensor.shape}, action_tensor: {action_tensor.shape}")
                
                q = self.q_network(state_tensor, action_tensor)
                q_values.append(q.item())
        
        best_idx = np.argmax(q_values)
        return candidates[best_idx]
    
    def store_experience(self, state, action_mcp, reward, next_state, done):
        """存储经验"""
        action_emb = self.encode_action(action_mcp)
        experience = (state, action_emb, reward, next_state, done)
        self.replay_buffer.append(experience)
    
    def train_step(self) -> Optional[float]:
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor并确保维度正确
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # 确保batch维度
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)
        
        # 当前Q值
        self.q_network.train()
        current_q = self.q_network(states, actions).squeeze()
        
        # 目标Q值（简化：使用相同action embedding）
        with torch.no_grad():
            next_q = self.target_network(next_states, actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 定期更新target网络
        self.training_steps += 1
        if self.training_steps % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """保存模型"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
        print(f"💾 DQN模型已保存到 {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        print(f"📂 DQN模型已从 {path} 加载")