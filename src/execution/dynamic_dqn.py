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
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, action_embedding):
        """
        Args:
            state: [batch_size, state_dim]
            action_embedding: [batch_size, action_emb_dim]
        Returns:
            q_values: [batch_size, 1]
        """
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
        
        # 简化版：只用任务嵌入作为状态
        # 完整版可以添加：历史成功率、资源使用情况等
        return task_emb
    
    def encode_action(self, mcp: MCPMetadata) -> np.ndarray:
        """编码动作（MCP）"""
        return self.retriever.encode_mcp(mcp)
    
    def select_action(self, state: np.ndarray, candidates: List[MCPMetadata]) -> MCPMetadata:
        """
        选择动作（MCP）
        
        核心创新：即使MCP Box增长，也能处理
        """
        # Epsilon-greedy探索
        if random.random() < self.epsilon:
            return random.choice(candidates)
        
        # 利用：选择Q值最高的MCP
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            q_values = []
            for mcp in candidates:
                action_emb = self.encode_action(mcp)
                action_tensor = torch.FloatTensor(action_emb).unsqueeze(0)
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
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # 当前Q值
        self.q_network.train()
        current_q = self.q_network(states, actions).squeeze()
        
        # 目标Q值（简化：使用相同action embedding，实际应该是next_state的最优action）
        with torch.no_grad():
            next_q = self.target_network(next_states, actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
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
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']