from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from src.core.data_structures import Task, MCPMetadata

class MCPRetriever:
    """MCP检索器（使用Sentence-BERT）"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.mcp_embeddings = {}
        
        # 获取嵌入维度
        test_emb = self.model.encode("test")
        self.embedding_dim = len(test_emb)
        print(f"嵌入维度: {self.embedding_dim}")
    
    def encode_task(self, task: Task) -> np.ndarray:
        """编码任务描述"""
        emb = self.model.encode(task.description, convert_to_numpy=True)
        # 确保是1D数组
        if len(emb.shape) == 0:
            emb = np.array([emb])
        return emb
    
    def encode_mcp(self, mcp: MCPMetadata) -> np.ndarray:
        """编码MCP"""
        text = f"{mcp.name}: {mcp.capability_description}"
        emb = self.model.encode(text, convert_to_numpy=True)
        # 确保是1D数组
        if len(emb.shape) == 0:
            emb = np.array([emb])
        return emb
    
    def update_embeddings(self, mcps: List[MCPMetadata]):
        """更新MCP嵌入缓存"""
        for mcp in mcps:
            if mcp.id not in self.mcp_embeddings:
                self.mcp_embeddings[mcp.id] = self.encode_mcp(mcp)
    
    def retrieve(self, task: Task, mcps: List[MCPMetadata], top_k: int = 5) -> List[MCPMetadata]:
        """
        检索最相关的MCP
        
        Args:
            task: 待执行的任务
            mcps: 候选MCP列表
            top_k: 返回top-k个
        
        Returns:
            按相似度排序的MCP列表
        """
        if not mcps:
            return []
        
        # 更新嵌入
        self.update_embeddings(mcps)
        
        # 编码任务
        task_emb = self.encode_task(task)
        
        # 计算相似度
        similarities = []
        for mcp in mcps:
            mcp_emb = self.mcp_embeddings[mcp.id]
            
            # 确保维度匹配
            if task_emb.shape != mcp_emb.shape:
                print(f"⚠️ 维度不匹配: task={task_emb.shape}, mcp={mcp_emb.shape}")
                continue
            
            similarity = np.dot(task_emb, mcp_emb) / (np.linalg.norm(task_emb) * np.linalg.norm(mcp_emb) + 1e-10)
            similarities.append((mcp, similarity))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mcp for mcp, _ in similarities[:top_k]]