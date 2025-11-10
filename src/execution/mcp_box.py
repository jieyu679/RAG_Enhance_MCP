from typing import List, Dict, Optional
from src.core.data_structures import MCPMetadata
import json
import os

class DynamicMCPBox:
    """动态MCP容器"""
    
    def __init__(self, save_path: str = "./outputs/mcp_box/mcp_box.json"):
        self.mcps: Dict[str, MCPMetadata] = {}
        self.save_path = save_path
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def add_mcp(self, mcp: MCPMetadata) -> bool:
        """添加新MCP"""
        if mcp.id in self.mcps:
            print(f"MCP {mcp.id} 已存在，跳过")
            return False
        
        self.mcps[mcp.id] = mcp
        print(f"✅ 添加MCP: {mcp.name} (总数: {len(self.mcps)})")
        self.save()
        return True
    
    def get_mcp(self, mcp_id: str) -> Optional[MCPMetadata]:
        """获取MCP"""
        return self.mcps.get(mcp_id)
    
    def get_all_mcps(self) -> List[MCPMetadata]:
        """获取所有MCP"""
        return list(self.mcps.values())
    
    def update_stats(self, mcp_id: str, success: bool, tokens: int):
        """更新MCP统计信息"""
        if mcp_id not in self.mcps:
            return
        
        mcp = self.mcps[mcp_id]
        if success:
            mcp.success_count += 1
        else:
            mcp.failure_count += 1
        mcp.total_tokens += tokens
    
    def save(self):
        """保存到文件"""
        data = {mcp_id: mcp.to_dict() for mcp_id, mcp in self.mcps.items()}
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """从文件加载"""
        if not os.path.exists(self.save_path):
            return
        
        with open(self.save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.mcps = {
            mcp_id: MCPMetadata.from_dict(mcp_data)
            for mcp_id, mcp_data in data.items()
        }
        print(f"📦 加载了 {len(self.mcps)} 个MCP")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_success = sum(mcp.success_count for mcp in self.mcps.values())
        total_failure = sum(mcp.failure_count for mcp in self.mcps.values())
        total = total_success + total_failure
        
        return {
            "total_mcps": len(self.mcps),
            "total_executions": total,
            "overall_success_rate": total_success / total if total > 0 else 0.0,
            "avg_success_rate": sum(mcp.success_rate for mcp in self.mcps.values()) / len(self.mcps) if self.mcps else 0.0
        }