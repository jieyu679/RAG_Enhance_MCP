import matplotlib.pyplot as plt
import json
import numpy as np

def plot_learning_curves(results_path: str = "./outputs/demo_results.json"):
    """绘制学习曲线"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 提取数据
    rewards = [r['episode_reward'] for r in results]
    mcp_box_sizes = [r['mcp_box_size'] for r in results]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 子图1: 奖励曲线
    axes[0].plot(rewards, marker='o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Query Index', fontsize=12)
    axes[0].set_ylabel('Episode Reward', fontsize=12)
    axes[0].set_title('Learning Curve: Reward over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: MCP Box增长曲线
    axes[1].plot(mcp_box_sizes, marker='s', color='green', linewidth=2, markersize=6)
    axes[1].set_xlabel('Query Index', fontsize=12)
    axes[1].set_ylabel('MCP Box Size', fontsize=12)
    axes[1].set_title('Co-Evolution: MCP Box Growth', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./outputs/learning_curves.png', dpi=300, bbox_inches='tight')
    print("✅ 学习曲线已保存到 ./outputs/learning_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()