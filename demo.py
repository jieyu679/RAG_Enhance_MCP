import yaml
import json
from src.system import MultiAgentCoEvolutionSystem
from src.execution.mcp_box import DynamicMCPBox
from src.core.data_structures import MCPMetadata
import os

def initialize_seed_mcps():
    """初始化种子MCP"""
    seed_mcps = [
        MCPMetadata(
            id="mcp_search",
            name="信息搜索",
            capability_description="从给定文本或数据源中搜索特定信息",
            parameters={"query": "搜索关键词", "source": "数据源"}
        ),
        MCPMetadata(
            id="mcp_calculate",
            name="数值计算",
            capability_description="执行数学计算，包括加减乘除、统计等",
            parameters={"expression": "数学表达式", "data": "输入数据"}
        ),
        MCPMetadata(
            id="mcp_summarize",
            name="文本总结",
            capability_description="对给定文本进行摘要和总结",
            parameters={"text": "待总结的文本", "max_length": "最大长度"}
        ),
        MCPMetadata(
            id="mcp_extract",
            name="信息提取",
            capability_description="从文本中提取结构化信息，如日期、人名、地点等",
            parameters={"text": "输入文本", "target_type": "提取类型"}
        ),
        MCPMetadata(
            id="mcp_transform",
            name="格式转换",
            capability_description="将数据从一种格式转换为另一种格式",
            parameters={"input_format": "输入格式", "output_format": "输出格式"}
        ),
        MCPMetadata(
            id="mcp_analyze",
            name="数据分析",
            capability_description="对数据进行统计分析，计算均值、趋势等",
            parameters={"data": "输入数据", "analysis_type": "分析类型"}
        ),
        MCPMetadata(
            id="mcp_generate",
            name="内容生成",
            capability_description="根据要求生成文本、表格或其他内容",
            parameters={"template": "生成模板", "parameters": "参数"}
        ),
        MCPMetadata(
            id="mcp_compare",
            name="信息对比",
            capability_description="比较两个或多个对象的异同",
            parameters={"items": "待比较项", "criteria": "比较标准"}
        ),
        MCPMetadata(
            id="mcp_sort",
            name="数据排序",
            capability_description="对数据按指定规则进行排序",
            parameters={"data": "输入数据", "key": "排序键"}
        ),
        MCPMetadata(
            id="mcp_filter",
            name="数据筛选",
            capability_description="根据条件筛选数据",
            parameters={"data": "输入数据", "condition": "筛选条件"}
        )
    ]
    
    # 保存种子MCP
    mcp_box = DynamicMCPBox()
    for mcp in seed_mcps:
        mcp_box.add_mcp(mcp)
    print(f"✅ 初始化了 {len(seed_mcps)} 个种子MCP")

def load_test_queries():
    """加载测试查询"""
    queries = [
        "计算1到100的和并分析这个结果",
        "从以下文本中提取所有人名，然后按字母顺序排序：张三是李四的朋友，王五也认识他们",
        "比较2023年和2024年的销售数据，生成一份总结报告",
        "搜索最近的AI新闻，提取关键信息并总结",
        "将以下数据转换为JSON格式：姓名=张三,年龄=25,城市=北京",
    ]
    return queries

def run_demo():
    """运行演示"""
    print("="*70)
    print("  多智能体协作 + 动态动作空间共同进化系统 Demo")
    print("="*70)
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查是否需要初始化
    if not os.path.exists("./outputs/mcp_box/mcp_box.json"):
        print("\n首次运行，初始化种子MCP...")
        initialize_seed_mcps()
    
    # 创建系统
    print("\n初始化系统...")
    system = MultiAgentCoEvolutionSystem(config)
    
    # 加载测试查询
    queries = load_test_queries()
    
    print(f"\n准备处理 {len(queries)} 个测试查询\n")
    input("按Enter开始...")
    
    # 处理查询
    results = []
    for i, query in enumerate(queries):
        result = system.process_query(query)
        results.append(result)
        
        # 定期保存
        if (i + 1) % 2 == 0:
            system.save_checkpoint("./outputs/models")
        
        print("\n暂停3秒...\n")
        import time
        time.sleep(3)
    
    # 最终统计
    print("\n" + "="*70)
    print("  实验总结")
    print("="*70)
    
    stats = system.get_statistics()
    print(f"\n总查询数: {stats['total_queries']}")
    print(f"平均奖励: {stats['avg_reward']:.2f}")
    print(f"\nMCP Box统计:")
    print(f"  - 总MCP数: {stats['mcp_box_stats']['total_mcps']}")
    print(f"  - 总执行次数: {stats['mcp_box_stats']['total_executions']}")
    print(f"  - 整体成功率: {stats['mcp_box_stats']['overall_success_rate']:.2%}")
    print(f"\nDQN统计:")
    print(f"  - 当前ε: {stats['dqn_epsilon']:.3f}")
    print(f"  - 训练步数: {stats['dqn_training_steps']}")
    
    # 保存结果
    with open("./outputs/demo_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "queries": queries,
            "results": results,
            "statistics": stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 ./outputs/demo_results.json")
    print("="*70)

if __name__ == "__main__":
    run_demo()