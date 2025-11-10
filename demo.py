import yaml
import json
from src.system import MultiAgentCoEvolutionSystem
from src.execution.mcp_box import DynamicMCPBox
from src.core.data_structures import MCPMetadata
import os

def initialize_seed_mcps():
    """Initialize seed MCPs"""
    seed_mcps = [
        MCPMetadata(
            id="mcp_search",
            name="Information Search",
            capability_description="Search for specific information from given text or data sources",
            parameters={"query": "Search keywords", "source": "Data source"}
        ),
        MCPMetadata(
            id="mcp_calculate",
            name="Numerical Calculation",
            capability_description="Perform mathematical calculations including arithmetic and statistics",
            parameters={"expression": "Mathematical expression", "data": "Input data"}
        ),
        MCPMetadata(
            id="mcp_summarize",
            name="Text Summarization",
            capability_description="Summarize and abstract given text",
            parameters={"text": "Text to summarize", "max_length": "Maximum length"}
        ),
        MCPMetadata(
            id="mcp_extract",
            name="Information Extraction",
            capability_description="Extract structured information from text, such as dates, names, locations",
            parameters={"text": "Input text", "target_type": "Extraction type"}
        ),
        MCPMetadata(
            id="mcp_transform",
            name="Format Conversion",
            capability_description="Convert data from one format to another",
            parameters={"input_format": "Input format", "output_format": "Output format"}
        ),
        MCPMetadata(
            id="mcp_analyze",
            name="Data Analysis",
            capability_description="Perform statistical analysis on data, compute means, trends, etc.",
            parameters={"data": "Input data", "analysis_type": "Analysis type"}
        ),
        MCPMetadata(
            id="mcp_generate",
            name="Content Generation",
            capability_description="Generate text, tables or other content based on requirements",
            parameters={"template": "Generation template", "parameters": "Parameters"}
        ),
        MCPMetadata(
            id="mcp_compare",
            name="Information Comparison",
            capability_description="Compare similarities and differences between two or more objects",
            parameters={"items": "Items to compare", "criteria": "Comparison criteria"}
        ),
        MCPMetadata(
            id="mcp_sort",
            name="Data Sorting",
            capability_description="Sort data according to specified rules",
            parameters={"data": "Input data", "key": "Sorting key"}
        ),
        MCPMetadata(
            id="mcp_filter",
            name="Data Filtering",
            capability_description="Filter data based on conditions",
            parameters={"data": "Input data", "condition": "Filter condition"}
        )
    ]
    
    # Save seed MCPs
    mcp_box = DynamicMCPBox()
    for mcp in seed_mcps:
        mcp_box.add_mcp(mcp)
    print(f"✅ Initialized {len(seed_mcps)} seed MCPs")

def load_test_queries():
    """Load test queries"""
    queries = [
        "Calculate the sum of numbers from 1 to 100 and analyze the result",
        "Extract all person names from the following text and sort them alphabetically: Zhang San is a friend of Li Si, and Wang Wu also knows them",
        "Compare sales data between 2023 and 2024, and generate a summary report",
        "Search for recent AI news, extract key information and summarize",
        "Convert the following data to JSON format: name=Zhang San, age=25, city=Beijing",
    ]
    return queries

def run_demo():
    """Run demonstration"""
    print("="*70)
    print("  Multi-Agent Collaboration + Dynamic Action Space Co-Evolution Demo")
    print("="*70)
    
    # Load configuration
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check if initialization is needed
    if not os.path.exists("./outputs/mcp_box/mcp_box.json"):
        print("\nFirst run, initializing seed MCPs...")
        initialize_seed_mcps()
    
    # Create system
    print("\nInitializing system...")
    system = MultiAgentCoEvolutionSystem(config)
    
    # Load test queries
    queries = load_test_queries()
    
    print(f"\nReady to process {len(queries)} test queries\n")
    input("Press Enter to start...")
    
    # Process queries
    results = []
    for i, query in enumerate(queries):
        result = system.process_query(query)
        results.append(result)
        
        # Save periodically
        if (i + 1) % 2 == 0:
            system.save_checkpoint("./outputs/models")
        
        print("\nPausing for 3 seconds...\n")
        import time
        time.sleep(3)
    
    # Final statistics
    print("\n" + "="*70)
    print("  Experiment Summary")
    print("="*70)
    
    stats = system.get_statistics()
    print(f"\nTotal Queries: {stats['total_queries']}")
    print(f"Average Reward: {stats['avg_reward']:.2f}")
    print(f"\nMCP Box Statistics:")
    print(f"  - Total MCPs: {stats['mcp_box_stats']['total_mcps']}")
    print(f"  - Total Executions: {stats['mcp_box_stats']['total_executions']}")
    print(f"  - Overall Success Rate: {stats['mcp_box_stats']['overall_success_rate']:.2%}")
    print(f"\nDQN Statistics:")
    print(f"  - Current ε: {stats['dqn_epsilon']:.3f}")
    print(f"  - Training Steps: {stats['dqn_training_steps']}")
    
    # Save results
    with open("./outputs/demo_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "queries": queries,
            "results": results,
            "statistics": stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to ./outputs/demo_results.json")
    print("="*70)

if __name__ == "__main__":
    run_demo()