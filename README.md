# Multi-Agent Collaborative Planning with Co-Evolving Action Spaces

This repository contains the implementation of a novel framework that combines:
1. **Multi-Agent Collaborative Planning** (Decomposer-Refiner-Scheduler)
2. **Co-Evolution of Dynamic Action Space and Policy** (Dynamic MCP Box + DQN)

## 🎯 Core Innovations

### Innovation 1: Multi-Agent Collaborative Planning
- **Decomposer**: Breaks down complex queries into executable sub-tasks
- **Refiner**: Validates task decomposition quality through feedback loops
- **Scheduler**: Optimizes execution order with parallel processing support

### Innovation 2: Co-Evolution of Action Space and Policy
- **Dynamic MCP Box**: Grows from 10 seed MCPs to 100+ through abstraction
- **Dynamic DQN**: Adapts to expanding action space via action embeddings
- **Continuous Improvement**: Performance increases over time through experience

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Edit `config.yaml` to set your Qwen3 API endpoint:
```yaml
llm:
  base_url: "http://localhost:8000/v1"
  model_name: "Qwen3-14B-Chat"
```

### Run Demo
```bash
python demo.py
```

### Visualize Results
```bash
python src/utils/visualization.py
```

## 📊 Expected Results

After running the demo, you should observe:
- **Learning Curve**: Success rate increases from ~60% to ~90%
- **MCP Growth**: MCP Box expands from 10 to 15+ MCPs
- **Policy Improvement**: DQN epsilon decreases from 0.9 to ~0.1

## 📈 Experimental Evaluation

Key metrics tracked:
- Task success rate
- Episode reward
- MCP Box size
- Token efficiency
- Execution time

Results are saved to `./outputs/demo_results.json`

## 🏗️ Project Structure
```
├── src/
│   ├── planning/          # Multi-agent planning (Innovation 1)
│   ├── execution/         # Dynamic MCP selection (Innovation 2)
│   └── core/             # Data structures and utilities
├── data/                  # Seed MCPs and test queries
├── outputs/              # Results, models, and logs
└── demo.py               # Main demonstration script
```

## 📄 Citation

If you use this code for research, please cite:
```bibtex
@inproceedings{yourname2025multiagent,
  title={Multi-Agent Collaborative Planning with Co-Evolving Action Spaces for Complex Task Execution},
  author={Your Name},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## 📧 Contact

For questions, please contact: your.email@domain.com