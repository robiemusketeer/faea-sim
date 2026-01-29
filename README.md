# FAEA: Frontier Agent as Embodied Agent

This repository contains evaluation code and results for the FAEA (Frontier Agent as Embodied Agent) research project, which explores how frontier AI models can directly control robotic manipulation tasks through code generation.

## Key Insight

Modern frontier AI models (like Claude Opus 4.5) can achieve strong performance on robotic manipulation benchmarks by:
1. Reading API documentation and understanding task requirements
2. Writing and iteratively refining control scripts
3. Using visual feedback and state information to correct errors

This approach requires no task-specific training or demonstrations beyond reference examples.

## Repository Structure

```
faea/
├── environments/              # Conda/Mamba environment specifications
│   ├── libero_sim.yml
│   ├── metaworld_sim.yml
│   └── maniskill_sim.yml
│
├── benchmarks/
│   ├── libero/               # LIBERO benchmark evaluation
│   │   └── src/              # Source code and prompt templates
│   │
│   ├── metaworld/            # MetaWorld benchmark evaluation
│   │   └── src/
│   │
│   └── maniskill/            # ManiSkill benchmark evaluation
│       └── src/
│
└── examples/                  # Example episode scripts
    ├── libero_pick_place.py
    ├── metaworld_pick_place.py
    └── maniskill_pick_cube.py
```

## Quick Start

### 1. Set up environment

For LIBERO:
```bash
micromamba create -f environments/libero_sim.yml
micromamba activate libero_sim
```

For MetaWorld:
```bash
micromamba create -f environments/metaworld_sim.yml
micromamba activate metaworld_sim
```

For ManiSkill:
```bash
micromamba create -f environments/maniskill_sim.yml
micromamba activate maniskill_sim
```

### 2. Run an experiment

```bash
# List available benchmarks
python benchmarks/libero/src/run_libero_claude.py --list-benchmarks

# List tasks in a benchmark
python benchmarks/libero/src/run_libero_claude.py --list-tasks --benchmark libero_10

# Run a single task
python benchmarks/libero/src/run_libero_claude.py --experiment test --task-id 0

# Run all tasks in a benchmark
python benchmarks/libero/src/run_libero_claude.py --experiment full_eval --all --benchmark libero_90
```

#### Prompt templates

Each benchmark includes two prompt templates:
- `prompt_template_baseline.md` — minimal prompt (default)
- `prompt_template_with_coaching.md` — includes action space references, tips, and boilerplate

Use `--prompt-template` to select a template:

```bash
# ManiSkill with coaching tips
python benchmarks/maniskill/src/run_maniskill_claude.py --experiment exp_001 --task-id 0 \
    --prompt-template benchmarks/maniskill/src/prompt_template_with_coaching.md

# MetaWorld with coaching tips
python benchmarks/metaworld/src/run_metaworld_claude.py --experiment exp_001 --task-id 40 \
    --prompt-template benchmarks/metaworld/src/prompt_template_with_coaching.md
```

### 3. View results

Results are saved to `benchmarks/<benchmark>/data/<experiment>/`:
- `run_summary.json`: Overall experiment statistics
- `episode_N.py`: Generated control script for task N
- `episode_N.mp4`: Video recording of the episode
- `meta_N.json`: Task metadata (success, num_tries)

## Requirements

- Python 3.10+
- Claude Agent SDK (`pip install claude-agent-sdk`)
- Benchmark-specific packages (libero, metaworld, or mani-skill)
- ANTHROPIC_API_KEY environment variable set

## Citation

If you use this code in your research, please cite:

```bibtex
@article{faea2026,
  title={Frontier Agent as Embodied Agent},
  author={FAEA Authors},
  year={2026}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
