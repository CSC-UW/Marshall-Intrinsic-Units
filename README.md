Code for Marshall et al., _Intrinsic Units: Identifying a system’s causal grain_. 

Installation:
```
cd path/to/Marshall-Intrinsic-Units
uv venv
uv sync --dev
```

Run all code from the project's root (the directory containing `pyproject.toml`):
```
uv run scripts/run_min_micro.py
...
```

The following options in `pyphi_config.yml`have been set to non-default values:
```
PROGRESS_BARS: false
SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI: true
WELCOME_OFF: true
```