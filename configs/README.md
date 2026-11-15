# Configs

YAML files parsed into validated pydantic models (`lase.utils.config`). Unknown keys are
rejected. Planned layout:

- `data/`: dataset paths, split manifest, sensor/frame calibration
- `filter/`: baseline noise, gating, initialisation
- `sim/`: simulator scenarios
- `training/`: optimiser, chunking, loss weights
- `experiments/`: complete experiment definitions composed from the above
