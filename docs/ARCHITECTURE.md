# Architecture Overview

This repository now follows a modular layout with clear separation between
configuration, data loading, model construction, training orchestration, and
evaluation. The key layers are:

## Configuration (`src/config/`)
Experiment, model, data, physics, and contrastive options are expressed as
`dataclass` schemas. The top-level `ExperimentConfig` is used across CLI tools,
training, and evaluation to provide a single source of truth for hyper-params
and runtime options.

## Data Pipeline (`src/data/datamodule.py`)
`FiLMDataModule` encapsulates dataset creation, splitting, randomised graph
variants, and dataloader construction. All training/evaluation entrypoints use
this module to guarantee consistent preprocessing.

## Models (`src/models/`)
Reusable building blocks live in `components/` (shared encoders, heads, MCR
support). `factory.py` exposes a single `create_model` helper that instantiates
the correct `Concat` or `Enhanced` variant based on `ModelConfig`.

## Training (`src/training/`)
- `trainer.py` converts configs into concrete objects (model, loss, data) and
  invokes the trainer engine.
- `trainer_engine.py` implements the `Trainer` class coordinating optimisation,
  validation, early stopping, and testing.
- `callbacks/` provide lifecycle hooks; the default stack includes a
  diagnostics saver and a console logger. Additional callbacks can be registered
  for checkpointing or custom logging.
- `loss_factory.py` translates physics configuration into the appropriate loss
  object (bulk or spatial).

## Evaluation (`src/evaluation/comprehensive_evaluation.py`)
The comprehensive evaluation script now reuses the same model factory,
datamodule, and trainer engine to compute test metrics before running coverage
or cross-metabolite analyses.

## Tests (`tests/`)
Unit tests cover config aliasing, component builders, callbacks, datamodule
splits, and a trainer smoke test. Tests that rely on PyTorch can be enabled via
`RUN_TORCH_TESTS=1`.

## CLI Usage
`scripts/training/train_concat_smiles.py` accepts CLI flags or an experiment
config file, constructs an `ExperimentConfig`, and delegates to
`train_experiment`. Example:

```bash
python scripts/training/train_concat_smiles.py \
  --config experiments/baseline.yaml --seed 42 --normalised
```

To evaluate a saved model:

```bash
python src/evaluation/comprehensive_evaluation.py --model-path trained_models/.../best_model.pt
```

Both commands honour the shared configuration flow and emit
`experiment_config.json` next to outputs.

## Design Principles & Coding Standards

To ensure long-term maintainability and flexibility, all code contributions must adhere to the following principles:

### 1. Modularity & Decoupling
*   **Single Responsibility:** Each module, class, and function should have one clear purpose. Avoid "God objects" that handle data loading, modeling, and training simultaneously.
*   **Interface-Based Design:** Components should interact through well-defined interfaces (e.g., config dataclasses, abstract base classes) rather than direct implementation dependencies.
*   **Dependency Injection:** Pass dependencies (like encoders, loss functions) into constructors rather than instantiating them inside classes. This facilitates testing and component swapping.
*   **Separation of Concerns:**
    *   **Logic vs. I/O:** Core mathematical logic (GNN layers, loss calculations) must be separated from file I/O and CLI parsing.
    *   **Config vs. Code:** Hyperparameters must be injected via `src/config` objects, never hardcoded.

### 2. Extensibility
*   **Factory Patterns:** Use factories (e.g., `src/models/factory.py`) to instantiate components based on configuration. This allows adding new model variants without modifying existing training loops.
*   **Composition over Inheritance:** Prefer composing complex models from smaller, reusable modules (in `src/models/modules/`) over deep inheritance hierarchies.

### 3. Testing
*   New components must be independently testable.
*   Avoid global state that makes unit testing difficult.
