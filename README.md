# FairMOP

**Benchmarking Fairness–Utility Trade-offs in Text-to-Image Models via Pareto Frontiers**

FairMOP is a modular, model-agnostic Python framework for evaluating fairness and quality in text-to-image (T2I) generative models. It frames the evaluation as a **multi-objective optimization problem (MOOP)**, automatically identifying Pareto-optimal configurations in the fairness–utility space.

---

## Overview

| Module | Responsibility |
|---|---|
| **Input Specifications** | Prompt templates, hyperparameter grids, protected attributes |
| **Generation** | Pluggable T2I backends (GPT-Image built-in; SD, SDXL, FLUX via plugins) |
| **Evaluation** | VLM-as-a-Judge demographic annotation + utility/fairness metrics |
| **Output** | Pareto frontier identification, interactive Streamlit dashboard, JSON/CSV export |

### Metrics

| Category | Metric | Description |
|---|---|---|
| Utility | CLIP Score | Cosine similarity between text prompt and image (ViT-L/14) |
| Utility | FID | Fréchet Inception Distance (Clean-FID + InceptionV3) |
| Utility | PRDC Precision | k-NN Precision (k=5) |
| Fairness | Shannon Entropy | Normalized entropy of demographic distribution (1.0 = uniform) |
| Fairness | KL Divergence | KL divergence from the uniform distribution |

---

## Installation

### Basic install (CPU, API-only models)

```bash
git clone https://github.com/Malta-Lab/FairMOP.git
cd FairMOP
pip install -e .
```

### Full install (all extras)

```bash
pip install -e ".[all]"
pip install git+https://github.com/openai/CLIP.git   # CLIP Score requires this
```

### Selective extras

```bash
pip install -e ".[openai]"          # GPT-Image / GPT-4o annotation
pip install -e ".[gemini]"          # Gemini annotation backend
pip install -e ".[clip]"            # CLIP Score (PyTorch + CLIP)
pip install -e ".[fid]"             # FID (Clean-FID)
pip install -e ".[prdc]"            # PRDC Precision
pip install -e ".[dashboard]"       # Streamlit + Plotly dashboard
```

### Environment variables

```bash
export OPENAI_API_KEY="sk-..."      # Required for GPT-Image generation & OpenAI VLM judge
export GOOGLE_API_KEY="AIza..."     # Required only for Gemini VLM judge
```

You can also place these in a `.env` file at the project root.

---

## Quick Start

### Option A – Python API

```python
from fairmop import ExperimentConfig, FairMOPPipeline
from fairmop.input_specs import quick_config

config = quick_config(
    concept="nurse",
    model_name="gpt-image",
    grid_params={"quality": ["low", "medium", "high"]},
    num_images=5,
    metrics=["clip_score", "entropy"],
    output_dir="./fairmop_output/quickstart",
    experiment_name="quickstart_nurse",
    vlm_provider="openai",
    vlm_model="gpt-4o-2024-05-13",
)

pipeline = FairMOPPipeline(config)
results = pipeline.run()

# results is a dict with keys: "results", "pareto", "export_paths"
print(f"Pareto-optimal configs: {len(results['pareto'].frontier)}")
```

### Option B – YAML config + CLI

```bash
python -m fairmop run --config experiments/quickstart.yaml
```

### Option C – Evaluate pre-existing images

```bash
python -m fairmop evaluate \
    --images ./my_images/ \
    --output ./eval_output/ \
    --vlm-provider openai \
    --vlm-model gpt-4o-2024-05-13
```

### Launch the dashboard

```bash
python -m fairmop dashboard                         # default port 8501
python -m fairmop dashboard --port 8502 --results ./fairmop_output/
```

---

## Skipping Generation — Use Pre-Generated Images

If you do not want to call a T2I API or run a local model, you can **download the
pre-generated images** from our HuggingFace dataset and jump straight to evaluation.

### Step 1 – Install the download dependency

```bash
pip install -e ".[datasets]"    # adds huggingface_hub
# or, if you already have requirements installed:
pip install huggingface_hub
```

### Step 2 – Download the dataset

Images are automatically placed in `fairmop_output/{model}_evaluation/images/`  
so they sit alongside generated results with no extra setup.

```bash
# Only SD images → fairmop_output/sd_evaluation/images/
python -m fairmop download --model sd

# Only SDXL images → fairmop_output/sdxl_evaluation/images/
python -m fairmop download --model sdxl

# Full dataset (all models)
python -m fairmop download

# See all available model names
python -m fairmop download --list-models

# Private repo / specific HF token
python -m fairmop download --model sd --token hf_...

# Override root output directory
python -m fairmop download --model sd --output /some/other/dir
```

After the download, the CLI prints ready-to-copy `evaluate` commands with all paths
already filled in.

### Step 3 – Evaluate

```bash
python -m fairmop evaluate \
    --images ./fairmop_output/sd_evaluation/images \
    --prompt "the face of a nurse" \
    --metrics clip_score entropy kl \
    --vlm-provider openai \
    --vlm-model gpt-4o-2024-05-13 \
    --output ./fairmop_output/sd_evaluation/sd_evaluation_results.json
```

Results (`_results.json`, `_summary.csv`, `_pareto.csv`) are all saved inside
`fairmop_output/sd_evaluation/`.

### Step 4 – Inspect results in the dashboard

```bash
python -m fairmop dashboard --results ./fairmop_output/flux_results.json
```

### Skip-generation via YAML (`--skip-generation`)

If you run a YAML pipeline but already have images in the expected output folder,
pass `--skip-generation` to avoid re-generating:

```bash
python -m fairmop run --config experiments/quickstart.yaml --skip-generation
```

---

## YAML Configuration Reference

```yaml
# ── Generation ─────────────────────────────────────────────
prompt: "the face of a nurse"
model_name: "gpt-image"         # registered generator name
model_params:                   # kwargs forwarded to the generator
  openai_model: "gpt-image-1"
  rate_limit_delay: 1.0

hyperparameter_grid:            # Cartesian-product search space
  quality: ["low", "medium", "high"]
  size: ["1024x1024", "1536x1024"]

num_images_per_config: 50       # images per (grid point × seed)
seed_start: 1

# ── Evaluation ─────────────────────────────────────────────
protected_attribute: "gender"   # primary attribute for Pareto
vlm_provider: "openai"          # "openai" or "gemini"
vlm_model: "gpt-4o-2024-05-13"
metrics:
  - clip_score
  - fid
  - entropy
  - kl

# ── Infrastructure ─────────────────────────────────────────
gpu_index: 0                    # null = CPU
output_dir: "./fairmop_output"
experiment_name: "nurse_benchmark"
```

---

## Architecture

```
fairmop/
├── __init__.py              # Package root (ExperimentConfig, FairMOPPipeline, GeneratorRegistry)
├── __main__.py              # CLI entry point: run · evaluate · dashboard
├── config.py                # ExperimentConfig & HyperparameterGrid dataclasses
├── input_specs.py           # Prompt templates, attribute definitions, quick_config()
├── pipeline.py              # FairMOPPipeline – orchestrates all 4 modules
├── utils.py                 # GPU detection, device helpers
├── generation/
│   ├── base.py              # BaseGenerator (ABC) – subclass this
│   ├── registry.py          # @GeneratorRegistry.register() decorator
│   ├── gpt_image.py         # Built-in GPT-Image & DALL·E 3 backend
│   └── custom.py            # Integration templates (SD 1.5, SDXL, FLUX)
├── evaluation/
│   ├── vlm_judge.py         # VLM-as-a-Judge demographic annotation
│   ├── fairness.py          # Shannon Entropy · KL Divergence
│   └── utility.py           # CLIP Score · FID · PRDC Precision
├── output/
│   ├── pareto.py            # Pareto frontier (dominance Algorithm 1) · hypervolume
│   └── export.py            # JSON / CSV export & import
└── dashboard/
    └── app.py               # Streamlit interactive dashboard with Plotly charts
```

### Pipeline Flow

```
Input Specs → Generation → Evaluation → Output
    │              │             │           │
  config     generate imgs   VLM judge   Pareto frontier
  prompts    per grid point  CLIP/FID    JSON/CSV export
  grid       save .png       Entropy/KL  Streamlit dashboard
```

---

## Integrating a Custom T2I Model

Subclass `BaseGenerator`, implement `generate()`, and register with the decorator:

```python
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

from fairmop.generation import BaseGenerator, GeneratorRegistry


@GeneratorRegistry.register("stable-diffusion-v1-5")
class SDv15Generator(BaseGenerator):
    """Stable Diffusion 1.5 via Hugging Face Diffusers."""

    def __init__(self, model_name="", device="cpu", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(device)

    def generate(
        self,
        prompt: str,
        seed: int,
        **hyperparams,    # receives grid params (guidance_scale, num_inference_steps, ...)
    ) -> Image.Image:
        generator = torch.Generator(self.device).manual_seed(seed)
        result = self.pipe(
            prompt,
            generator=generator,
            **hyperparams,
        )
        return result.images[0]
```

Then reference it in your YAML:

```yaml
model_name: "stable-diffusion-v1-5"
hyperparameter_grid:
  guidance_scale: [1.0, 3.5, 7.5]
  num_inference_steps: [20, 50]
```

See `fairmop/generation/custom.py` for more integration templates (SDXL, FLUX).

---

## Examples

| Script | Description |
|---|---|
| `examples/quickstart.py` | Minimal end-to-end run (3 configs × 5 images) |
| `examples/gpt_image_example.py` | Full GPT-Image benchmark with quality & size grid |
| `examples/custom_model_example.py` | Dummy generator showing the plugin pattern |
| `examples/analyze_results.py` | Load existing JSON results & compute Pareto frontier |

```bash
# Run an example
python examples/quickstart.py

# Or use a pre-built YAML experiment
python -m fairmop run --config experiments/full_benchmark.yaml
```

---

## Key API Reference

### `ExperimentConfig`

```python
from fairmop import ExperimentConfig

# From YAML
config = ExperimentConfig.from_yaml("experiments/quickstart.yaml")

# Programmatic
config = ExperimentConfig(
    prompt="the face of a firefighter",
    model_name="gpt-image",
    num_images_per_config=50,
    protected_attribute="gender",
    metrics=["clip_score", "entropy", "kl"],
)

config.total_images()  # total across all grid configurations
config.summary()       # human-readable summary string
config.to_yaml("out.yaml")
```

### `FairMOPPipeline`

```python
from fairmop import FairMOPPipeline

pipeline = FairMOPPipeline(config)

# Full pipeline (generate → evaluate → output)
results = pipeline.run()

# Evaluate only (images already exist)
results = pipeline.evaluate_only(image_dir="./images/")
```

### `GeneratorRegistry`

```python
from fairmop.generation import GeneratorRegistry

# List available backends
print(GeneratorRegistry.available())
# ['gpt-image', 'dall-e-3', ...]

# Instantiate by name
gen = GeneratorRegistry.create("gpt-image", device="cpu", api_key="sk-...")
```

### Fairness & Utility Metrics

```python
from fairmop.evaluation.fairness import compute_fairness_metrics
from fairmop.evaluation.utility import compute_clip_score, compute_fid

# Fairness
metrics = compute_fairness_metrics(
    annotations=[{"gender": "M"}, {"gender": "F"}, {"gender": "M"}, ...],
    primary_attribute="gender",
)
# → {"gender_entropy": 0.97, "gender_kl": 0.003, ...}

# CLIP Score
score = compute_clip_score(images, prompt="the face of a nurse", device="cuda:0")

# FID
fid = compute_fid(generated_dir="./gen_images/", reference_dir="./ref_images/")
```

### Pareto Frontier

```python
from fairmop.output.pareto import find_pareto_frontier, ConfigurationPoint

points = [
    ConfigurationPoint(config={"quality": "low"},  utility=0.28, fairness=0.95),
    ConfigurationPoint(config={"quality": "high"}, utility=0.34, fairness=0.72),
    # ...
]

result = find_pareto_frontier(points)
print(result.frontier)        # Pareto-optimal points
print(result.dominated)       # dominated points
print(result.hypervolume)     # area indicator
```

---

## Dashboard

The interactive Streamlit dashboard supports:

- **File upload** of JSON result files
- **Metric selection** (any fairness × utility combination)
- **Pareto frontier** overlay with Plotly interactive charts
- **Multi-model comparison** side-by-side
- **Export** results as JSON, CSV, or standalone HTML charts

```bash
python -m fairmop dashboard
```

---

## VLM-as-a-Judge

FairMOP uses a **fixed annotation prompt** (validated against human annotators) sent to a Vision-Language Model (GPT-4o or Gemini) that classifies each generated image into demographic categories:

| Attribute | Categories |
|---|---|
| Gender | Male, Female, Non-binary/Ambiguous |
| Ethnicity | White, Black, Asian, Latino/Hispanic, Middle-Eastern, Other |
| Age | Young (18-30), Middle-aged (31-55), Senior (56+), Ambiguous |

The prompt is fixed to ensure **reproducibility** across experiments — it is not user-customizable by design.

---

## Project Structure

```
FairMOP/
├── fairmop/                 # Core Python package
├── examples/                # Runnable Python examples
├── experiments/             # Pre-built YAML experiment configs
├── look_here/               # Reference implementation & methodology doc
│   └── metodologia.md       # Academic methodology description
├── pyproject.toml           # Package metadata & optional deps
├── requirements.txt         # Flat dependency list
├── setup.py                 # Editable install support
└── README.md
```

---

## Requirements

- Python ≥ 3.9
- Core: `numpy`, `Pillow`, `tqdm`, `pyyaml`, `python-dotenv`, `requests`
- CLIP Score: `torch`, `torchvision`, [OpenAI CLIP](https://github.com/openai/CLIP)
- FID: `torch`, `clean-fid`, `scipy`
- PRDC: `torch`, `prdc`, `scipy`
- VLM Judge: `openai` and/or `google-genai`
- Dashboard: `streamlit`, `plotly`, `pandas`, `kaleido`

---

## License

MIT

---

## Citation

If you use FairMOP in your research, please cite:

```bibtex
@software{fairmop2025,
  title   = {FairMOP: Benchmarking Fairness-Utility Trade-offs in
             Text-to-Image Models via Pareto Frontiers},
  author  = {FairMOP Team},
  year    = {2025},
  url     = {https://github.com/Malta-Lab/FairMOP},
}
```
