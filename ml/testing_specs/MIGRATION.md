# MIGRATION — Reusing the Testing Framework

> How to instantiate `ml/testing_specs/` for a new ML project.

The SENTINEL testing framework is designed to be **project-agnostic**. The
core abstractions — gates, probes, label quality, threshold sensitivity —
apply to any classification or multi-label model.

---

## The 3-step migration

### Step 1: Copy the framework

```bash
# From a new project root
mkdir -p testing_specs
cp -r /path/to/sentinel/ml/testing_specs/framework/* testing_specs/
cp /path/to/sentinel/ml/testing_specs/00_rules.md testing_specs/
cp /path/to/sentinel/ml/testing_specs/synthetic_probes.py testing_specs/
cp /path/to/sentinel/ml/testing_specs/label_quality.py testing_specs/
```

### Step 2: Customize the probes for your domain

```bash
# Edit testing_specs/synthetic_probes.py
# Replace the PROBES list with contracts from YOUR domain
# For example, for an image classifier:
PROBES = [
    {
        "id": "img_cat",
        "class": "Cat",
        "operator": ">",
        "threshold": 0.50,
        "description": "Image of a cat should classify as Cat",
        "source": "<path/to/test/cat.jpg>",
    },
    {
        "id": "img_dog",
        "class": "Cat",
        "operator": "<",
        "threshold": 0.30,
        "description": "Image of a dog should NOT classify as Cat",
        "source": "<path/to/test/dog.jpg>",
    },
    # ... more probes
]
```

### Step 3: Wire it into your promotion pipeline

```python
# In your promote_model.py equivalent:
from testing_specs.synthetic_probes import run_all_probes, summarize

def promote(checkpoint_path):
    # Existing F1 gate
    if not f1_gate_pass(checkpoint_path):
        return "BLOCKED: F1"
    
    # NEW: behavioral gate
    results = run_all_probes(checkpoint=checkpoint_path)
    summary = summarize(results)
    if not summary["all_passed"]:
        return f"BLOCKED: behavioral probes {summary['failed']} failed"
    
    return "OK"
```

---

## What stays the same across projects

- **00_rules.md** — Universal invariants (gate assertions, cross-checks,
  attestations, no floating findings)
- **The 4-rule framework** (read before claiming, validate your validation,
  no floating findings, procedures are not knowledge)
- **The gate pattern** — every check produces PASS/FAIL/UNVERIFIED + a written
  attestation
- **The promotion gate structure** (F1, calibration, behavioral, label
  quality, drift baseline)

## What changes per project

- **The probes** — replace SENTINEL's smart contracts with your domain
  examples (images, text, tabular rows, etc.)
- **The class names** — change `SENTINEL_CLASSES` to your classes
- **The label quality thresholds** — your data may have different priors
- **The model adapter** — `_CheckpointPredictor` needs to use your model
  loading code
- **The spec files (A–L)** — these are SENTINEL-specific. You may want
  to rename, replace, or remove some. The framework survives without them.

---

## Templates provided

In `framework/templates/`:

| Template | Use case |
|---|---|
| `sentinel_v2.yaml` | SENTINEL multi-label code vulnerability (the original) |
| `image_classification.yaml` | Vision projects (placeholder) |
| `text_classification.yaml` | NLP projects (placeholder) |
| `tabular_regression.yaml` | Tabular regression (placeholder) |

Each template has:
- List of gates to run
- Probe definitions (with placeholders for the user to fill in)
- Label quality thresholds
- Pass/fail criteria

To use a template:
```bash
python ml/testing_specs/framework/cli.py init --template <name> > my_project_gates.yaml
```

---

## What the framework gives you for free

- **30+ fixed regression probes** — any future model regression on these
  will be caught
- **Label quality audit** — catches over-labeled classes before training
- **Per-class threshold sensitivity** — finds poorly-calibrated classes
- **Cross-tool consistency** — flags if model is overfitting to tool output
- **Auto-reproducibility** — re-runs and compares with reference
- **Auto-floating-findings** — refuses session close if findings are unwritten

---

## What's specific to SENTINEL

- **Graph-based model** — `_CheckpointPredictor` uses PyTorch Geometric
- **CodeBERT** — tokenizer model name is hardcoded
- **Multi-label** — 10 vulnerability classes
- **Smart contracts** — synthetic probes are .sol files
- **Slither** — used for static analysis
- **AuditRegistry.sol** — on-chain storage of audit results

For a new project, replace the model adapter and probe format. The rest
of the framework is generic.

---

## Example: Adapting for a medical imaging classifier

```python
# testing_specs/synthetic_probes.py
PROBES = [
    {
        "id": "xray_normal",
        "class": "Pneumonia",
        "operator": "<",
        "threshold": 0.30,
        "description": "Normal chest X-ray should NOT classify as Pneumonia",
        "source": "/data/test/normal_xray_001.png",
    },
    {
        "id": "xray_pneumonia",
        "class": "Pneumonia",
        "operator": ">",
        "threshold": 0.70,
        "description": "Pneumonia X-ray should classify as Pneumonia",
        "source": "/data/test/pneumonia_xray_001.png",
    },
    # ... more
]

SENTINEL_CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19", ...]
```

```python
# testing_specs/_model_adapter.py (new file for your project)
import torch
from torchvision import models, transforms

class MedicalImagePredictor:
    def __init__(self, checkpoint_path):
        self.model = models.resnet50()
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
    
    def predict(self, payload):
        source = payload["source_code"]  # or "source"
        img = self._load_image(source)
        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
        return {
            "probabilities": dict(zip(SENTINEL_CLASSES, probs)),
            "label": "...",
        }
```

That's it. The rest of the framework (label quality, threshold sensitivity,
auto-reproducibility) is generic.
