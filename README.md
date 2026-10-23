# lora-lab

A local lab for evaluating LoRAs with ComfyUI: it renders a balanced grid,
scores each image on separate axes, and ranks candidates with confidence
intervals instead of a single hand-assigned average.

The subject can be anything a LoRA is trained for — a person, an object, a
product, a style. Nothing in the tool assumes characters.

> **Status:** rendering and persistence work; the scoring core (subject
> calibration, attribute adherence, defect/health metrics) and the ranking
> statistics are implemented and tested. The model-backed judge and the
> acceptance UI are the next stage; see [Roadmap](#roadmap).

## Why

Averaging a handful of 1–5 ratings mostly measures which cells happened to be
rendered how often. Two seeds of the same LoRA can differ by a full point, and
one scenario can be systematically harder than another. lora-lab instead:

- **balanced grid** — every combination of LoRA, epoch, strength, scenario and
  seed, with the *same* seeds across all LoRAs
- **idempotency** — every job has a hash over all render-relevant parameters; an
  interrupted run resumes where it stopped
- **separate axes** — subject fidelity, prompt adherence, defects, stability,
  robustness; never one blended grade
- **confidence** — ranking uses a lower confidence bound, not a mean

## Quick start

```bash
git clone <repo> lora-lab && cd lora-lab
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# the committed config is an example; put your real one in config/local/
mkdir -p config/local
cp config/subjects.yaml config/scenarios.yaml config/grid.yaml config/local/
$EDITOR config/local/*.yaml

.venv/bin/lora-lab doctor          # bindings, grid, ComfyUI
.venv/bin/lora-lab plan            # job count + time estimate
.venv/bin/lora-lab generate --limit 8
.venv/bin/lora-lab serve --host 0.0.0.0 --port 8099
```

`doctor` works without a running ComfyUI: widget mapping falls back to a
built-in table for the MOSS nodes.

## Configuration

| File | Contents |
|---|---|
| `config/project.yaml` | ComfyUI URL, data directory, optional local output mount |
| `config/subjects.yaml` | trigger word, reference and training folders |
| `config/scenarios.yaml` | prompts + binary attributes (the contract with the judge) |
| `config/grid.yaml` | LoRAs, epochs, strengths, seeds, node bindings, render profiles |

Files in `config/local/` override the committed examples, so a clone runs out of
the box and a real installation keeps its own prompts out of git.

### Bindings

`grid.yaml` addresses workflow nodes by id. For the bundled MOSS workflow:

```yaml
bindings:
  lora_node: 11      # the LoRA under test
  positive: {id: 4, input: text}
  negative: {id: 5, input: text}
  seed:     {id: 7, input: seed}
  save:     {id: 9, input: filename_prefix}
  size_node: 6
  drop_nodes: [12, 13]   # note + prompt-list helper
```

The runner sets node modes from the binding rather than from node titles, and
records the resulting prompt. A genuine bypass (`mode = 4`) is resolved by
reconnecting the model chain, which also gives a real `base_only` baseline.

### Render profiles

`fast` is the everyday path (8 steps, CFG 1, distilled speed LoRA). `reference`
is the quality reference (30 steps, CFG 4, `res_multistep`, no distilled LoRA).
A ranking measured only on the fast path can be distorted by the distillation,
so the top candidates are re-checked on the reference profile.

## Commands

```
lora-lab doctor                 validate config, workflow bindings, ComfyUI
lora-lab plan                   expand the grid, estimate duration
lora-lab generate [--limit N]   render (idempotent, resumable)
lora-lab status                 progress from SQLite
lora-lab score [--identity]     run scorers over stored images
lora-lab score --vlm-url URL --vlm-model NAME
                                attribute verdicts from a local VLM
lora-lab report [--json]        rank LoRAs from stored scores
lora-lab check-refs             verify references are held out from training
lora-lab discover               scan /datasets and /models/loras, print config
lora-lab import-fizgig /fizgig  list training runs and their settings
lora-lab discover               scan /datasets and /models/loras, print config
lora-lab import-fizgig /fizgig  list training runs and their settings
lora-lab audit PATH             dataset audit (resolutions, captions, duplicates)
lora-lab import-fizgig FILE     training parameters from a .safetensors header
lora-lab serve                  web dashboard
```

## Scoring

Each image gets separate axes, never one blended grade:

| Axis | Source | Missing data |
|---|---|---|
| `identity` | embedding similarity to the subject's reference centroid, calibrated to P(same subject) | `None` when no face/reference, never 0 |
| `adherence` | weighted share of satisfied attributes from the scenario checklist | `None` when nothing could be judged |
| `quality` | model-free health (resolution × sharpness); an aesthetic scorer can replace it | `None` if unreadable |

The calibration is fitted on genuine and impostor pairs from your own reference
sets, so a raw cosine of 0.45 means the same thing in every run. Attribute
verdicts come from a local VLM over an OpenAI-compatible endpoint (vLLM,
llama.cpp, Ollama, LM Studio) or from a JSON file (`--verdicts`) for offline or
human review rounds. Refusals and malformed answers are counted per run, not
raised.

Ranking renormalises over the axes that exist, averages per scenario first (so a
heavily sampled scenario cannot dominate), and ranks by a two-level cluster
bootstrap lower bound. `report` also prints seed stability and each LoRA's two
weakest scenarios.

## Trainer metadata

Fizgig (like kohya) stores training parameters in the safetensors JSON header as
`ss_*` keys. `import-fizgig` reads them without torch:

```bash
lora-lab import-fizgig /path/lora-000066.safetensors --json
```

Rank, LR, optimizer, epochs, trigger, bucket info and `lownoise_pct` can be
imported instead of typed in by hand.

## Unraid / Docker

```bash
cd docker
docker compose up -d          # pulls ghcr.io/lum1nat0r/lora-lab:latest
```

The image is built by `.github/workflows/docker.yml` on every push to `main`.
An empty `config` mount is seeded from the examples baked into the image, so a
fresh deployment starts without copying files first.

The host port is set with `LORA_LAB_PORT` (default 8199); the container always
listens on 8099. `COMFYUI_URL` points at the existing ComfyUI container. When its output folder
is mounted (`COMFY_OUTPUT_DIR`), images are read from disk instead of over HTTP.
The mount contract and the container paths for subjects, references and
datasets are in [docs/PATHS.md](docs/PATHS.md).

## Architecture

```
config/                    experiment plan (YAML); local/ is uncommitted
workflows/                 ComfyUI exports (UI and API format)
src/lora_lab/
  config.py                pydantic models + loader
  comfy/workflow.py        UI -> API, bindings, bypass, hashing
  comfy/client.py          /prompt, /history, /view, /free
  grid.py                  job expansion
  runner.py                rendering, resume, persistence
  db.py                    SQLite (runs, jobs, images, scores)
  audit.py                 dataset audit
  importers/fizgig.py      safetensors metadata
  web/                     FastAPI dashboard
```

## Roadmap

1. **Judge calibration** — validate the VLM against a human-labelled gold set
   and gate the automated ranking on agreement.
2. **Defect judge** — semantic defects (extra limbs, text, ignored clothing)
   from the same VLM, severity weighted.
3. **Acceptance** — blind A/B tournament with Bradley-Terry.
4. **Loop** — trigger a new training run, reuse the identical grid, compare
   paired.

## License

MIT
