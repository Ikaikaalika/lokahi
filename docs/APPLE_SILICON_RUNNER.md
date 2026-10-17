# Self-hosted Apple Silicon runner

GitHub-hosted macOS runners are virtual machines: they compile Metal code but
cannot execute Metal compute or reach the Neural Engine. Hardware evidence
therefore comes from a Mac you control, registered as a self-hosted GitHub
Actions runner. The `apple-silicon` workflow
([`.github/workflows/apple-silicon.yml`](../.github/workflows/apple-silicon.yml))
then builds Lōkahi, checks the Metal engine against the NumPy oracle, runs
the native probes, and optionally runs the scoreboard: the pinned models
against a fresh MLX-LM control and the recorded best, posted on the branch's
pull request ([improvement loop](IMPROVEMENT_LOOP.md)). Every run is
serialized so GPU, ANE, and storage measurements never overlap.

## 1. Lock down the repository first

This repository is public. A pull request from a fork can add its own
workflow that targets `self-hosted`, so before registering the runner:

1. **Settings → Actions → General → Fork pull request workflows from outside
   collaborators:** select **Require approval for all outside collaborators**.
   Never approve a workflow run from a fork that touches `.github/`.
2. **Settings → Actions → General → Workflow permissions:** keep the default
   read-only `GITHUB_TOKEN`.

The `apple-silicon` workflow itself only runs on manual dispatch and on pushes
to `claude/**` branches, which require write access. The self-hosted job's
token is read-only. The scoreboard comment is posted by a separate job on a
GitHub-hosted runner. That job holds `pull-requests: write` and only reads
the uploaded Markdown.

## 2. Prepare the Mac

Use a dedicated standard (non-admin) macOS account, for example `lokahi-ci`,
so jobs cannot read your personal files or keychain.

```sh
xcode-select --install                      # or a full Xcode
xcodebuild -downloadComponent MetalToolchain # Xcode 26+: only needed for xcrun metal
brew install cmake                          # or the cmake.org installer
/usr/bin/python3 -m pip install --user numpy pytest mlx==0.32.4 mlx-lm==0.32.2
```

The pinned MLX versions match the ladder's MLX-LM baseline
(`benchmarks/targets/commoncompute_m1_model_ladder_v1.json`), which tracks the
current MLX release: the control Lōkahi has to beat. Bump both together. If you use a
different interpreter, set the repository variable `LOKAHI_PYTHON` to its path.

## 3. Register the runner

In **Settings → Actions → Runners → New self-hosted runner**, pick **macOS** and
**ARM64**, then run the commands GitHub shows from the `lokahi-ci` account.
When `config.sh` asks for labels, add:

```text
lokahi,apple-m1
```

Use the chip of that Mac for the second label (`apple-m4-max`, and so on). Run
it as a service so it survives logouts:

```sh
./svc.sh install
./svc.sh start
```

If Metal or ANE probes fail only under the service, stop it and run `./run.sh`
in a logged-in session to compare.

## 4. Enable the workflow

Create these repository variables (**Settings → Secrets and variables →
Actions → Variables**):

| Variable | Value | Purpose |
|---|---|---|
| `LOKAHI_HW_RUNNER` | `enabled` | Turns the hardware job on; without it the job is skipped |
| `LOKAHI_PYTHON` | optional interpreter path | Defaults to `/usr/bin/python3` |
| `LOKAHI_MODEL_ROOT` | approved SSD directory | Scoreboard only; Hugging Face cache layout |
| `LOKAHI_SCOREBOARD_ON_PUSH` | `enabled` | Optional: run the scoreboard on every push to `claude/**`, not only on manual benchmark runs |

`LOKAHI_MODEL_ROOT` must be on an SSD you have approved for model storage. It
must never be `/Volumes/Tyler HDD`; the workflow refuses that path.

## 5. Stage models (scoreboard only)

The workflow never downloads. Stage pinned snapshots yourself, into the
approved directory, using the Hugging Face cache layout:

```sh
export HF_HUB_CACHE="$LOKAHI_MODEL_ROOT"
huggingface-cli download mlx-community/gemma-3-1b-it-qat-4bit \
  --revision 15fed4eafb456c6fcb2a1165f19ac609670ed14b
```

The scoreboard matrix is `qwen3-0.6b`, `llama-3.2-1b` and `gemma-3-1b`; a
model that is not staged is reported as "not staged" rather than failing the
run. The pinned repositories and revisions are in
[`benchmarks/targets/commoncompute_m1_model_ladder_v1.json`](../benchmarks/targets/commoncompute_m1_model_ladder_v1.json).

## 6. Run it

**Actions → apple-silicon → Run workflow**, choose `correctness` or
`benchmark` (optionally narrowing the model list). Results are uploaded as the
run's `apple-silicon-evidence-*` artifact:

- the environment record and probe JSON;
- under `scoreboard/`, each model's benchmark, MLX-LM control and profile,
  plus `scoreboard.json`, `scoreboard.md` and `promote/`.

The scoreboard also appears in the job summary and as a comment on the
branch's open pull request. A regression, nondeterministic tokens, or a model
whose benchmark failed makes the run fail after the evidence is uploaded.
Promote a new best by committing its record from `promote/`
([how](../benchmarks/results/scoreboard/README.md)).

Keep the Mac idle and on power during benchmark runs. Record anything unusual
in the evidence notes.

## Evidence labels

| Step | Evidence kind |
|---|---|
| `tests/test_native_engine.py` with `LOKAHI_TEST_BACKENDS=cpu,metal` | Correctness (generated fixtures vs NumPy oracle) |
| `tests/test_mlx_crosscheck.py` | Correctness (oracle vs MLX and MLX-LM) |
| Native Metal and ANE probes | Hardware qualification of single operators |
| Scoreboard benchmarks and MLX-LM controls | Hardware evidence for each pinned model, chip, OS build, and workload |
| Scoreboard attribution (`lokahi profile`) | Attribution only: per-section time with synchronization between sections |

Common Compute fleet runs (other chips) use the same evidence format; see
[`ROADMAP.md`](ROADMAP.md).
