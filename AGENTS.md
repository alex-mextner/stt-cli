# stt-cli — agent guide

Speech-to-text for any audio or video file on macOS. Python 3.11+, `uv`, fully typed,
async. Zero required third-party runtime dependencies.

## Run it

```
./bin/stt --help
./bin/stt doctor                 # every dependency and how to install what is missing
./bin/stt rec.m4a -f txt
```

## Layout

| Module | Responsibility |
| --- | --- |
| `cli.py` | dispatcher; commands self-register from `commands/` |
| `commands/` | one file per subcommand (`NAME`, `SUMMARY`, `run(argv) -> int`) |
| `pipeline.py` | the stage order: VAD, chunk, decode, variants, clean, speakers, LLM, render |
| `media.py` | ffprobe/ffmpeg: probe, normalize, archive-encode, recording start time |
| `vad.py` | speech-span detection (Silero via whisper.cpp, or ffmpeg `silencedetect`) |
| `chunks.py` | splice speech into a few long decode chunks and map timestamps back |
| `backends/` | engines behind one `Backend` protocol (`whispercpp`, `mlx`) |
| `cleaning.py`, `phrases.py` | hallucination and repetition-loop scrubbing |
| `variants.py` | second-opinion decodings for low-confidence segments |
| `llm.py`, `postprocess.py` | LLM correction and structured summary via an installed agent CLI |
| `diarize.py` | optional pyannote speaker diarization |
| `hf.py`, `auth.py` | Hugging Face credentials and the `stt login` browser flow |
| `dictionary.py`, `fuzzy.py` | terminology: prompt biasing, exact fixes, phonetic flags |
| `archive.py` | content-addressed store plus a SQLite index |
| `formats.py`, `timestamps.py` | renderers and the three timestamp modes |
| `registry.py` | the model pool: one human name per model, per-engine ids, sizes |
| `resources.py` | disk and memory guards before any large download or write |

## Invariants

- **Import-clean at the top.** `cli.py` and every command module import, at module level,
  only the standard library and first-party modules that are themselves standard-library
  only. Third-party imports (`mlx_whisper`, `pyannote`, `torch`) happen inside the function
  that needs them, so `stt --help` works on a bare machine. The guarantee is about what a
  bare machine has installed, not about the shape of the internal import graph: `cli.py`
  imports every command module to build the help text, so a module-level `from .. import
  pipeline` is fine and a module-level `import torch` is not.
- **Zero required runtime dependencies.** Anything heavy is an optional extra installed on
  demand. Do not add a `dependencies` entry to `pyproject.toml` without a very good reason.
- **Every external process in the transcription path goes through `proc.run`.** It has a
  timeout, turns a missing binary into a diagnosed error, and is `async` so independent work
  overlaps. The deliberate exceptions are all synchronous, interactive and outside that
  path: the `pip install` in `commands/diarize_cmd.py` (streams progress to the user's
  terminal), the `open` in `commands/archive.py` (fire-and-forget Finder reveal), and the
  `urllib` calls plus `pbpaste` poll in `hf.py`/`auth.py` (a login is a conversation with a
  human, and `urllib` keeps the token out of an argument vector that `curl` would expose).
- **Every failure is an `SttError` subclass** with what/why/how and a stable exit code
  (see `_errors.py`). Never `sys.exit(1)` with a bare message.
- **Nothing durable in `/tmp`.** Audio and transcripts live in the archive under
  `STT_HOME` (default `~/Library/Application Support/stt-cli`). Only genuinely scratch
  intermediates use a temporary directory, deleted in the same run.
- **The cache key covers words, not presentation.** `archive.FINGERPRINT_KEYS` must list
  every setting that changes the transcript, and must not list output format or timestamp
  mode, because those re-render from the archive.
- **Cleaning never deletes silently.** Flag first, drop second, always report what went.
- **Every boolean CLI switch a stored preference can supply is a `BooleanOptionalAction`
  defaulting to `None`.** A `store_true` can only turn something on, which makes a stored
  preference impossible to override, and an unmentioned flag would overwrite a stored value
  with `False`. A switch that is only ever a decision about the single invocation in front
  of you — `--yes`, `--quiet`, `--force`, `--status` — has nothing stored behind it to
  overwrite and stays a plain `store_true`; the test is whether `config` can hold a value
  for it, not whether the type happens to be a boolean.
- **Render options come from the merged `Settings`, never straight from argv** — otherwise a
  stored preference is silently dead, and an enrichment left on an archived transcript (a
  summary, speaker labels) leaks into a run that did not ask for it.
- **Every file in the archive is written through `archive.write_atomic`** (or the ffmpeg
  `.part`-then-rename equivalent). A truncated transcript looks present forever.
- **A naive `recorded_at` is wall-clock time and must be localized, never converted.**
- **The dictionary corrects only what the user wrote down.** An `aka` spelling is a fact and
  is substituted; a phonetic near-match is a suspicion and is only ever flagged, for the
  reader and for the LLM pass. Never promote a near-match to a replacement.
- **Anything that changes the words has to be settled before the fingerprint is computed** —
  the dictionary digest and the comparison mode `--fix` implies, in `pipeline._resolve`;
  what the installed engines can actually do, in `_settle_engine_limits`. A value that is
  decided later is a value the cache does not know about, and the stale transcript gets
  served as if it had it.
- **The archive is asked before any engine is touched — for the un-settled key only.** A run
  stored by a fully capable machine is served without resolving, probing or downloading
  anything. A run whose key was narrowed by settling (an engine that could not pin the
  glossary, a cross-check that was missing, whisper.cpp's bought context budget) can only be
  found by a second lookup, after the probes that reproduce that key — so on those machines
  a cache hit does cost the probes, and `usable_cross_backends` will fetch a missing
  cross-check model on the way. That is the price of a key that tells the truth; if it ever
  needs to be cheaper, split "is this model obtainable" from "obtain it", do not move the
  answer back after the fingerprint.
- **`-mc 0` disables whisper.cpp's initial prompt entirely** (verified: two runs with and
  without `--prompt` were byte-identical). A carried glossary therefore has to buy itself a
  context budget — see `PROMPT_CONTEXT` in `backends/whispercpp.py`.
- **A credential is never printed, and never stored anywhere but the provider's own
  location.** `hf.token_source()` returns a path or a variable name, never a token, and the
  Hugging Face token goes to the file `huggingface_hub` reads so that pyannote is logged in
  too. A store private to stt would make stt the only thing that works.
- **Never decode one VAD span per engine invocation.** That was the first implementation and
  it reloaded a 1.6 GB model 431 times for one hour of audio. Speech is spliced into a few
  long chunks by `chunks.py` instead.

## Adding things

- **A command**: drop a module in `commands/` exposing `NAME`, `SUMMARY`, `run(argv)`.
  The dispatcher finds it; nothing else changes.
- **An engine**: implement the `Backend` protocol in `backends/`, add it to
  `backends.ORDER`, and give each model an id for it in `registry.MODELS`.
- **A model**: append a `ModelSpec` to `registry.MODELS`. The size field is load-bearing:
  `resources.py` uses it to refuse a download that will not fit.
- **A setting**: add the field to `config.Settings`, then decide deliberately whether it
  belongs in `archive.FINGERPRINT_KEYS` (it changes the words), in `ENRICHMENTS` (it adds to
  a finished transcript), or in neither (it only changes presentation). Bump
  `archive.DECODE_REVISION` when the decoding behaviour itself changes.
- **A hallucination phrase**: `phrases.ALWAYS` for unmistakable artefacts,
  `phrases.CONTEXTUAL` for ordinary speech that is only suspicious over silence. Users
  extend the first tier via `<STT_HOME>/hallucinations.txt`.

## Checks

```
uv run --extra dev ruff check .
uv run --extra dev mypy stt_cli
uv run --extra test pytest -q
```

Tests must not need a model, a GPU or a network. Anything that would is mocked.

## Writing reports for people (Telegram, PR bodies, spec summaries)

These are read by a human, not another agent, so optimize for being understood:

- Never invent abbreviations or compress terms into fragments. Write the full term.
- Prefer fewer points explained in full sentences over more points compressed into
  jargon. A short list of clear sentences beats a long list of cryptic stubs.
- Expand every non-obvious term at first use (name it in full, then abbreviate if you
  must reuse it).
- When a message exceeds the channel's length limit, cut secondary content — drop whole
  points — rather than compressing the wording of what remains.
