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
| `archive.py` | content-addressed store plus a SQLite index |
| `formats.py`, `timestamps.py` | renderers and the three timestamp modes |
| `registry.py` | the model pool: one human name per model, per-engine ids, sizes |
| `resources.py` | disk and memory guards before any large download or write |

## Invariants

- **Import-clean at the top.** `cli.py` and every command module import only the standard
  library at module level. Heavy or optional imports (`mlx_whisper`, `pyannote`, `torch`)
  happen inside the function that needs them, so `stt --help` works on a bare machine.
- **Zero required runtime dependencies.** Anything heavy is an optional extra installed on
  demand. Do not add a `dependencies` entry to `pyproject.toml` without a very good reason.
- **Every external process in the transcription path goes through `proc.run`.** It has a
  timeout, turns a missing binary into a diagnosed error, and is `async` so independent work
  overlaps. The two deliberate exceptions are synchronous, interactive and outside that path:
  the `pip install` in `commands/diarize_cmd.py` (streams progress to the user's terminal)
  and the `open` in `commands/archive.py` (fire-and-forget Finder reveal).
- **Every failure is an `SttError` subclass** with what/why/how and a stable exit code
  (see `_errors.py`). Never `sys.exit(1)` with a bare message.
- **Nothing durable in `/tmp`.** Audio and transcripts live in the archive under
  `STT_HOME` (default `~/Library/Application Support/stt-cli`). Only genuinely scratch
  intermediates use a temporary directory, deleted in the same run.
- **The cache key covers words, not presentation.** `archive.FINGERPRINT_KEYS` must list
  every setting that changes the transcript, and must not list output format or timestamp
  mode, because those re-render from the archive.
- **Cleaning never deletes silently.** Flag first, drop second, always report what went.
- **Every boolean CLI switch is a `BooleanOptionalAction` defaulting to `None`.** A
  `store_true` can only turn something on, which makes a stored preference impossible to
  override, and an unmentioned flag would overwrite a stored value with `False`.
- **Render options come from the merged `Settings`, never straight from argv** — otherwise a
  stored preference is silently dead, and an enrichment left on an archived transcript (a
  summary, speaker labels) leaks into a run that did not ask for it.
- **Every file in the archive is written through `archive.write_atomic`** (or the ffmpeg
  `.part`-then-rename equivalent). A truncated transcript looks present forever.
- **A naive `recorded_at` is wall-clock time and must be localized, never converted.**
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
