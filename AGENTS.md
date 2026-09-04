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
| `live/` | `stt mic`: microphone, gate, two persistent models, and the keyboard |

## Live dictation (`live/`)

`stt mic` is the one part of the tool that writes outside its own process, so it has
invariants of its own on top of the ones below.

| module | what it is |
| --- | --- |
| `quartz.py`, `tap.py` | the four macOS calls, through `ctypes`: type, backspace, watch, ask |
| `capture.py` | the microphone as PCM, via the ffmpeg that is already required |
| `gate.py` | where one utterance ends, so silence never reaches a model |
| `server.py` | a `whisper-server` process: a model that stays loaded between questions |
| `typist.py` | what is on screen, what is ours, and the smallest edit between the two |
| `session.py` | the two-speed loop; `dictation.py` wires it up, `status.py` reports |

- **stt deletes only characters stt typed, and only while it still owns them.** Ownership
  ends the instant the user presses a key or clicks — see `Typist.disown`, which deliberately
  neither tidies the half-finished draft nor appends the finished sentence after it.
- **One writer, and it is the event loop.** The draft pass and the accurate pass write to the
  same caret. Typing was on a thread until it was measured — two hundred backspaces cost
  three milliseconds — and a thread meant two writes could interleave, with the cancel that
  was supposed to prevent it unable to stop a thread it had only cancelled the await of.
  `Session._put` is synchronous, and that is what makes one-at-a-time a guarantee.
- **The draft pass and the accurate pass never overlap.** They decode different audio; if
  their answers could arrive in either order the text would assemble itself backwards.
  Drafting stops while a sentence settles, and a sentence starting during that must not make
  the typist forget what is on screen — see `Typist.begin`.
- **A key code never leaves `tap.py` for anything but a comparison.** A system-wide key-down
  tap is a keylogger with a different name, and the only defence that holds is that nothing
  writes one to a log, a file or the archive.
- **Never call `quartz.type_text` or `press_backspace` outside `stt mic` without first
  focusing a throwaway document.** They type into the frontmost window of whoever is at the
  machine; the per-session marker only tells our own tap to ignore them. A benchmark of the
  posting cost once went into a browser window somebody was reading. Everything about WHAT
  gets typed is testable through `typist.py` and a fake keyboard, which is where it belongs.
- **Silence is never handed to a model, and the thresholds come from measurements.**
  `gate.py` is what `vad.py` is for a file, and it matters more here: an invented sentence in
  a file is a line to delete, and an invented sentence here is typed into somebody's window.
  The numbers in it were taken from recordings of this machine's own microphone — a quiet
  room at a median RMS of 16 with transients to 1014, speech at 4550 — and re-tuning them
  means re-measuring, not guessing. `live/meter.py` is how you re-measure: it runs the real
  capture through the real gate and prints levels, and `stt mic --check` is that module with
  a terminal attached. Note what the measurements say about the limit of the approach: a
  keystroke reaches 1014 and a quiet voice 812, so LEVEL alone cannot separate them and
  duration has to. `THRESHOLD_MINIMUM` (the bar a frame must clear) and `FLOOR_MINIMUM` (a
  clamp that only keeps the learned floor off exact zero) are deliberately two constants —
  they were one, and the single name put the effective bar at six times its documented value.
- **One session at a time, and it ends by itself.** Two `stt mic` processes type into the
  same window and each deletes what IT believes it wrote, which is by then interleaved with
  the other's — an exclusive lock in `dictation._the_only_session` refuses the second, before
  it loads a model rather than after. And a session nobody has spoken into for half an hour
  stops: a microphone left open is a room being recorded, and the clicks an empty room makes
  do not count as somebody speaking (see `Session._end`).
- **Nothing that holds a resource is left holding it.** Two models in memory, a recording
  light, a system-wide event tap and a lock file, all released on every path out including a
  server that never became ready. Both subprocesses have their output drained continuously,
  because a pipe nobody reads fills up and then blocks the writer — the microphone or the
  model stopping mid-session with nothing said anywhere.
- **Two things that look like free wins and are not, both measured:** `audio_ctx` made
  `large-v3-turbo` four times slower, and `no_speech_prob` answered 0.000 for room noise it
  had just invented words out of. The notes are in `server.py`; do not reach for either
  again without new measurements.

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

**Check a fix by removing it and watching its test fail.** Five tests in `test_live.py` passed
whether or not the code they named was there — they asserted on the machinery rather than on
the path through it, which is the shape this mistake takes every time. A test that survives
the deletion of its own fix is guarding nothing, and the only way to find out is to try.

**Read the exit code of the command, not of the pipe.** `pytest -q | tail -3` reports
`tail`'s status, which is always zero, and the truncation hides the "N passed" line that would
have contradicted it. Four runs were reported here as passing on that basis while a test was
deadlocking; CI then hung for forty minutes on both Python versions and found it instead.
Redirect to a file, print `$?` immediately, and look at the summary line:
`pytest -q > /tmp/out.txt 2>&1; echo "PYTEST=$?"; tail -2 /tmp/out.txt`.

**A lock the tests can reach twice on one thread must be an `RLock`.** `Typist._keyboard`
guards a race between the event tap's thread and the loop's, so a plain `Lock` closes it —
and deadlocks the suite, which simulates a click mid-edit by calling `disown` from inside a
fake keyboard's keystroke, on the same thread. Production never reenters; the tests must.

**Clear `__pycache__` before believing a result you are surprised by.** Python decides a
cached `.pyc` is current from the source's size and mtime, and a rewrite that lands inside the
same clock tick at the same size does not always invalidate it. That cost half an hour here:
`inspect.getsource` showed the new code while the interpreter ran the old bytecode, so a test
failed against a function whose source was plainly correct — and, worse, the mutation checks
before it had been running against a mixture. `find . -name __pycache__ -not -path "./.venv/*"
-exec rm -rf {} +` before a mutation run, and any time a result contradicts the file.

## Writing reports for people (Telegram, PR bodies, spec summaries)

These are read by a human, not another agent, so optimize for being understood:

- Never invent abbreviations or compress terms into fragments. Write the full term.
- Prefer fewer points explained in full sentences over more points compressed into
  jargon. A short list of clear sentences beats a long list of cryptic stubs.
- Expand every non-obvious term at first use (name it in full, then abbreviate if you
  must reuse it).
- When a message exceeds the channel's length limit, cut secondary content — drop whole
  points — rather than compressing the wording of what remains.
