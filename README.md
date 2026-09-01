# stt

Speech-to-text on macOS for **any** audio or video file — without the ffmpeg incantation,
without the hallucinated subtitle credits, and without paying for the same transcription
twice.

```bash
stt meeting.m4a                                  # text on stdout
stt talk.mp4 -f srt -o subs/                     # subtitles, straight from a video
stt call.ogg -t absolute --tz Europe/Belgrade    # wall-clock timestamps
stt meeting.m4a --fix --summary -f md            # corrected transcript + structured summary
stt meeting.m4a --diarize -f speakers            # split into speaker turns
```

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/alex-mextner/stt-cli/main/install.sh | bash
stt setup       # detects what you already have, explains the options, picks an engine
```

The installer adds `ffmpeg` and a speech engine if they are missing, and registers `stt`
as an agent skill so coding agents find it on their own. `stt doctor` reports every
dependency and how to install anything absent.

## What it actually fixes

**The command line.** `ffmpeg -i "$N.ogg" -ar 16000 -ac 1 -c:a pcm_s16le "$N.wav" &&
whisper-cli -f "$N.wav"` breaks the moment a filename has a space or a comma in it, only
accepts a couple of containers, and leaves you holding a WAV you did not want. `stt <file>`
takes any container, any codec, audio or video, and any filename.

**Hallucinations over silence.** Whisper-family models are trained to always produce text.
Hand one thirty seconds of room tone and it does not stay quiet — it emits whatever followed
silence in its training data, which was largely subtitle files. That is where
`Субтитры сделал DimaTorzok`, `Продолжение следует…`, `Thanks for watching!` and
`Subtitles by the Amara.org community` come from in a recording of your own meeting.

`stt` runs voice-activity detection first and splices out the silence, so the model is never
shown the audio it would invent over. Whatever still gets through meets a second filter with
two tiers: unmistakable artefacts go unconditionally, and ordinary phrases that happen to be
the model's favourite filler are only dropped when the evidence agrees — low confidence, or
sitting over a stretch the detector called silent. Nothing is deleted silently; every removal
is reported, and `--no-clean` flags without dropping.

**Repetition loops.** A decoder that falls into a cycle repeats a phrase until the chunk
ends. Detected structurally — a repeated word group, in any language — rather than from a
word list, and collapsed.

**Doing the work twice.** Every run is stored in a local archive keyed by the audio's content
and the settings that affect the words. Asking for a different format, or timestamps you did
not want the first time, re-renders from the archive in milliseconds. `stt archive ls`.

## Output

| Format | What it is |
| --- | --- |
| `txt` | plain text, optionally with timestamps, speakers, confidence |
| `md` | a document: source metadata, summary if present, transcript |
| `json` | everything — segments, per-segment confidence, alternatives, flags |
| `srt` `vtt` | subtitles |
| `csv` `tsv` | one row per segment, for a spreadsheet |
| `speakers` | dialogue, one block per speaker turn |
| `summary` | the structured summary on its own |

`-f txt,srt` for several, `-f all` for the usual set.

### Timestamps

`--timestamps none` (default), `relative` (`1:02:05` from the start), or `absolute`
(`2026-03-31 14:35:02`). Absolute time is anchored to when the recording was made, worked out
from the container's own `creation_time` tag, a date in the filename, or the file's creation
time on disk — in that order, and the output says which one it used. Override it with
`--recorded-at`, and pick a zone with `--tz Europe/Belgrade`.

A date read out of a filename is treated as wall-clock time, not as an instant: a file named
`talk-2026-03-22 19.51.58.ogg` says 19:51 no matter which timezone your laptop is in today.
A `creation_time` tag *is* a real instant, so that one is converted into the display zone.

## Variants and confidence

Every segment carries the decoder's confidence. Where that confidence is low, `stt` can go
back and decode just that moment again:

```bash
stt rec.m4a --variants 2 --show-variants          # extra decodings of the shaky parts
stt rec.m4a --variant-model large-v3              # cross-check against a different model
```

A second decoding at a higher temperature shows what else the same model considered. A second
*model* is stronger evidence: two sizes of one model tend to make the same mistake, so their
agreement proves little, while agreement across architectures is real corroboration. Only
low-confidence segments are re-decoded, worst first, up to a cap — clean audio costs nothing
extra.

`--show-variants` prints the alternatives with their scores:

```
[0:14:22] (0.41) и тогда мы решили сделать бридж
    ├ alt (0.38, whispercpp:large-v3-turbo@t0.4): и тогда мы решили сделать бренч
    ├ alt (0.52, whispercpp:large-v3): и тогда мы решили сделать бридж-бот
```

## Correction and summary

```bash
stt rec.m4a --fix                  # an LLM cleans up the transcript
stt rec.m4a --summary -f md        # headline, topics, decisions, actions, open questions
stt rec.m4a --fix --text both      # print the corrected text and the original side by side
```

These borrow an agent CLI you already have logged in — `codex`, `claude` or `opencode` — so
there is no second API key and no second bill. Pick one with `--fix-with`.

The correction pass is given each segment's confidence and every alternative reading, not
just the text. That is the difference between a model editing uniformly and guessing, and a
model leaving the confident parts alone while choosing between real candidates where the
speech model was unsure. Enabling `--fix` therefore gathers variants automatically, whether
or not you asked to see them. The speech model's original wording is always kept as a
variant, so `--text raw` gets it back without re-running anything.

## Speakers

```bash
stt diarize install         # ~2.5 GB, only when you want it
stt rec.m4a --diarize -f speakers
```

Speaker diarization uses `pyannote.audio`, which means PyTorch and a gated Hugging Face
model. It is not installed by default and never downloads anything without telling you the
size first.

## Archive

```bash
stt archive ls                        # everything transcribed, newest first
stt archive show <run-id> -f vtt      # re-render an old run, no GPU involved
stt archive path <run-id> --open      # reveal the run's folder in Finder
stt archive usage                     # what it all costs on disk
stt archive gc --older-than 180       # dry run; add --yes to actually delete
```

Audio is kept as mono Opus at 24 kbit/s — about eleven megabytes an hour — so keeping every
recording is affordable. Everything lives under `~/Library/Application Support/stt-cli`, or
wherever you point `STT_HOME`.

Because the archive holds its own copy, a recording stays usable after you move or delete the
original: `stt old-recording.m4a -f srt` finds the archived audio by path and carries on.

Adding a summary or speaker labels to something already transcribed costs only that pass —
those are additions rather than changes, so they are not part of the cache key:

```bash
stt meeting.m4a                # transcribe (minutes)
stt meeting.m4a --summary      # cache hit; runs only the summary (seconds)
```

## Defaults

If every run of yours is `-l ru -t absolute --tz Europe/Belgrade`, those are not options,
they are your defaults:

```bash
stt config set language ru
stt config set timestamps absolute
stt config set timezone Europe/Belgrade
stt config list
```

A flag still wins over a stored value for a single run — in both directions. Every switch has
a negative twin, so a stored default can always be turned off for one run: `--no-fix`,
`--no-summary`, `--no-diarize`, `--no-clean`, `--no-cache`, `--no-keep-media`,
`--no-show-variants`.

## Models and engines

```bash
stt models ls          # the pool, with sizes, languages and rough quality/speed
stt models pull large-v3
```

Two engines, one interface. **whisper.cpp** runs natively on Metal and brings the Silero
voice-activity detector plus per-token confidence. **mlx-whisper** installs with `uv` in
about a minute with no compiler and downloads its own models. `stt setup` looks at what you
have, explains the trade-off, and remembers your answer; `--backend` overrides it per run.

Adding a model is one entry in `stt_cli/registry.py`, and adding an engine is one class — so
if something better than Whisper shows up, it is an addition rather than a rewrite. Downloads
check free disk space and physical memory first, and refuse rather than fail at 94 percent.

## Exit codes

`0` ok · `2` bad usage · `4` unknown model/format/command · `5` missing file · `7` network ·
`8` permission or missing token · `9` not enough disk or memory · `10` engine failure ·
`127` missing dependency.

## Requirements

macOS (Apple Silicon for the MLX engine), Python 3.11+, `ffmpeg`, and one speech engine.
Nothing else: the package itself has zero third-party runtime dependencies.

MIT.
