# KokoroVoice — project context for Claude Code

## What this is

Kokoro-82M neural TTS registered as **real macOS system voices** (Spoken
Content / Speak Selection / VoiceOver) via an Audio Unit speech synthesis
extension (`AVSpeechSynthesisProviderAudioUnit`, macOS 13+ API; we target
macOS 15 because KokoroSwift requires it). Everything runs locally on Apple
Silicon via MLX. Two voices wired up: `af_heart` ("Kokoro Heart", US female,
top-rated) and `am_michael` ("Kokoro Michael", US male).

## Current state (June 2026)

Scaffold written by Claude in Cowork mode from API research, **never compiled**
— expect minor build fixes, that's the immediate job. One git commit exists.

- `project.yml` — XcodeGen manifest, two targets: `KokoroVoice` (SwiftUI host
  app) and `KokoroVoiceExtension` (app extension, embedded). The `.xcodeproj`
  is generated, not committed.
- `App/` — host app: registers the extension by being launched; verification
  UI (lists registered Kokoro voices via `AVSpeechSynthesisVoice.speechVoices()`,
  speaks test text through `AVSpeechSynthesizer`).
- `Extension/` — the actual synthesizer:
  - `KokoroAudioUnit.swift` — `AVSpeechSynthesisProviderAudioUnit` subclass.
    Whole-utterance synthesis in `synthesizeSpeechRequest`, playback via
    `internalRenderBlock` (bocoup-sample pattern), 24kHz mono float32.
  - `VoiceManifest.swift` — declarative voice list; add entries to ship more
    voices (model is shared; each voice = one ~0.5MB embedding in voices.npz).
  - `SSML.swift` — regex SSML→plain-text (KokoroSwift takes plain text).
  - `AudioUnitFactory.swift` — `NSExtensionPrincipalClass`.
- `Resources/` — **empty until you run `./download_models.sh`** (fetches
  `kokoro-v1_0.safetensors` ~600MB + `voices.npz` ~15MB, both from the
  KokoroTestApp repo, guaranteed compatible with the pinned library).

## Build & verify

```bash
./download_models.sh        # once
xcodegen                    # regenerate after any project.yml change
open KokoroVoice.xcodeproj  # set signing Team on BOTH targets
```

Run the `KokoroVoice` scheme → app window → "Refresh" should show 2 voices →
"Speak" (first synthesis loads the model, takes seconds). Then System
Settings → Accessibility → Spoken Content → System voice → Kokoro Heart.

Debugging: extension logs are prefixed `KokoroVoice:` (Console.app or
`log stream --predicate 'eventMessage CONTAINS "KokoroVoice"'`).
Extension registration check: `pluginkit -m | grep -i kokoro`.

## Verified API facts (from upstream sources — don't re-derive)

From [mlalma/KokoroTestApp](https://github.com/mlalma/KokoroTestApp)
`TestAppModel.swift` (authoritative usage of KokoroSwift 1.0.8):

- `KokoroTTS(modelPath: URL)` — init (a `g2p:` param exists with default `.misaki`)
- `try engine.generateAudio(voice: MLXArray, language: .enUS | .enGB, text: String)`
  returns tuple `([Float], tokenArray?)`; tokens carry `text`, `start_ts`,
  `end_ts` (word-highlighting material)
- Sample rate constant: `KokoroTTS.Constants.samplingRate` (24000) — the AU
  hardcodes 24_000; consider referencing the constant once it compiles
- Voices: `NpyzReader.read(fileFromPath:)` (from `MLXUtilsLibrary`) →
  `[String: MLXArray]`, keys like `"af_heart.npy"`; npz ships 28 English voices
- Kokoro naming: `a*`/`b*` = US/UK English, `f`/`m` = gender; pick `.enUS` vs
  `.enGB` by prefix

From [bocoup/apple-custom-speech-synthesizer](https://github.com/bocoup/apple-custom-speech-synthesizer)
(working AU speech extension sample):

- Extension Info.plist: `NSExtensionPointIdentifier: com.apple.AudioUnit`,
  `AudioComponents` entry with `type: ausp`, 4-char `subtype`/`manufacturer`
  (ours: `kkro`/`Vicn`), `tags: [Speech Synthesizer]`, `sandboxSafe: true`
- Render pattern: fill output from a pre-synthesized `AVAudioPCMBuffer`, set
  `actionFlags = .offlineUnitRenderAction_Complete` when exhausted

## Known weak spots (most likely build failures)

1. **MLXUtilsLibrary pinned to `branch: main`** in project.yml — if SPM
   resolution conflicts with kokoro-ios's pin, use the exact revision from
   kokoro-ios `Package.resolved`.
2. Exact spelling of `KokoroTTS.Language` cases and the tuple shape of
   `generateAudio` — confirmed from the test app, but the library moves fast;
   check against the pinned version's source in
   `~/Library/Developer/Xcode/DerivedData/.../SourcePackages/checkouts/kokoro-ios`.
3. `AVSpeechSynthesisProviderVoice` init signature in `VoiceManifest.swift`
   (taken from bocoup sample, older SDK — verify against current SDK).
4. xcodegen rendering of the nested `NSExtension` dict in project.yml —
   inspect the generated `Extension/Info.plist` after first `xcodegen` run.
5. MLX inside a sandboxed extension (Metal shader compilation) — believed
   fine; if synthesis dies silently, test the same code path inside the host
   app first to isolate.

## Roadmap (after it builds)

1. Word highlighting: emit marker events using KokoroSwift token timestamps.
2. Map Spoken Content rate slider: parse `<prosody rate>` from the SSML
   instead of stripping it (KokoroSwift has a speed parameter).
3. Streaming: synthesize sentence-by-sentence so long selections start faster.
4. More voices: add entries to `VoiceManifest.swift` (try `af_bella`,
   `bf_emma` — British voices exercise the `.enGB` path).
5. Maybe: swap in a future emotive model (MisoTTS-8B is too big for an
   extension; Kokoro is the right size for this architecture).

## Conventions

- Don't commit model files or the generated `.xcodeproj` (see `.gitignore`).
- Voice identifiers: `com.vicnaum.kokorovoice.<npz key without .npy>` — the
  app's status UI filters on this prefix; keep them in sync.
