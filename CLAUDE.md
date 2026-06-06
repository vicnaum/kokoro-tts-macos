# KokoroVoice — project context for Claude Code

## What this is

Kokoro-82M neural TTS registered as **real macOS system voices** (Spoken
Content / Speak Selection / VoiceOver) via an Audio Unit speech synthesis
extension (`AVSpeechSynthesisProviderAudioUnit`, macOS 13+ API; we target
macOS 15 because KokoroSwift requires it). Everything runs locally on Apple
Silicon via MLX. Two voices wired up: `af_heart` ("Kokoro Heart", US female,
top-rated) and `am_michael` ("Kokoro Michael", US male).

## Current state (June 2026)

**Builds, signs (ad-hoc), registers, and synthesizes — verified working on
Xcode 26.3 / macOS 26 / Apple Silicon.** Both voices produce correct 24 kHz
neural audio through the real system speech path (`AVSpeechSynthesizer` →
extension → KokoroSwift/MLX), confirmed by capturing the PCM via
`AVSpeechSynthesizer.write` (24000 Hz, non-silent). Getting from the original
scaffold to a green build took one code fix, a dependency-graph re-pin, and a
one-line fork of KokoroSwift — see **Resolved build issues** below.

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

## Resolved build issues (June 2026)

What actually broke on Xcode 26.3 and how it's fixed. All fixes are persistent
(in `project.yml`, `Extension/`, or the KokoroSwift fork) — nothing lives only in
DerivedData.

1. **`KokoroTTS.Language` doesn't exist** — the enum is top-level in the module,
   so it's `KokoroSwift.Language`, not nested. Fixed in `KokoroAudioUnit.swift`.
2. **KokoroSwift's manifest under-declares `MLXFast`** — its sources
   `import MLXFast` but the target only depended on MLX/MLXNN/MLXRandom/MLXFFT.
   Implicit-module builds tolerate the transitive import via MLXNN; Xcode 16+/26
   **explicit-module** builds fail with `no such module 'MLXFast'`. Fixed via a
   one-line fork — `vicnaum/kokoro-ios @ mlxfast-fix` (commit pinned by `revision`
   in project.yml). `SWIFT_ENABLE_EXPLICIT_MODULES: NO` is also set as a belt-and-
   suspenders against other under-declared transitive imports.
3. **Dependency graph re-pinned to KokoroSwift's tested `Package.resolved`** — the
   old `MLXUtilsLibrary: branch: main` resolved to commit `66f7cd5` ("removed
   BenchmarkTimer"), which KokoroSwift still calls → `cannot find 'BenchmarkTimer'`.
   project.yml now pins MLXUtilsLibrary `0.0.6` and mlx-swift `0.29.1` (exact),
   the combination KokoroSwift 1.0.10 was built against. Do **not** float these.
4. **Metal Toolchain is a separate download in Xcode 26** — MLX compiles `.metal`
   shaders; first build fails with `cannot execute tool 'metal'`. One-time fix:
   `xcodebuild -downloadComponent MetalToolchain` (~700 MB).
5. **CLI signing** — automatic signing needs an Apple account `xcodebuild` can
   reach, and it can't ("No Account for Team"). For local runs, build ad-hoc (see
   recipe). For a distributable signed build, set your Team in Xcode and build there.
   The `NSExtension`/`ausp` AudioComponent dict (weak-spot we worried about) renders
   correctly from project.yml — no change needed.

Non-obvious runtime fact: the system **prepends the extension bundle id** to each
voice identifier, so the registered id is
`com.vicnaum.kokorovoice.extension.com.vicnaum.kokorovoice.af_heart`, not the bare
`com.vicnaum.kokorovoice.af_heart`. The host app's `hasPrefix` filter still matches
(the prefix is contained), but `AVSpeechSynthesisVoice(identifier:)` on the **bare**
id silently falls back to the default system voice — pick the voice object out of
`speechVoices()` instead of constructing it from the short id.

Non-obvious runtime fact #2 — **Spoken Content needs the app in `/Applications`**:
AVSpeechSynthesizer (apps, incl. our host app's Speak button) loads the extension
from anywhere, but the sandboxed Spoken Content daemons (`AXVisualSupportAgent` →
`MauiAUSP`) cannot load our `ausp/kkro/Vicn` audio unit out of
`~/Library/Developer/Xcode/DerivedData`. The tell (unified log): `findNext
ausp/kkro/Vicn -> ausp/kona/appl` then `CoreSynthesizer … retryFallbackVoice` →
you hear a robotic Apple voice. Fix: copy the built `.app` to `/Applications`,
`lsregister -f` it, and `killall AXVisualSupportAgent MauiAUSP` (or log out/in).

### Headless build + verify recipe (no Xcode GUI)

```bash
brew install xcodegen
xcodebuild -downloadComponent MetalToolchain      # once, Xcode 26+
./download_models.sh                              # once
xcodegen
xcodebuild -project KokoroVoice.xcodeproj -scheme KokoroVoice -configuration Debug \
  -destination 'platform=macOS' \
  CODE_SIGN_IDENTITY="-" CODE_SIGN_STYLE=Manual CODE_SIGNING_REQUIRED=YES \
  CODE_SIGNING_ALLOWED=YES DEVELOPMENT_TEAM="" PROVISIONING_PROFILE_SPECIFIER="" \
  SWIFT_ENABLE_EXPLICIT_MODULES=NO build
open <DerivedData>/.../Build/Products/Debug/KokoroVoice.app   # registers extension
pluginkit -m | grep -i kokoro                                 # confirm registration
```

First build compiles all of MLX from source (~10 min); later builds are fast.
First synthesis loads the 312 MB model in the extension (a few seconds).

## Distribution (Developer ID + notarization) — verified working

Shipping to another Mac needs a **paid Apple Developer Program** account (team
`HAVQAJTX35`; the `JP93BJRSQ8` cert is the free Personal Team and can't notarize).
Four non-obvious things, all now baked into `project.yml`:

1. **Embed the dynamic frameworks.** `KokoroSwift` (and `MLXUtilsLibrary`) are
   `type: .dynamic`; MLX/Cmlx/Numerics build dynamic alongside them. App
   extensions can't carry frameworks, so the **host app** embeds them
   (`KokoroVoice` target depends on `package: KokoroSwift` with `embed: true` —
   its closure pulls the other four; listing them too → "duplicate tasks"). The
   extension finds them via its `@executable_path/../../../../Frameworks` rpath.
   Without this the extension crashes at launch (`dyld: Library not loaded:
   @rpath/KokoroSwift.framework`). Debug builds *masked* this by loading the
   framework from DerivedData — it would never have run on another Mac.
2. **Release strips `get-task-allow`.** `configs.Release` sets
   `CODE_SIGN_INJECT_BASE_ENTITLEMENTS: NO`; otherwise the notary rejects the
   build ("The executable requests the com.apple.security.get-task-allow
   entitlement"). Debug keeps it (debuggable).
3. **Hardened runtime is fine for MLX** — no JIT entitlement needed; Metal
   compiles shaders out-of-process. (`voices.npz` triggers a benign notary
   *warning* — it's a zip of npy arrays, no executables.)
4. After swapping the `.app` in `/Applications` several times, `speechVoices()`
   can return 0 (stale AudioComponent cache) — `killall coreaudiod` refreshes it.

Recipe (one-time: create a "Developer ID Application" cert + a notary keychain
profile: `xcrun notarytool store-credentials KokoroNotary --apple-id <id>
--team-id HAVQAJTX35`):

```bash
xcodebuild -project KokoroVoice.xcodeproj -scheme KokoroVoice -configuration Release \
  -destination 'platform=macOS' \
  CODE_SIGN_IDENTITY="Developer ID Application: VIKTAR NAUMIK (HAVQAJTX35)" \
  CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM=HAVQAJTX35 PROVISIONING_PROFILE_SPECIFIER="" \
  ENABLE_HARDENED_RUNTIME=YES OTHER_CODE_SIGN_FLAGS="--timestamp" \
  SWIFT_ENABLE_EXPLICIT_MODULES=NO build
ditto -c -k --keepParent <Release>/KokoroVoice.app /tmp/k.zip
xcrun notarytool submit /tmp/k.zip --keychain-profile KokoroNotary --wait   # -> Accepted
xcrun stapler staple <Release>/KokoroVoice.app
spctl -a -vvv -t exec <Release>/KokoroVoice.app          # -> accepted / Notarized Developer ID
# package: stapled app + /Applications symlink into a UDZO dmg, then
# codesign --timestamp the dmg, notarytool submit it, stapler staple it.
```

Target Mac requirement: **Apple Silicon, macOS 15+** (Kokoro/MLX).

## Roadmap

**Done (v1.1.0):** streaming / sentence-chunked synthesis (Kokoro is non-
autoregressive with a hard 510-token limit, so whole-text synthesis threw
`tooManyTokens` on long selections — now chunked, hybrid first-sentence start,
oversized chunks re-split via the `tooManyTokens` catch); text normalization
in `SSML.swift` (symbols → words, numbers, line-breaks → sentence pauses,
verified with the `PhonemeDump` dev tool).

**Done (v1.2.0):** word highlighting — `emitWordMarkers` builds
`AVSpeechSynthesisMarker(.word)` from `MToken.start_ts` + a cursor-search that
maps each token back to its range in the original SSML, handed to the AU's
`speechSynthesisOutputMetadataBlock` (byteSampleOffset = frame × 4 for float32).
Verified emitting correctly via `AVSpeechSynthesizer`'s `willSpeakRange` delegate
(13/13 words, right ranges). **Caveat:** rendering is host-dependent — apps that
implement `willSpeakRange` highlight; macOS **Spoken Content** highlighting of
third-party voices is a longstanding, inconsistent Apple bug (not our code).

**Done (v1.3.0):** (a) **rate slider** — `SSML.speedMultiplier` parses
`<prosody rate="PERCENT">` and threads it to `generateAudio(speed:)`; the host
injects a stray invisible char into the rate value (`Float("12.5")` → nil), so
the value is digit/dot-filtered. Clamped to **[0.6, 1.4]** — below ~0.6× Kokoro
drags/slurs, above ~1.4× it drops words (the slider's raw 12.5–400% extremes are
unusable). (b) **parentheses → pauses** — Misaki speaks bracketed content but
drops the bracket chars with no pause, gluing the aside onto its neighbors;
`normalizeSymbols` now turns `()[]{}` into commas (verified via PhonemeDump).
(c) **streaming smoothness on slower Macs** — live playback underran on a slower
machine (synthesis fell behind playback → silence gaps/stutter, machine-
dependent; fine on a fast Mac). Fix: the live render block now **primes**
(buffers `livePrimeChunks = 2` chunks before starting, re-buffers after any
underrun → one clean pause instead of a stutter), and `maxCharBudget` dropped
400 → **160** so each chunk's synthesis finishes well within the previous chunk's
playback. Trade-off: a slightly longer delay before the first word. *Live priming
can't be verified from this (fast) Mac — needs testing on the slower machine; if
first-word latency is too high or it falls back to the robotic voice, lower
`livePrimeChunks` to 1.*

1. More voices: add entries to `VoiceManifest.swift` (try `af_bella`,
   `bf_emma` — British voices exercise the `.enGB` path). Wanted with an in-app
   **preview** UX (preview one sentence per voice).
2. Longer paragraph pauses: insert silence chunks between paragraphs in the AU.
3. Maybe: swap in a future emotive model (MisoTTS-8B is too big for an
   extension; Kokoro is the right size for this architecture).

**Dev tool — `PhonemeDump`** (`Tools/`, `type: tool` in project.yml): runs
`SSML.swift` + Misaki's real G2P and prints raw → normalized → phonemes, so you
can debug pronunciation and add normalization rules without listening. Build the
`PhonemeDump` scheme, then run it from the build products dir with
`DYLD_FRAMEWORK_PATH=<Build/Products/Debug>:<…/PackageFrameworks>` (it links the
dynamic MLX/Misaki frameworks). `--raw` skips normalization.

## Conventions

- Don't commit model files or the generated `.xcodeproj` (see `.gitignore`).
- Voice identifiers: `com.vicnaum.kokorovoice.<npz key without .npy>` — the
  app's status UI filters on this prefix; keep them in sync.
