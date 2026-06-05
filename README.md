# KokoroVoice — Kokoro TTS as a macOS Spoken Content voice

Registers [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) neural voices
as real system voices via an Audio Unit speech synthesis extension
(`AVSpeechSynthesisProviderAudioUnit`). Once enabled, they work with
**Spoken Content** (Speak Selection), **VoiceOver**, and any app using
`AVSpeechSynthesizer`. Everything runs locally on Apple Silicon via MLX.

**Voices included:** Kokoro Heart (`af_heart`, US female — top-rated) and
Kokoro Michael (`am_michael`, US male). Add more in
`Extension/VoiceManifest.swift` — one entry per voice, the model is shared.

## Download (no build required)

Grab the notarized installer from
**[Releases](../../releases/latest)** → open `KokoroVoice.dmg` → drag
**KokoroVoice** into Applications → launch it once → then
System Settings → Accessibility → Spoken Content → System voice →
**Kokoro Heart** / **Kokoro Michael**. Requires Apple Silicon + macOS 15+.

The rest of this README is for building from source.

## Requirements

- Apple Silicon Mac, macOS 15.0+
- Xcode 16+
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`
- Free Apple ID is enough for local signing

## Setup

```bash
cd KokoroVoice
chmod +x download_models.sh
./download_models.sh      # fetches model (~600MB) + voices (~15MB) into Resources/
xcodegen                  # generates KokoroVoice.xcodeproj
open KokoroVoice.xcodeproj
```

In Xcode:

1. Select the **KokoroVoice** target → Signing & Capabilities → set your **Team**.
   Repeat for the **KokoroVoiceExtension** target.
2. Build & run the **KokoroVoice** scheme (this registers the extension).
3. In the app, hit **Refresh** — you should see "2 Kokoro voice(s) registered".
   Click **Speak** to test. **The first synthesis takes several seconds** (model
   load); after that it's faster than real-time.
4. Enable system-wide: System Settings → Accessibility → Spoken Content →
   **System voice** → pick *Kokoro Heart* or *Kokoro Michael*.
   Turn on *Speak selection* to get the read-aloud hotkey (default ⌥⎋).

## How it works

```
App (KokoroVoice)                  ← host; registers extension, test UI
└── PlugIns/KokoroVoiceExtension.appex
    ├── Info.plist                 ← AudioComponents type "ausp" (speech synthesizer)
    ├── AudioUnitFactory           ← NSExtensionPrincipalClass
    ├── KokoroAudioUnit            ← AVSpeechSynthesisProviderAudioUnit subclass
    │     speechVoices             ← advertises voices from VoiceManifest
    │     synthesizeSpeechRequest  ← SSML → plain text → KokoroSwift → PCM buffer
    │     internalRenderBlock      ← streams buffer to system, signals completion
    └── Resources: kokoro-v1_0.safetensors + voices.npz
```

Dependencies (SPM, resolved automatically): [KokoroSwift](https://github.com/mlalma/kokoro-ios)
(MLX Swift port of Kokoro) and [MLXUtilsLibrary](https://github.com/mlalma/MLXUtilsLibrary)
(reads `voices.npz`).

## Adding more voices

1. Open `Extension/VoiceManifest.swift`, add e.g.:
   ```swift
   KokoroVoiceDefinition(npzKey: "af_bella.npy", displayName: "Kokoro Bella",
                         identifier: identifierPrefix + "af_bella", language: "en-US"),
   ```
2. Check the key exists in the npz (28 voices ship in it):
   `python3 -c "import zipfile; print(zipfile.ZipFile('Resources/voices.npz').namelist())"`
3. Rebuild. British voices (`b*` prefix) automatically use the `en-GB` G2P path.

## Troubleshooting

- **Voices don't appear in the app / System Settings** — the extension
  registers on app launch; quit and relaunch once. Still nothing: move
  KokoroVoice.app to /Applications, launch it, then log out/in. Check
  registration with: `pluginkit -m | grep kokoro`
- **First "Speak" is silent for a while** — normal: the 600MB model loads on
  the first request inside the extension. Subsequent requests are fast.
- **Synthesis fails / no audio at all** — check Console.app for messages
  containing `KokoroVoice:`. Most likely the model files didn't make it into
  the extension bundle: verify both files sit in `Resources/` *before* running
  `xcodegen`, and that they appear under "Copy Bundle Resources" of the
  extension target.
- **SPM resolution conflict on MLXUtilsLibrary** — pin it to the exact
  revision in KokoroSwift's `Package.resolved` instead of `branch: main` in
  `project.yml`.
- **Memory** — expect ~1GB RAM in the extension process while speaking
  (fp32 model + MLX runtime). Fine on your 24-36GB machine.

## Known limitations (v1)

- English only (Misaki G2P). Other Kokoro languages need eSpeak-NG G2P.
- Speech rate/pitch sliders in Spoken Content are ignored (SSML prosody is
  stripped). Could be mapped to KokoroSwift's speed parameter later.
- No word highlighting yet — KokoroSwift 1.0.8 exposes per-token timestamps,
  so `AVSpeechSynthesisProviderOutputBlock` marker events are a natural next
  step.
- The whole utterance is synthesized before playback starts (no streaming),
  so very long selections have a noticeable lead-in.

## License & credits

MIT — see [LICENSE](LICENSE). Built on:

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — the neural TTS model (Apache-2.0)
- [KokoroSwift](https://github.com/mlalma/kokoro-ios) — MLX Swift port of Kokoro (MIT).
  This project pins a [one-line fork](https://github.com/vicnaum/kokoro-ios/tree/mlxfast-fix)
  that adds the missing `MLXFast` package dependency.
- [MLX Swift](https://github.com/ml-explore/mlx-swift) and
  [MLXUtilsLibrary](https://github.com/mlalma/MLXUtilsLibrary) (MIT)
- Render pattern from [bocoup/apple-custom-speech-synthesizer](https://github.com/bocoup/apple-custom-speech-synthesizer)
