//
//  VoiceManifest.swift
//  KokoroVoiceExtension
//
//  Central place to declare which Kokoro voices the extension registers
//  with the system. Add a new entry here (and make sure the key exists in
//  voices.npz) to ship more voices — the model is shared between all of them.
//

import AVFoundation

struct KokoroVoiceDefinition {
    /// Key inside voices.npz (note the ".npy" suffix).
    let npzKey: String
    /// Name shown in System Settings → Spoken Content.
    let displayName: String
    /// Unique, stable identifier for the system voice.
    let identifier: String
    /// BCP-47 language code.
    let language: String
}

enum KokoroVoiceManifest {
    static let identifierPrefix = "com.vicnaum.kokorovoice."

    /// Kokoro naming: a/b = US/UK English, f/m = female/male.
    static let voices: [KokoroVoiceDefinition] = [
        KokoroVoiceDefinition(
            npzKey: "af_heart.npy",
            displayName: "Kokoro Heart",
            identifier: identifierPrefix + "af_heart",
            language: "en-US"
        ),
        KokoroVoiceDefinition(
            npzKey: "am_michael.npy",
            displayName: "Kokoro Michael",
            identifier: identifierPrefix + "am_michael",
            language: "en-US"
        ),
    ]

    static func definition(forIdentifier identifier: String) -> KokoroVoiceDefinition? {
        voices.first { $0.identifier == identifier }
    }

    static func providerVoices() -> [AVSpeechSynthesisProviderVoice] {
        voices.map { def in
            AVSpeechSynthesisProviderVoice(
                name: def.displayName,
                identifier: def.identifier,
                primaryLanguages: [def.language],
                supportedLanguages: [def.language]
            )
        }
    }
}
