//
//  SSML.swift
//  KokoroVoiceExtension
//
//  Turns the system's SSML speech request into clean plain text for Kokoro.
//  Beyond stripping markup it does light **text normalization** so the Misaki
//  G2P pronounces things correctly:
//
//   - symbols that Misaki drops or mis-guesses are spelled out
//     (~400 -> "approximately 400", 90% -> "90 percent", A → B -> "A to B",
//      & -> "and"); without this `~400` phonemizes to "axes".
//   - line/paragraph structure becomes sentence breaks: each line gets terminal
//     punctuation so Kokoro intonates and pauses instead of gluing
//     "heading\nThen" into one mispronounced word.
//
//  Tip: the `PhonemeDump` dev tool runs this + the real G2P so you can verify
//  pronunciation changes without listening.
//

import Foundation

enum SSML {
    static func plainText(from ssml: String) -> String {
        var text = ssml

        // 1. Drop SSML/XML tags.
        text = text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)

        // 2. Decode the common XML entities (do this before symbol rules so
        //    "&amp;" -> "&" -> "and").
        let entities: [(String, String)] = [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
            ("&quot;", "\""), ("&apos;", "'"), ("&#39;", "'"),
        ]
        for (entity, character) in entities {
            text = text.replacingOccurrences(of: entity, with: character)
        }

        // 3. Spell out symbols Misaki can't read.
        text = normalizeSymbols(text)

        // 4. Turn line/paragraph structure into sentence breaks (pauses +
        //    intonation), then collapse the remaining whitespace.
        text = normalizeLineBreaks(text)
        text = text.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Symbol normalization

    /// Replace symbols that the G2P drops or sends to its neural guesser with
    /// their spoken-word equivalents. Verified against `PhonemeDump`.
    private static func normalizeSymbols(_ input: String) -> String {
        var text = input

        func regex(_ pattern: String, _ replacement: String) {
            text = text.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
        }

        // Number-aware rules first.
        regex(#"~\s*(?=\d)"#, "approximately ")            // ~400 -> approximately 400
        regex(#"(?<=\d)\s*[–—-]\s*(?=\d)"#, " to ")          // 400–500 / 3-5 -> 400 to 500
        regex(#"(?<=\d)\s*%"#, " percent")                  // 90% -> 90 percent
        regex(#"(?<=\d)\s*°"#, " degrees")                  // 20° -> 20 degrees

        // Plain symbol -> word. Multi-character arrows before single chars.
        let literal: [(String, String)] = [
            ("-->", " to "), ("->", " to "), ("=>", " to "),
            ("→", " to "), ("⇒", " to "), ("←", " from "),
            ("&", " and "), ("%", " percent"), ("@", " at "),
            ("×", " times "), ("÷", " divided by "), ("±", " plus or minus "),
            ("°", " degrees "), ("=", " equals "), ("≈", " approximately "),
            ("•", ", "), ("·", " "), ("~", " "),   // stray tilde / bullets
            ("*", " "), ("`", " "), ("|", " "),    // markdown / code noise
        ]
        for (from, to) in literal {
            text = text.replacingOccurrences(of: from, with: to)
        }
        return text
    }

    // MARK: - Structure → sentence breaks

    /// Each non-empty line becomes its own sentence: trailing terminal
    /// punctuation is preserved, otherwise a period is added so Kokoro pauses
    /// and applies sentence intonation instead of gluing lines together. Blank
    /// lines (paragraph breaks) are collapsed — the per-line periods already
    /// create the pauses via the synthesizer's sentence chunking.
    private static func normalizeLineBreaks(_ input: String) -> String {
        let terminalPunctuation: Set<Character> = [".", "!", "?", "…"]
        var lines: [String] = []
        for rawLine in input.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            if let last = line.last, terminalPunctuation.contains(last) {
                lines.append(line)
            } else {
                // Drop a dangling soft separator (,;:) before adding the period.
                let trimmed = line.replacingOccurrences(of: "[,;:]+$", with: "", options: .regularExpression)
                lines.append(trimmed + ".")
            }
        }
        return lines.joined(separator: " ")
    }
}
