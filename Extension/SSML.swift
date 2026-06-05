//
//  SSML.swift
//  KokoroVoiceExtension
//
//  The system hands speech requests to the extension as SSML.
//  KokoroSwift takes plain text, so we strip the markup. (A later version
//  could honor <prosody rate=...> by adjusting the engine speed.)
//

import Foundation

enum SSML {
    static func plainText(from ssml: String) -> String {
        var text = ssml

        // Drop tags.
        text = text.replacingOccurrences(
            of: "<[^>]+>",
            with: " ",
            options: .regularExpression
        )

        // Decode the common XML entities.
        let entities: [(String, String)] = [
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&quot;", "\""),
            ("&apos;", "'"),
            ("&#39;", "'"),
        ]
        for (entity, character) in entities {
            text = text.replacingOccurrences(of: entity, with: character)
        }

        // Collapse whitespace.
        text = text.replacingOccurrences(
            of: "\\s+",
            with: " ",
            options: .regularExpression
        )

        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
