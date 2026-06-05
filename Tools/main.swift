//
//  PhonemeDump  (dev tool, not shipped)
//
//  Runs our SSML/text normalization + Misaki's English G2P (the exact
//  phonemizer Kokoro uses) and prints raw -> normalized -> phonemes, so we can
//  SEE pronunciation without listening. Add `--raw` to skip normalization.
//
//  Usage:
//    PhonemeDump "the cap is ~400 chars"
//    printf 'multi\nline' | PhonemeDump
//

import Foundation
import MisakiSwift

var args = Array(CommandLine.arguments.dropFirst())
let skipNormalization = args.first == "--raw"
if skipNormalization { args.removeFirst() }

let raw: String
if args.isEmpty {
    raw = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) ?? ""
} else {
    raw = args.joined(separator: " ")
}

let normalized = skipNormalization ? raw : SSML.plainText(from: raw)
let g2p = EnglishG2P(british: false)
let (phonemes, tokens) = g2p.phonemize(text: normalized)

print("=== RAW ===\n\(raw)\n")
if !skipNormalization { print("=== NORMALIZED ===\n\(normalized)\n") }
print("=== PHONEMES ===\n\(phonemes)\n")
print("=== TOKENS (text -> phonemes) ===")
for token in tokens {
    print("  \(token.text.debugDescription)  ->  \(token.phonemes ?? "·(none)")")
}
