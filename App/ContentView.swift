//
//  ContentView.swift
//  KokoroVoice
//
//  Minimal verification UI: shows whether the system sees the Kokoro voices
//  and speaks a test sentence through AVSpeechSynthesizer (the same path
//  Spoken Content uses).
//

import AVFoundation
import SwiftUI

struct ContentView: View {
    private static let identifierPrefix = "com.vicnaum.kokorovoice."

    /// Shown in the title and header so it's obvious which build is installed
    /// (read from the bundle, so it always matches the actual app).
    private static var appVersion: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String ?? "?"
        return "v\(short) (\(build))"
    }

    @State private var kokoroVoices: [AVSpeechSynthesisVoice] = []
    @State private var selectedIdentifier: String = ""
    @State private var text = """
    Kokoro is a neural text to speech model that runs entirely on your Mac. It produces natural sounding speech without any internet connection. This is a longer passage on purpose, so that playback lasts long enough for any stutter or breakup to happen. Keep listening through to the end of this sentence, and the next one too. That should be plenty of audio to capture what is going wrong.
    """

    @State private var diagText = ""
    @State private var capturing = false
    @State private var diagNote = ""

    private let synthesizer = AVSpeechSynthesizer()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("Kokoro Voice")
                    .font(.largeTitle.bold())
                Text(Self.appVersion)
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            GroupBox("Status") {
                HStack {
                    Image(systemName: kokoroVoices.isEmpty ? "xmark.circle.fill" : "checkmark.circle.fill")
                        .foregroundStyle(kokoroVoices.isEmpty ? .red : .green)
                    Text(kokoroVoices.isEmpty
                         ? "No Kokoro voices registered yet. Try Refresh; if it persists, see the README troubleshooting section."
                         : "\(kokoroVoices.count) Kokoro voice(s) registered with the system.")
                    Spacer()
                    Button("Refresh") { reloadVoices() }
                }
                .padding(4)
            }

            GroupBox("Test synthesis") {
                VStack(alignment: .leading, spacing: 12) {
                    Picker("Voice", selection: $selectedIdentifier) {
                        ForEach(kokoroVoices, id: \.identifier) { voice in
                            Text(voice.name).tag(voice.identifier)
                        }
                    }
                    .disabled(kokoroVoices.isEmpty)

                    TextEditor(text: $text)
                        .font(.body)
                        .frame(height: 90)
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))

                    HStack {
                        Button("Speak") { speak() }
                            .keyboardShortcut(.defaultAction)
                            .disabled(kokoroVoices.isEmpty)
                        Button("Stop") { synthesizer.stopSpeaking(at: .immediate) }
                        Spacer()
                        Text("First synthesis loads the model — expect a delay.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(4)
            }

            GroupBox("Diagnostics — send this to help fix the breakup") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("1. Press Speak above and let it play (even if it breaks up).\n2. Click Capture diagnostics.\n3. Click Copy and paste it into a message, or Save to Desktop and send the file.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack {
                        Button(capturing ? "Capturing…" : "Capture diagnostics") { captureDiagnostics() }
                            .keyboardShortcut("d", modifiers: [.command])
                            .disabled(capturing)
                        Button("Copy") { copyDiagnostics() }
                            .disabled(diagText.isEmpty)
                        Button("Save to Desktop") { saveDiagnostics() }
                            .disabled(diagText.isEmpty)
                        Spacer()
                        if !diagNote.isEmpty {
                            Text(diagNote).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    if !diagText.isEmpty {
                        ScrollView {
                            Text(diagText)
                                .font(.system(.caption2, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(6)
                        }
                        .frame(height: 150)
                        .background(RoundedRectangle(cornerRadius: 4).fill(.quaternary.opacity(0.3)))
                        .overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))
                    }
                }
                .padding(4)
            }

            GroupBox("Enable in Spoken Content") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("System Settings → Accessibility → Spoken Content → System voice — Kokoro voices appear in the voice picker once registered.")
                    Button("Open Spoken Content Settings") {
                        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.universalaccess?TextToSpeech") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                }
                .padding(4)
            }

            Spacer()
        }
        .padding(24)
        .navigationTitle("Kokoro Voice \(Self.appVersion)")
        .onAppear { reloadVoices() }
    }

    private func reloadVoices() {
        kokoroVoices = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.identifier.hasPrefix(Self.identifierPrefix) }
        if selectedIdentifier.isEmpty {
            selectedIdentifier = kokoroVoices.first?.identifier ?? ""
        }
    }

    private func speak() {
        guard let voice = AVSpeechSynthesisVoice(identifier: selectedIdentifier) else { return }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        synthesizer.speak(utterance)
    }

    // MARK: - Diagnostics capture (temporary; reads the extension's [diag] logs)

    /// Runs `log show` to pull the extension's diagnostic lines from the unified
    /// log (the extension runs in its own process, so we can't read them in-proc).
    /// Needs the app to be non-sandboxed — see project.yml (diagnostic build only).
    private func captureDiagnostics() {
        capturing = true
        diagNote = ""
        DispatchQueue.global(qos: .userInitiated).async {
            let lines = Self.runLogShow()
            DispatchQueue.main.async {
                if lines.isEmpty {
                    diagText = ""
                    diagNote = "No diagnostics found — press Speak first, then Capture."
                } else {
                    diagText = lines
                    diagNote = "Captured. Now click Copy or Save to Desktop."
                }
                capturing = false
            }
        }
    }

    private static func runLogShow() -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/log")
        process.arguments = [
            "show", "--last", "15m", "--info",
            "--predicate", "eventMessage CONTAINS \"KokoroVoice: [diag]\"",
            "--style", "compact",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let raw = String(data: data, encoding: .utf8) ?? ""
            // Keep only real diagnostic lines (the trailing space after "[diag] "
            // excludes the empty stream marker and `log`'s own echo of the
            // predicate), trimmed to the "KokoroVoice: …" payload.
            let marker = "KokoroVoice: [diag] "
            let payload = raw.split(separator: "\n").compactMap { line -> String? in
                guard let range = line.range(of: marker) else { return nil }
                return String(line[range.lowerBound...])
            }
            return payload.joined(separator: "\n")
        } catch {
            return "Could not read logs: \(error.localizedDescription)"
        }
    }

    private func copyDiagnostics() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(diagText, forType: .string)
        diagNote = "Copied — paste it into a message."
    }

    private func saveDiagnostics() {
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Desktop/kokoro_diag.txt")
        do {
            try diagText.write(to: url, atomically: true, encoding: .utf8)
            NSWorkspace.shared.activateFileViewerSelecting([url])
            diagNote = "Saved kokoro_diag.txt to the Desktop."
        } catch {
            diagNote = "Couldn't save to Desktop: \(error.localizedDescription)"
        }
    }
}

#Preview {
    ContentView()
}
