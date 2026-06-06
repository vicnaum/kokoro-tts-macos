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
    @State private var text = "Hello! I am Kokoro, a neural voice running entirely on this Mac."

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
}

#Preview {
    ContentView()
}
