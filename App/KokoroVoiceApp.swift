//
//  KokoroVoiceApp.swift
//  KokoroVoice
//
//  Host app for the Kokoro speech synthesis extension. Launching this app
//  registers the embedded extension with the system; the UI lets you verify
//  the voices work before enabling them in Spoken Content.
//

import SwiftUI

@main
struct KokoroVoiceApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 520, minHeight: 480)
        }
    }
}
