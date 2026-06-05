//
//  AudioUnitFactory.swift
//  KokoroVoiceExtension
//
//  Principal class of the extension: hands the system an instance of the
//  Kokoro speech synthesis audio unit.
//

import CoreAudioKit

public class AudioUnitFactory: NSObject, AUAudioUnitFactory {

    var audioUnit: AUAudioUnit?

    public func beginRequest(with context: NSExtensionContext) {}

    @objc
    public func createAudioUnit(with componentDescription: AudioComponentDescription) throws -> AUAudioUnit {
        let unit = try KokoroAudioUnit(componentDescription: componentDescription, options: [])
        audioUnit = unit
        return unit
    }
}
