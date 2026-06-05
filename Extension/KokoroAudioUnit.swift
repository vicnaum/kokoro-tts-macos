//
//  KokoroAudioUnit.swift
//  KokoroVoiceExtension
//
//  AVSpeechSynthesisProviderAudioUnit subclass that renders speech with the
//  Kokoro-82M neural TTS model (via KokoroSwift / MLX).
//
//  Flow: the system calls synthesizeSpeechRequest(_:) with SSML + the chosen
//  voice. We synthesize the full utterance into a PCM buffer, then the render
//  block streams it out frame by frame and signals completion.
//

import AVFoundation
import KokoroSwift
import MLX
import MLXUtilsLibrary

public class KokoroAudioUnit: AVSpeechSynthesisProviderAudioUnit {

    // MARK: - Audio plumbing

    private var outputBus: AUAudioUnitBus
    private var _outputBusses: AUAudioUnitBusArray!
    private var format: AVAudioFormat

    private var request: AVSpeechSynthesisProviderRequest?
    private var currentBuffer: AVAudioPCMBuffer?
    private var framePosition: AVAudioFramePosition = 0

    // MARK: - Kokoro engine (lazy: loaded on first synthesis request)

    private var engine: KokoroTTS?
    private var voiceEmbeddings: [String: MLXArray] = [:]

    private static let sampleRate: Double = 24_000 // KokoroTTS.Constants.samplingRate

    // MARK: - Lifecycle

    @objc
    override init(componentDescription: AudioComponentDescription,
                  options: AudioComponentInstantiationOptions) throws {

        let basicDescription = AudioStreamBasicDescription(
            mSampleRate: Self.sampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagsNativeFloatPacked | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        format = AVAudioFormat(
            cmAudioFormatDescription: try! CMAudioFormatDescription(
                audioStreamBasicDescription: basicDescription
            )
        )
        outputBus = try AUAudioUnitBus(format: format)

        try super.init(componentDescription: componentDescription, options: options)

        _outputBusses = AUAudioUnitBusArray(
            audioUnit: self,
            busType: .output,
            busses: [outputBus]
        )
    }

    // MARK: - Voice catalog

    public override var speechVoices: [AVSpeechSynthesisProviderVoice] {
        get { KokoroVoiceManifest.providerVoices() }
        set {}
    }

    public override var outputBusses: AUAudioUnitBusArray {
        _outputBusses
    }

    // MARK: - Engine loading

    private func resourceBundle() -> Bundle {
        // In an app extension Bundle.main is the .appex itself; the class
        // bundle is a fallback in case resources end up framework-side.
        if Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors") != nil {
            return Bundle.main
        }
        return Bundle(for: KokoroAudioUnit.self)
    }

    private func ensureEngineLoaded() -> Bool {
        if engine != nil, !voiceEmbeddings.isEmpty { return true }

        let bundle = resourceBundle()
        guard
            let modelURL = bundle.url(forResource: "kokoro-v1_0", withExtension: "safetensors"),
            let voicesURL = bundle.url(forResource: "voices", withExtension: "npz")
        else {
            NSLog("KokoroVoice: model resources missing from extension bundle")
            return false
        }

        if engine == nil {
            engine = KokoroTTS(modelPath: modelURL)
        }
        if voiceEmbeddings.isEmpty {
            voiceEmbeddings = NpyzReader.read(fileFromPath: voicesURL) ?? [:]
        }
        return engine != nil && !voiceEmbeddings.isEmpty
    }

    // MARK: - Synthesis

    public override func synthesizeSpeechRequest(_ speechRequest: AVSpeechSynthesisProviderRequest) {
        request = speechRequest
        framePosition = 0
        currentBuffer = nil

        let text = SSML.plainText(from: speechRequest.ssmlRepresentation)
        guard !text.isEmpty, ensureEngineLoaded(), let engine else { return }

        let definition = KokoroVoiceManifest.definition(
            forIdentifier: speechRequest.voice.identifier
        ) ?? KokoroVoiceManifest.voices[0]

        guard let embedding = voiceEmbeddings[definition.npzKey] else {
            NSLog("KokoroVoice: voice \(definition.npzKey) not found in voices.npz")
            return
        }

        // Kokoro naming: 'a' prefix = US English, 'b' = British English.
        let language: KokoroTTS.Language = definition.npzKey.hasPrefix("b") ? .enGB : .enUS

        do {
            let (audio, _) = try engine.generateAudio(
                voice: embedding,
                language: language,
                text: text
            )
            currentBuffer = makeBuffer(from: audio)
        } catch {
            NSLog("KokoroVoice: synthesis failed: \(error)")
        }
    }

    public override func cancelSpeechRequest() {
        request = nil
        currentBuffer = nil
        framePosition = 0
    }

    private func makeBuffer(from samples: [Float]) -> AVAudioPCMBuffer? {
        guard !samples.isEmpty,
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: format,
                  frameCapacity: AVAudioFrameCount(samples.count)
              )
        else { return nil }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        let destination = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { source in
            destination.update(from: source.baseAddress!, count: source.count)
        }
        return buffer
    }

    // MARK: - Rendering

    public override var internalRenderBlock: AUInternalRenderBlock {
        return { [weak self] actionFlags, _, frameCount, _, outputAudioBufferList, _, _ in
            let output = UnsafeMutableAudioBufferListPointer(outputAudioBufferList)[0]
            let frames = output.mData!.assumingMemoryBound(to: Float32.self)

            // Silence by default.
            for frame in 0..<Int(frameCount) {
                frames[frame] = 0.0
            }

            guard let self, let buffer = self.currentBuffer else {
                // Nothing to play (failed or cancelled synthesis): tell the
                // system we are done instead of stalling.
                actionFlags.pointee = .offlineUnitRenderAction_Complete
                return noErr
            }

            let source = UnsafeMutableAudioBufferListPointer(buffer.mutableAudioBufferList)[0]
            let sourceFrames = source.mData!.assumingMemoryBound(to: Float32.self)

            for frame in 0..<Int(frameCount) {
                guard self.framePosition < AVAudioFramePosition(buffer.frameLength) else {
                    actionFlags.pointee = .offlineUnitRenderAction_Complete
                    break
                }
                frames[frame] = sourceFrames[Int(self.framePosition)]
                self.framePosition += 1
            }

            if self.framePosition >= AVAudioFramePosition(buffer.frameLength) {
                actionFlags.pointee = .offlineUnitRenderAction_Complete
            }
            return noErr
        }
    }
}
