//
//  KokoroAudioUnit.swift
//  KokoroVoiceExtension
//
//  AVSpeechSynthesisProviderAudioUnit subclass that renders speech with the
//  Kokoro-82M neural TTS model (via KokoroSwift / MLX).
//
//  Flow (streaming): the system calls synthesizeSpeechRequest(_:) with SSML +
//  the chosen voice. We return immediately and synthesize on a background queue,
//  pushing PCM chunks onto a queue; the render block drains it incrementally so
//  live playback starts after the FIRST chunk instead of the whole selection.
//
//  Chunking: Kokoro is non-autoregressive with a hard 510-token limit, so input
//  MUST be chunked. We chunk at sentence boundaries (natural prosodic resets),
//  "hybrid"-style: the opening sentence ships on its own for the fastest start,
//  then later sentences are packed together (more cross-sentence context) up to
//  a conservative budget. Oversized chunks (run-ons, URLs, code) are caught at
//  synth time via `tooManyTokens` and re-split at the next natural boundary
//  (clause -> word -> hard cut), so no input is ever dropped.
//
//  Offline vs live: when the host renders offline (AVSpeechSynthesizer.write,
//  capture-to-file) it drains faster than we synthesize, so the render block
//  *waits* for the next chunk (safe — not the real-time thread), producing gap-
//  free audio. Live playback can't block the real-time thread, so instead it
//  *primes*: it buffers a couple of chunks before starting (and re-buffers after
//  any underrun), giving a slower Mac a head start so it plays gap-free rather
//  than stuttering. The cost is a slightly longer delay before the first word.
//

import AVFoundation
import KokoroSwift
import MLX
import MLXUtilsLibrary
import NaturalLanguage

public class KokoroAudioUnit: AVSpeechSynthesisProviderAudioUnit {

    // MARK: - Audio plumbing

    private var outputBus: AUAudioUnitBus
    private var _outputBusses: AUAudioUnitBusArray!
    private var format: AVAudioFormat

    // MARK: - Kokoro engine (lazy: loaded on first synthesis request)

    private var engine: KokoroTTS?
    private var voiceEmbeddings: [String: MLXArray] = [:]

    private static let sampleRate: Double = 24_000 // KokoroTTS.Constants.samplingRate

    /// Target chunk size in characters. Sized for *streaming smoothness*, not
    /// the 510-token model limit (we stay well under it): smaller chunks mean a
    /// chunk's synthesis finishes well within the previous chunk's playback, so a
    /// slower Mac stays ahead instead of underrunning mid-utterance. The trade-off
    /// is more sentence-boundary prosody resets. `tooManyTokens` is still the hard
    /// guard for pathological inputs.
    private static let maxCharBudget = 160

    // MARK: - Streaming state
    //
    // Producer: `synthQueue` (background). Consumer: the render block. All of it
    // is guarded by `cond`, an NSCondition: the producer signals it after each
    // chunk so an offline render waiting for audio can wake up.

    /// Live playback waits until this many chunks are buffered before it starts —
    /// and re-waits after an underrun — so a slower Mac builds a head start and
    /// plays gap-free instead of stuttering. Offline rendering ignores this (it
    /// can safely block per-chunk; it's not the real-time thread).
    private static let livePrimeChunks = 2

    private let cond = NSCondition()
    private var pendingChunks: [[Float]] = []   // synthesized chunks awaiting playback
    private var currentChunk: [Float] = []      // chunk currently being rendered
    private var chunkIndex: Int = 0             // read cursor within currentChunk
    private var synthesisFinished = false       // all input has been synthesized
    private var livePrimed = false              // live playback has buffered its head start
    private var generation: Int = 0             // bumped per request/cancel to drop stale work

    private let synthQueue = DispatchQueue(
        label: "com.vicnaum.kokorovoice.synth", qos: .userInitiated
    )

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

    // MARK: - Synthesis (streaming, chunked)

    public override func synthesizeSpeechRequest(_ speechRequest: AVSpeechSynthesisProviderRequest) {
        let text = SSML.plainText(from: speechRequest.ssmlRepresentation)
        let speed = SSML.speedMultiplier(from: speechRequest.ssmlRepresentation)
        let voiceIdentifier = speechRequest.voice.identifier

        // Reset streaming state and claim a new generation atomically.
        cond.lock()
        generation &+= 1
        let gen = generation
        pendingChunks.removeAll(keepingCapacity: true)
        currentChunk = []
        chunkIndex = 0
        synthesisFinished = false
        livePrimed = false
        cond.signal()
        cond.unlock()

        guard !text.isEmpty else { finishSynthesis(generation: gen); return }

        synthQueue.async { [weak self] in
            self?.runStreamingSynthesis(text: text, voiceIdentifier: voiceIdentifier,
                                        speed: speed, request: speechRequest, generation: gen)
        }
    }

    private func runStreamingSynthesis(text: String, voiceIdentifier: String, speed: Float,
                                       request: AVSpeechSynthesisProviderRequest, generation gen: Int) {
        guard ensureEngineLoaded(), let engine else { finishSynthesis(generation: gen); return }

        let definition = KokoroVoiceManifest.definition(forIdentifier: voiceIdentifier)
            ?? KokoroVoiceManifest.voices[0]
        guard let embedding = voiceEmbeddings[definition.npzKey] else {
            NSLog("KokoroVoice: voice \(definition.npzKey) not found in voices.npz")
            finishSynthesis(generation: gen)
            return
        }

        // Kokoro naming: 'a' prefix = US English, 'b' = British English.
        let language: KokoroSwift.Language = definition.npzKey.hasPrefix("b") ? .enGB : .enUS

        // Word-highlighting state (threaded locally, no shared races): a cursor
        // into the ORIGINAL SSML — tokens are matched back to it in order — and a
        // running frame offset into the audio we've produced so far.
        let ssml = request.ssmlRepresentation
        var ssmlCursor = ssml.startIndex
        var frameOffset = 0

        for chunk in Self.chunks(for: text) {
            guard isCurrent(gen) else { return } // superseded or cancelled
            frameOffset += synthesizeChunk(chunk, engine: engine, embedding: embedding,
                                           language: language, speed: speed, request: request, ssml: ssml,
                                           ssmlCursor: &ssmlCursor, startFrame: frameOffset,
                                           generation: gen, depth: 0)
        }
        finishSynthesis(generation: gen)
    }

    /// Synthesize one chunk; returns the number of audio frames produced so the
    /// caller can advance the marker frame offset. On `tooManyTokens` (the char
    /// budget under-counted phonemes) split at the next natural boundary and
    /// retry each piece, summing their frames. Depth-bounded so it always
    /// terminates (worst case: hard character cut).
    @discardableResult
    private func synthesizeChunk(_ text: String, engine: KokoroTTS, embedding: MLXArray,
                                 language: KokoroSwift.Language, speed: Float,
                                 request: AVSpeechSynthesisProviderRequest, ssml: String,
                                 ssmlCursor: inout String.Index, startFrame: Int,
                                 generation gen: Int, depth: Int) -> Int {
        guard isCurrent(gen) else { return 0 }
        do {
            let (audio, tokens) = try engine.generateAudio(voice: embedding, language: language,
                                                           text: text, speed: speed)
            enqueue(audio, generation: gen)
            if let tokens {
                emitWordMarkers(tokens, request: request, ssml: ssml,
                                ssmlCursor: &ssmlCursor, startFrame: startFrame, generation: gen)
            }
            return audio.count
        } catch KokoroTTS.KokoroTTSError.tooManyTokens where depth < 6 {
            var produced = 0
            for piece in Self.splitTooLong(text) {
                produced += synthesizeChunk(piece, engine: engine, embedding: embedding,
                                            language: language, speed: speed, request: request, ssml: ssml,
                                            ssmlCursor: &ssmlCursor, startFrame: startFrame + produced,
                                            generation: gen, depth: depth + 1)
            }
            return produced
        } catch {
            NSLog("KokoroVoice: synthesis failed for \(text.count)-char chunk: \(error)")
            return 0
        }
    }

    // MARK: - Word highlighting

    /// Build `.word` markers from Kokoro's per-token timestamps and hand them to
    /// the system's metadata block. Each token is matched back to the original
    /// SSML (in order) for the highlight range; tokens rewritten by normalization
    /// (e.g. "~" -> "approximately") simply aren't found and are skipped.
    /// `byteSampleOffset` is the byte position in our 24 kHz float32 audio.
    private func emitWordMarkers(_ tokens: [MToken], request: AVSpeechSynthesisProviderRequest,
                                 ssml: String, ssmlCursor: inout String.Index,
                                 startFrame: Int, generation gen: Int) {
        guard let block = speechSynthesisOutputMetadataBlock, isCurrent(gen) else { return }
        var markers: [AVSpeechSynthesisMarker] = []
        for token in tokens {
            guard let startTs = token.start_ts,
                  let range = Self.advanceSSMLRange(for: token.text, in: ssml, cursor: &ssmlCursor)
            else { continue }
            let frame = startFrame + Int(startTs * Self.sampleRate)
            let byteOffset = frame * MemoryLayout<Float>.size   // float32: 4 bytes/frame
            markers.append(AVSpeechSynthesisMarker(markerType: .word,
                                                   forTextRange: range,
                                                   atByteSampleOffset: byteOffset))
        }
        if !markers.isEmpty { block(markers, request) }
    }

    /// Find `tokenText` in `ssml` from `cursor` (case-insensitive), advancing the
    /// cursor past the match. Returns the NSRange, or nil if not present verbatim.
    private static func advanceSSMLRange(for tokenText: String, in ssml: String,
                                         cursor: inout String.Index) -> NSRange? {
        let needle = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty, cursor < ssml.endIndex,
              let found = ssml.range(of: needle, options: [.caseInsensitive],
                                     range: cursor..<ssml.endIndex)
        else { return nil }
        cursor = found.upperBound
        return NSRange(found, in: ssml)
    }

    public override func cancelSpeechRequest() {
        cond.lock()
        generation &+= 1            // invalidate any in-flight background synthesis
        pendingChunks.removeAll(keepingCapacity: true)
        currentChunk = []
        chunkIndex = 0
        livePrimed = false
        synthesisFinished = true    // let the render block complete (and wake if waiting)
        cond.signal()
        cond.unlock()
    }

    // MARK: - Streaming queue helpers (producer side)

    private func isCurrent(_ gen: Int) -> Bool {
        cond.lock(); defer { cond.unlock() }
        return gen == generation
    }

    private func enqueue(_ samples: [Float], generation gen: Int) {
        guard !samples.isEmpty else { return }
        cond.lock()
        if gen == generation { pendingChunks.append(samples) }
        cond.signal()
        cond.unlock()
    }

    private func finishSynthesis(generation gen: Int) {
        cond.lock()
        if gen == generation { synthesisFinished = true }
        cond.signal()
        cond.unlock()
    }

    // MARK: - Text chunking

    /// Hybrid streaming chunks: the opening sentence ships alone (fastest start),
    /// then subsequent sentences are packed together (max cross-sentence context)
    /// up to `maxCharBudget`. Approximate by design — `tooManyTokens` is the
    /// real guard in `synthesizeChunk`.
    private static func chunks(for text: String) -> [String] {
        let sentences = self.sentences(in: text)
        guard !sentences.isEmpty else { return [text] }

        var chunks: [String] = []
        var buffer = ""
        func flush() { if !buffer.isEmpty { chunks.append(buffer); buffer = "" } }

        for (index, sentence) in sentences.enumerated() {
            if index == 0 {
                chunks.append(sentence)                                  // hybrid fast start
            } else if buffer.isEmpty {
                buffer = sentence
            } else if buffer.count + 1 + sentence.count <= maxCharBudget {
                buffer += " " + sentence                                 // pack for context
            } else {
                flush(); buffer = sentence
            }
        }
        flush()
        return chunks
    }

    /// Sentence segmentation. NLTokenizer handles abbreviations far better than
    /// splitting on `.?!`.
    private static func sentences(in text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let sentence = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !sentence.isEmpty { result.append(sentence) }
            return true
        }
        return result.isEmpty ? [text] : result
    }

    /// Break an over-long chunk into two pieces at the most natural boundary
    /// nearest the middle: clause punctuation first, then any whitespace, and as
    /// a last resort (one enormous token) a hard character cut.
    private static func splitTooLong(_ text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let chars = Array(trimmed)
        guard chars.count > 1 else { return [trimmed] }
        let middle = chars.count / 2

        let clausePunctuation: Set<Character> = [";", ":", ",", "—", "–", ")", "("]
        func nearestBreak(where matches: (Character) -> Bool) -> Int? {
            for radius in 0..<chars.count {
                for candidate in [middle - radius, middle + radius]
                where candidate > 0 && candidate < chars.count - 1 {
                    if matches(chars[candidate]) { return candidate + 1 }
                }
            }
            return nil
        }

        let cut = nearestBreak { clausePunctuation.contains($0) }
            ?? nearestBreak { $0 == " " }
            ?? middle   // no boundary at all: hard cut

        let first = String(chars[0..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
        let second = String(chars[cut...]).trimmingCharacters(in: .whitespacesAndNewlines)
        return [first, second].filter { !$0.isEmpty }
    }

    // MARK: - Rendering (consumer side)

    public override var internalRenderBlock: AUInternalRenderBlock {
        return { [weak self] actionFlags, _, frameCount, _, outputAudioBufferList, _, _ in
            let output = UnsafeMutableAudioBufferListPointer(outputAudioBufferList)[0]
            let frames = output.mData!.assumingMemoryBound(to: Float32.self)
            let frameCountInt = Int(frameCount)

            guard let self else {
                for i in 0..<frameCountInt { frames[i] = 0 }
                actionFlags.pointee = .offlineUnitRenderAction_Complete
                return noErr
            }

            self.cond.lock()

            // Live priming: don't begin (or resume after an underrun) until a few
            // chunks are buffered, so a slower Mac plays gap-free instead of
            // stuttering. Offline rendering skips this and waits per-chunk below.
            if !self.isRenderingOffline, !self.livePrimed {
                if !self.synthesisFinished, self.pendingChunks.count < Self.livePrimeChunks {
                    self.cond.unlock()
                    for i in 0..<frameCountInt { frames[i] = 0 }
                    return noErr
                }
                self.livePrimed = true
            }

            var filled = 0
            while filled < frameCountInt {
                // Need a fresh chunk?
                if self.chunkIndex >= self.currentChunk.count {
                    if self.pendingChunks.isEmpty {
                        if self.synthesisFinished {
                            self.cond.unlock()
                            for i in filled..<frameCountInt { frames[i] = 0 }
                            actionFlags.pointee = .offlineUnitRenderAction_Complete
                            return noErr
                        }
                        // Underrun, more is coming. Offline: wait for it (the
                        // render thread isn't real-time). Live: emit silence and
                        // pick up the audio on the next render callback.
                        if self.isRenderingOffline {
                            self.cond.wait()   // releases the lock until signaled
                            continue
                        }
                        // Live underrun: re-prime so we resume only once enough is
                        // buffered again — one clean pause instead of a stutter.
                        self.livePrimed = false
                        self.cond.unlock()
                        for i in filled..<frameCountInt { frames[i] = 0 }
                        return noErr
                    }
                    self.currentChunk = self.pendingChunks.removeFirst()
                    self.chunkIndex = 0
                }

                let available = self.currentChunk.count - self.chunkIndex
                let take = min(available, frameCountInt - filled)
                self.currentChunk.withUnsafeBufferPointer { src in
                    (frames + filled).update(from: src.baseAddress! + self.chunkIndex, count: take)
                }
                filled += take
                self.chunkIndex += take
            }
            self.cond.unlock()
            return noErr
        }
    }
}
