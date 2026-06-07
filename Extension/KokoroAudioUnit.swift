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
//  Real-time-safe playback: the producer writes synthesized samples into a fixed,
//  pre-allocated buffer and publishes the valid sample count via an atomic. The
//  render block (the audio thread) does nothing but a lock-free atomic read and a
//  memcpy from the already-written region — no mutex, no waiting, no allocation,
//  no ARC. That matters because the host renders us "offline" but pulls paced ~
//  real-time, so a slow/contended render callback (locks, ARC, logging) makes the
//  audio output skip mid-word under CPU load. A NSCondition is used ONLY on the
//  cold path (warm-up before any audio exists, or rare starvation) — never on the
//  steady-state hot path.
//

import AVFoundation
import KokoroSwift
import MLX
import MLXUtilsLibrary
import NaturalLanguage
import Synchronization

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

    /// Kokoro pads every synthesized chunk with silence (~0.3s leading, ~0.75s
    /// trailing). Fine for a whole utterance, but when we chunk, each seam becomes
    /// a ~1s dead-air gap that sounds like a stutter/dropout. We cap each chunk's
    /// edge silence to these amounts so seams become a normal sentence pause.
    private static let leadingSilenceCap = Int(0.06 * sampleRate)   // 60 ms
    private static let trailingSilenceCap = Int(0.34 * sampleRate)  // 340 ms

    // MARK: - Playback buffer (real-time-safe producer→consumer hand-off)

    /// Max audio per utterance, in samples (~6 min at 24 kHz ≈ 35 MB). Longer
    /// utterances are truncated (logged) — far beyond any normal request.
    private static let audioCapacity = Int(sampleRate) * 360

    /// Stable, pre-allocated sample buffer. The producer appends; the render block
    /// reads the [0, availableCount) region, which the producer never rewrites.
    private let audioBuffer = UnsafeMutablePointer<Float>.allocate(capacity: KokoroAudioUnit.audioCapacity)

    private let availableCount = Atomic<Int>(0)   // samples ready to play (producer→render)
    private let producerDone = Atomic<Bool>(false)
    private let generation = Atomic<Int>(0)       // bumped per request/cancel to drop stale work
    private let cond = NSCondition()              // cold-path wait/signal ONLY (never the hot path)

    private var producerWritten = 0   // producer/synthQueue-thread only
    private var renderCursor = 0      // render-thread only
    private var renderGen = -1        // render-thread only
    private var renderColdEntries = 0 // render-thread only (diagnostic)

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

    deinit { audioBuffer.deallocate() }

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

        // Claim a new generation and reset the published buffer state. (producerWritten
        // is reset on synthQueue, which is serial, so it can't race a prior synth.)
        let gen = generation.wrappingAdd(1, ordering: .relaxed).newValue
        availableCount.store(0, ordering: .releasing)
        producerDone.store(false, ordering: .releasing)
        cond.lock(); cond.broadcast(); cond.unlock()   // wake any render waiting on the old generation

        guard !text.isEmpty else { finishSynthesis(generation: gen); return }

        synthQueue.async { [weak self] in
            self?.runStreamingSynthesis(text: text, voiceIdentifier: voiceIdentifier,
                                        speed: speed, request: speechRequest, generation: gen)
        }
    }

    private func runStreamingSynthesis(text: String, voiceIdentifier: String, speed: Float,
                                       request: AVSpeechSynthesisProviderRequest, generation gen: Int) {
        producerWritten = 0   // serial synthQueue: safe to reset here
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

        // Word-highlighting state: a cursor into the ORIGINAL SSML (tokens are
        // matched back to it in order). Frame offsets come from `producerWritten`.
        let ssml = request.ssmlRepresentation
        var ssmlCursor = ssml.startIndex

        let chunkList = Self.chunks(for: text)
        NSLog("KokoroVoice: [diag] stream start gen=\(gen) chars=\(text.count) chunks=\(chunkList.count) voice=\(definition.npzKey) speed=\(String(format: "%.2f", speed)) budget=\(Self.maxCharBudget)")
        let streamT0 = CFAbsoluteTimeGetCurrent()

        for chunk in chunkList {
            guard isCurrent(gen) else { return } // superseded or cancelled
            synthesizeChunk(chunk, engine: engine, embedding: embedding,
                            language: language, speed: speed, request: request, ssml: ssml,
                            ssmlCursor: &ssmlCursor, generation: gen, depth: 0)
        }
        let streamWall = CFAbsoluteTimeGetCurrent() - streamT0
        let audioSec = Double(producerWritten) / Self.sampleRate
        NSLog("KokoroVoice: [diag] stream done gen=\(gen) audio=\(String(format: "%.2f", audioSec))s synthWall=\(String(format: "%.2f", streamWall))s overallRTF=\(String(format: "%.2f", streamWall / max(audioSec, 0.001)))")
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
                                 ssmlCursor: inout String.Index,
                                 generation gen: Int, depth: Int) -> Int {
        guard isCurrent(gen) else { return 0 }
        do {
            let synthT0 = CFAbsoluteTimeGetCurrent()
            let (rawAudio, tokens) = try engine.generateAudio(voice: embedding, language: language,
                                                              text: text, speed: speed)
            let synthSec = CFAbsoluteTimeGetCurrent() - synthT0
            // Trim Kokoro's per-chunk silence padding so concatenated chunks don't
            // get ~1s gaps at the seams.
            let (audio, leadingRemoved) = Self.capEdgeSilence(rawAudio,
                leadingCap: Self.leadingSilenceCap, trailingCap: Self.trailingSilenceCap)
            let rawSec = Double(rawAudio.count) / Self.sampleRate
            let chunkAudioSec = Double(audio.count) / Self.sampleRate
            NSLog("KokoroVoice: [diag] synth chunk chars=\(text.count) depth=\(depth) audio=\(String(format: "%.2f", chunkAudioSec))s (raw \(String(format: "%.2f", rawSec))s) synth=\(String(format: "%.2f", synthSec))s RTF=\(String(format: "%.2f", synthSec / max(rawSec, 0.001))) (RTF<1.0 = faster than real-time)")
            let baseFrame = producerWritten
            let written = appendSamples(audio, generation: gen)
            if let tokens {
                emitWordMarkers(tokens, request: request, ssml: ssml,
                                ssmlCursor: &ssmlCursor, startFrame: baseFrame - leadingRemoved, generation: gen)
            }
            return written
        } catch KokoroTTS.KokoroTTSError.tooManyTokens where depth < 6 {
            var produced = 0
            for piece in Self.splitTooLong(text) {
                produced += synthesizeChunk(piece, engine: engine, embedding: embedding,
                                            language: language, speed: speed, request: request, ssml: ssml,
                                            ssmlCursor: &ssmlCursor, generation: gen, depth: depth + 1)
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
            let frame = max(0, startFrame + Int(startTs * Self.sampleRate))
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
        generation.wrappingAdd(1, ordering: .relaxed)   // invalidate any in-flight synthesis
        availableCount.store(0, ordering: .releasing)
        producerDone.store(true, ordering: .releasing)  // let a waiting render complete
        cond.lock(); cond.broadcast(); cond.unlock()
    }

    // MARK: - Producer side (synthQueue)

    private func isCurrent(_ gen: Int) -> Bool {
        gen == generation.load(ordering: .acquiring)
    }

    /// Append synthesized samples to the playback buffer and publish the new count.
    /// Producer/synthQueue thread only. Returns frames actually written (may be less
    /// than `samples.count` if the buffer is full). Wakes a cold-path render.
    @discardableResult
    private func appendSamples(_ samples: [Float], generation gen: Int) -> Int {
        guard !samples.isEmpty, gen == generation.load(ordering: .acquiring) else { return 0 }
        let n = min(samples.count, Self.audioCapacity - producerWritten)
        if n > 0 {
            samples.withUnsafeBufferPointer { src in
                (audioBuffer + producerWritten).update(from: src.baseAddress!, count: n)
            }
            producerWritten += n
            availableCount.store(producerWritten, ordering: .releasing)  // publish (release): samples are written before this
            cond.lock(); cond.signal(); cond.unlock()
        }
        if n < samples.count {
            NSLog("KokoroVoice: [diag] playback buffer full at \(Self.audioCapacity / Int(Self.sampleRate))s — truncating utterance")
        }
        return n
    }

    private func finishSynthesis(generation gen: Int) {
        guard gen == generation.load(ordering: .acquiring) else { return }
        producerDone.store(true, ordering: .releasing)
        cond.lock(); cond.broadcast(); cond.unlock()
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

    /// Trim Kokoro's silence padding from a chunk's edges, keeping at most
    /// `leadingCap`/`trailingCap` samples. Returns the trimmed audio and how many
    /// leading samples were removed (so word-marker offsets stay aligned). Only the
    /// contiguous run from each edge is touched — internal pauses are preserved.
    private static func capEdgeSilence(_ audio: [Float], leadingCap: Int, trailingCap: Int)
        -> (audio: [Float], leadingRemoved: Int) {
        let threshold: Float = 0.01
        let n = audio.count
        guard n > 0 else { return (audio, 0) }
        var firstVoice = 0
        while firstVoice < n, abs(audio[firstVoice]) < threshold { firstVoice += 1 }
        guard firstVoice < n else { return (audio, 0) }   // all silence: leave untouched
        var lastVoice = n - 1
        while lastVoice > firstVoice, abs(audio[lastVoice]) < threshold { lastVoice -= 1 }
        let start = max(0, firstVoice - leadingCap)
        let end = min(n - 1, lastVoice + trailingCap)
        if start == 0, end == n - 1 { return (audio, 0) }
        return (Array(audio[start...end]), start)
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

            // New utterance? reset our render-thread-only read cursor.
            let gen = self.generation.load(ordering: .acquiring)
            if gen != self.renderGen {
                self.renderGen = gen
                self.renderCursor = 0
                self.renderColdEntries = 0
            }

            var filled = 0
            while filled < frameCountInt {
                let avail = self.availableCount.load(ordering: .acquiring)
                let ready = avail - self.renderCursor
                if ready > 0 {
                    // HOT PATH — lock-free, allocation-free, ARC-free: copy from the
                    // stable already-written region. This is ALL the audio thread does
                    // in steady state, so a CPU spike can't make it miss its deadline.
                    let take = min(ready, frameCountInt - filled)
                    (frames + filled).update(from: self.audioBuffer + self.renderCursor, count: take)
                    filled += take
                    self.renderCursor += take
                    continue
                }
                if self.producerDone.load(ordering: .acquiring) {
                    for i in filled..<frameCountInt { frames[i] = 0 }
                    actionFlags.pointee = .offlineUnitRenderAction_Complete
                    NSLog("KokoroVoice: [diag] render complete offline=\(self.isRenderingOffline) audio=\(String(format: "%.2f", Double(self.renderCursor) / Self.sampleRate))s coldWaits=\(self.renderColdEntries)")
                    return noErr
                }
                // COLD PATH — no audio yet (warm-up, or rare starvation). Block until
                // there's more. This never runs on the steady-state hot path, so it
                // can't cause the load-induced mid-word stutter; it only covers the
                // gap before the first samples exist (and keeps offline capture exact).
                self.renderColdEntries += 1
                self.cond.lock()
                while self.generation.load(ordering: .acquiring) == gen,
                      self.availableCount.load(ordering: .acquiring) - self.renderCursor <= 0,
                      !self.producerDone.load(ordering: .acquiring) {
                    self.cond.wait()
                }
                self.cond.unlock()
                if self.generation.load(ordering: .acquiring) != gen {
                    // a new utterance arrived while we waited; emit silence — the next
                    // callback picks it up via the cursor reset at the top.
                    for i in filled..<frameCountInt { frames[i] = 0 }
                    return noErr
                }
            }
            return noErr
        }
    }
}
