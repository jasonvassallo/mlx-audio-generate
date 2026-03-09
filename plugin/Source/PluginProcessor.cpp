#include "PluginProcessor.h"
#include "PluginEditor.h"

// ---------------------------------------------------------------------------
// APVTS Parameter Layout — exposed to Push 2, automation, MIDI mapping
// ---------------------------------------------------------------------------

juce::AudioProcessorValueTreeState::ParameterLayout
MLXAudioGenProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Model: 0 = musicgen, 1 = stable_audio
    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        juce::ParameterID ("model", 1), "Model",
        juce::StringArray { "MusicGen", "Stable Audio" }, 0));

    // Duration
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("seconds", 1), "Duration",
        juce::NormalisableRange<float> (0.5f, 60.0f, 0.5f), 5.0f));

    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("bars", 1), "Bars", 1, 32, 4));

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("barsMode", 1), "Bars Mode", false));

    // BPM
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("manualBpm", 1), "Manual BPM",
        juce::NormalisableRange<float> (40.0f, 240.0f, 1.0f), 120.0f));

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("dawBpm", 1), "DAW BPM Sync", true));

    // MusicGen params
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("temperature", 1), "Temperature",
        juce::NormalisableRange<float> (0.1f, 2.0f, 0.05f), 1.0f));

    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("topK", 1), "Top K", 1, 500, 250));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("guidance", 1), "Guidance",
        juce::NormalisableRange<float> (0.0f, 10.0f, 0.1f), 3.0f));

    // Stable Audio params
    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("steps", 1), "Steps", 1, 100, 8));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("cfgScale", 1), "CFG Scale",
        juce::NormalisableRange<float> (0.0f, 15.0f, 0.1f), 6.0f));

    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        juce::ParameterID ("sampler", 1), "Sampler",
        juce::StringArray { "Euler", "RK4" }, 0));

    // Seed
    params.push_back (std::make_unique<juce::AudioParameterInt> (
        juce::ParameterID ("seed", 1), "Seed", -1, 99999, -1));

    // Playback
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("loop", 1), "Loop", true));

    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("midiTrigger", 1), "MIDI Trigger", false));

    // Effects
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID ("fxEnabled", 1), "FX Enabled", false));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("compThreshold", 1), "Comp Threshold",
        juce::NormalisableRange<float> (-60.0f, 0.0f, 1.0f), 0.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("compRatio", 1), "Comp Ratio",
        juce::NormalisableRange<float> (1.0f, 20.0f, 0.1f), 1.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("delayTime", 1), "Delay Time",
        juce::NormalisableRange<float> (0.0f, 1000.0f, 1.0f), 0.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("delayMix", 1), "Delay Mix",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.0f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("delayFeedback", 1), "Delay Feedback",
        juce::NormalisableRange<float> (0.0f, 0.95f, 0.01f), 0.3f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("reverbSize", 1), "Reverb Size",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.5f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("reverbDamping", 1), "Reverb Damping",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.5f));

    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("reverbMix", 1), "Reverb Mix",
        juce::NormalisableRange<float> (0.0f, 1.0f, 0.01f), 0.0f));

    // Output gain (dB)
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("outputGain", 1), "Output Gain",
        juce::NormalisableRange<float> (-24.0f, 12.0f, 0.1f), 0.0f));

    // Crossfade loop (ms)
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID ("loopFade", 1), "Loop Crossfade",
        juce::NormalisableRange<float> (0.0f, 50.0f, 1.0f), 10.0f));

    return { params.begin(), params.end() };
}

// Helper to read APVTS params
#define PARAM_FLOAT(id)  apvts.getRawParameterValue(id)->load()
#define PARAM_INT(id)    (int) apvts.getRawParameterValue(id)->load()
#define PARAM_BOOL(id)   (apvts.getRawParameterValue(id)->load() >= 0.5f)
#define PARAM_CHOICE(id) (int) apvts.getRawParameterValue(id)->load()

// ---------------------------------------------------------------------------
// Background generation thread
// ---------------------------------------------------------------------------

class GenerationThread : public juce::Thread
{
public:
    GenerationThread (MLXAudioGenProcessor& p)
        : juce::Thread ("MLX Generation"), processor (p) {}
    void run() override { processor.runGeneration(); }
private:
    MLXAudioGenProcessor& processor;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

MLXAudioGenProcessor::MLXAudioGenProcessor()
    : AudioProcessor (BusesProperties()
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "MLXAudioGen", createParameterLayout())
{
}

MLXAudioGenProcessor::~MLXAudioGenProcessor()
{
    stopTimer();
    if (generationThread && generationThread->isThreadRunning())
        generationThread->stopThread (5000);
}

void MLXAudioGenProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = (juce::uint32) samplesPerBlock;
    spec.numChannels = 2;
    delayLine.prepare (spec);
    delayLine.setMaximumDelayInSamples ((int) sampleRate);
    reverb.prepare (spec);

    if (! serverLauncher.isServerAlive())
    {
        auto* launcher = &serverLauncher;
        auto* self = this;
        juce::Thread::launch ([launcher, self] {
            launcher->ensureServerRunning();
            juce::ScopedLock lock (self->stateLock);
            self->statusMessage = launcher->getStatus();
        });
    }
}

void MLXAudioGenProcessor::releaseResources() {}

// ---------------------------------------------------------------------------
// Audio processing
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& midi)
{
    buffer.clear();

    if (auto* playHead = getPlayHead())
    {
        if (auto pos = playHead->getPosition())
            if (auto bpm = pos->getBpm())
                dawBpm = (float) *bpm;
    }

    // MIDI trigger
    if (PARAM_BOOL ("midiTrigger"))
    {
        for (const auto metadata : midi)
        {
            if (metadata.getMessage().isNoteOn() && ! generating.load())
            {
                juce::MessageManager::callAsync ([this] { triggerGeneration(); });
                break;
            }
        }
    }

    if (! playing.load() || ! hasAudio.load() || generatedAudio.getNumSamples() == 0)
        return;

    bool looping = PARAM_BOOL ("loop");
    const int numChannels = juce::jmin (buffer.getNumChannels(), generatedAudio.getNumChannels());
    const int numSamples = buffer.getNumSamples();
    const int totalSamples = generatedAudio.getNumSamples();
    int pos = playbackPosition.load();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        int writePos = 0, readPos = pos;
        while (writePos < numSamples)
        {
            int toCopy = juce::jmin (numSamples - writePos, totalSamples - readPos);
            if (toCopy > 0)
                buffer.copyFrom (ch, writePos, generatedAudio, ch, readPos, toCopy);
            writePos += toCopy;
            readPos += toCopy;
            if (readPos >= totalSamples)
            {
                if (looping) readPos = 0;
                else { playing.store (false); break; }
            }
        }
    }

    pos += numSamples;
    if (pos >= totalSamples)
        pos = looping ? pos % totalSamples : totalSamples;
    playbackPosition.store (pos);

    // Crossfade at loop point to eliminate clicks
    float fadeMs = PARAM_FLOAT ("loopFade");
    if (looping && fadeMs > 0.0f && totalSamples > 0)
    {
        int fadeSamples = (int) (fadeMs * 0.001f * (float) currentSampleRate);
        fadeSamples = juce::jmin (fadeSamples, totalSamples / 4);
        int curPos = playbackPosition.load();

        // Fade out near end of buffer
        if (curPos >= totalSamples - fadeSamples && curPos < totalSamples)
        {
            int distFromEnd = totalSamples - (curPos - numSamples);
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            {
                auto* d = buffer.getWritePointer (ch);
                for (int i = 0; i < numSamples; ++i)
                {
                    int sampleDist = distFromEnd - i;
                    if (sampleDist >= 0 && sampleDist < fadeSamples)
                    {
                        float fade = (float) sampleDist / (float) fadeSamples;
                        d[i] *= fade;
                    }
                }
            }
        }
    }

    // Apply effects chain
    if (PARAM_BOOL ("fxEnabled"))
        applyEffects (buffer);

    // Output gain
    float gainDb = PARAM_FLOAT ("outputGain");
    if (std::abs (gainDb) > 0.05f)
    {
        float gain = std::pow (10.0f, gainDb / 20.0f);
        buffer.applyGain (gain);
    }
}

// ---------------------------------------------------------------------------
// Effects
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::updateEffectsParameters()
{
    juce::dsp::Reverb::Parameters rp;
    rp.roomSize = PARAM_FLOAT ("reverbSize");
    rp.damping = PARAM_FLOAT ("reverbDamping");
    rp.wetLevel = PARAM_FLOAT ("reverbMix");
    rp.dryLevel = 1.0f - rp.wetLevel * 0.5f;
    reverb.setParameters (rp);
    delayLine.setDelay (PARAM_FLOAT ("delayTime") * 0.001f * (float) currentSampleRate);
}

void MLXAudioGenProcessor::applyEffects (juce::AudioBuffer<float>& buffer)
{
    updateEffectsParameters();
    const int nc = buffer.getNumChannels();
    const int ns = buffer.getNumSamples();

    float compT = PARAM_FLOAT ("compThreshold");
    float compR = PARAM_FLOAT ("compRatio");
    float delayMx = PARAM_FLOAT ("delayMix");
    float delayFb = PARAM_FLOAT ("delayFeedback");
    float delayT = PARAM_FLOAT ("delayTime");

    if (compR > 1.0f)
    {
        for (int ch = 0; ch < nc; ++ch)
        {
            auto* d = buffer.getWritePointer (ch);
            for (int i = 0; i < ns; ++i)
            {
                float a = std::abs (d[i]);
                float db = a > 0.0001f ? 20.0f * std::log10 (a) : -80.0f;
                if (db > compT)
                {
                    float g = std::pow (10.0f, ((compT + (db - compT) / compR) - db) / 20.0f);
                    d[i] *= g;
                }
            }
        }
    }

    if (delayT > 0.0f && delayMx > 0.0f)
    {
        for (int ch = 0; ch < nc; ++ch)
        {
            auto* d = buffer.getWritePointer (ch);
            for (int i = 0; i < ns; ++i)
            {
                float delayed = delayLine.popSample (ch);
                delayLine.pushSample (ch, d[i] + delayed * delayFb);
                d[i] = d[i] * (1.0f - delayMx) + delayed * delayMx;
            }
        }
    }

    if (PARAM_FLOAT ("reverbMix") > 0.0f)
    {
        juce::dsp::AudioBlock<float> block (buffer);
        juce::dsp::ProcessContextReplacing<float> ctx (block);
        reverb.process (ctx);
    }
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::togglePlayback()
{
    if (! hasAudio.load()) return;
    if (playing.load()) { playing.store (false); return; }
    if (playbackPosition.load() >= generatedAudio.getNumSamples())
        playbackPosition.store (0);
    playing.store (true);
}

void MLXAudioGenProcessor::stopPlayback()
{
    playing.store (false);
    playbackPosition.store (0);
}

float MLXAudioGenProcessor::getPlaybackProgress() const
{
    if (! hasAudio.load() || generatedAudio.getNumSamples() == 0) return 0.0f;
    return (float) playbackPosition.load() / (float) generatedAudio.getNumSamples();
}

// ---------------------------------------------------------------------------
// DAW integration
// ---------------------------------------------------------------------------

float MLXAudioGenProcessor::getEffectiveBpm() const
{
    return PARAM_BOOL ("dawBpm") ? dawBpm : PARAM_FLOAT ("manualBpm");
}

float MLXAudioGenProcessor::getEffectiveSeconds() const
{
    if (! PARAM_BOOL ("barsMode"))
        return PARAM_FLOAT ("seconds");
    float bpm = getEffectiveBpm();
    if (bpm <= 0.0f) bpm = 120.0f;
    return (float) PARAM_INT ("bars") * 4.0f * (60.0f / bpm);
}

juce::String MLXAudioGenProcessor::buildFullPrompt() const
{
    auto full = prompt;
    if (keySignature.isNotEmpty())
        full += " in " + keySignature;
    float bpm = getEffectiveBpm();
    if (bpm > 0.0f && (PARAM_BOOL ("barsMode") || PARAM_BOOL ("dawBpm")))
        full += ", " + juce::String ((int) bpm) + " BPM";
    return full;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::triggerGeneration()
{
    if (generating.load()) return;
    if (prompt.isEmpty())
    {
        juce::ScopedLock lock (stateLock);
        lastError = "Prompt is required";
        return;
    }
    if (! serverLauncher.isServerAlive())
    {
        juce::ScopedLock lock (stateLock);
        lastError = serverLauncher.getStatus();
        if (lastError.isEmpty()) lastError = "Server not running";
        return;
    }

    generating.store (true);
    progress.store (0.0f);
    playing.store (false);
    { juce::ScopedLock lock (stateLock); statusMessage = "Submitting..."; lastError = {}; }

    generationThread = std::make_unique<GenerationThread> (*this);
    generationThread->startThread();
}

void MLXAudioGenProcessor::runGeneration()
{
    auto* obj = new juce::DynamicObject();
    bool isStable = PARAM_CHOICE ("model") == 1;
    obj->setProperty ("model", isStable ? "stable_audio" : "musicgen");
    obj->setProperty ("prompt", buildFullPrompt());
    obj->setProperty ("seconds", (double) getEffectiveSeconds());

    if (! isStable)
    {
        obj->setProperty ("temperature", (double) PARAM_FLOAT ("temperature"));
        obj->setProperty ("top_k", PARAM_INT ("topK"));
        obj->setProperty ("guidance_coef", (double) PARAM_FLOAT ("guidance"));
    }
    else
    {
        obj->setProperty ("steps", PARAM_INT ("steps"));
        obj->setProperty ("cfg_scale", (double) PARAM_FLOAT ("cfgScale"));
        obj->setProperty ("sampler", PARAM_CHOICE ("sampler") == 1 ? "rk4" : "euler");
        if (negativePrompt.isNotEmpty())
            obj->setProperty ("negative_prompt", negativePrompt);
    }

    int s = PARAM_INT ("seed");
    if (s >= 0) obj->setProperty ("seed", s);

    auto jsonBody = juce::JSON::toString (juce::var (obj));
    auto jobId = httpClient.submitGeneration (jsonBody);

    if (jobId.isEmpty())
    {
        juce::ScopedLock lock (stateLock);
        lastError = "Failed to connect to server";
        statusMessage = "Error";
        generating.store (false);
        return;
    }

    { juce::ScopedLock lock (stateLock); statusMessage = "Generating..."; }

    for (int i = 0; i < 1200; ++i)
    {
        if (juce::Thread::currentThreadShouldExit()) { generating.store (false); return; }
        juce::Thread::sleep (500);

        auto sj = httpClient.fetchStatus (jobId);
        if (sj.isEmpty()) continue;

        auto parsed = juce::JSON::parse (sj);
        if (auto* so = parsed.getDynamicObject())
        {
            auto st = so->getProperty ("status").toString();
            float pv = (float) so->getProperty ("progress");
            progress.store (pv);

            if (st == "done")
            {
                { juce::ScopedLock lock (stateLock); statusMessage = "Downloading..."; }
                auto wav = httpClient.downloadAudio (jobId);
                if (wav.getSize() > 0)
                {
                    auto is = std::make_unique<juce::MemoryInputStream> (wav, false);
                    juce::WavAudioFormat fmt;
                    auto rd = std::unique_ptr<juce::AudioFormatReader> (
                        fmt.createReaderFor (is.release(), true));
                    if (rd)
                    {
                        generatedAudio.setSize ((int) rd->numChannels, (int) rd->lengthInSamples);
                        rd->read (&generatedAudio, 0, (int) rd->lengthInSamples, 0, true, true);
                        playbackPosition.store (0);
                        hasAudio.store (true);
                        playing.store (true);
                        pendingDecision.store (true);
                        float dur = (float) rd->lengthInSamples / (float) rd->sampleRate;
                        { juce::ScopedLock lock (stateLock);
                          statusMessage = "Preview — Keep or Discard? (" + juce::String (dur, 1) + "s)"; }
                    }
                    else
                    { juce::ScopedLock lock (stateLock); lastError = "WAV decode failed"; statusMessage = "Error"; }
                }
                else
                { juce::ScopedLock lock (stateLock); lastError = "Download failed"; statusMessage = "Error"; }

                progress.store (1.0f);
                generating.store (false);
                return;
            }
            else if (st == "error")
            {
                auto em = so->getProperty ("error").toString();
                juce::ScopedLock lock (stateLock);
                lastError = em.isNotEmpty() ? em : "Failed";
                statusMessage = "Error";
                generating.store (false);
                return;
            }

            { juce::ScopedLock lock (stateLock);
              statusMessage = "Generating " + juce::String ((int) (pv * 100)) + "%"; }
        }
    }

    { juce::ScopedLock lock (stateLock); lastError = "Timeout (10 min)"; statusMessage = "Error"; }
    generating.store (false);
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

juce::String MLXAudioGenProcessor::getStatusMessage() const
{ juce::ScopedLock lock (stateLock); return statusMessage; }

juce::String MLXAudioGenProcessor::getLastError() const
{ juce::ScopedLock lock (stateLock); return lastError; }

void MLXAudioGenProcessor::timerCallback() {}

// ---------------------------------------------------------------------------
// Beat-grid trimmer
// ---------------------------------------------------------------------------

float MLXAudioGenProcessor::getSixteenthNoteSamples() const
{
    float bpm = getEffectiveBpm();
    if (bpm <= 0.0f) bpm = 120.0f;
    return (60.0f / bpm / 4.0f) * (float) currentSampleRate;
}

float MLXAudioGenProcessor::getTotalBeats() const
{
    if (! hasAudio.load() || generatedAudio.getNumSamples() == 0) return 0.0f;
    float bpm = getEffectiveBpm();
    if (bpm <= 0.0f) bpm = 120.0f;
    return (float) generatedAudio.getNumSamples() / (float) currentSampleRate * bpm / 60.0f;
}

int MLXAudioGenProcessor::getTrimStartSamples() const
{
    float s16 = getSixteenthNoteSamples();
    return juce::jmax (0, (int) (std::round (trimStartBeats * 4.0f) * s16));
}

int MLXAudioGenProcessor::getTrimEndSamples() const
{
    if (! hasAudio.load()) return 0;
    int total = generatedAudio.getNumSamples();
    if (trimEndBeats < 0.0f) return total;
    float s16 = getSixteenthNoteSamples();
    return juce::jmin (total, (int) (std::round (trimEndBeats * 4.0f) * s16));
}

void MLXAudioGenProcessor::applyTrim()
{
    if (! hasAudio.load()) return;
    int start = getTrimStartSamples(), end = getTrimEndSamples();
    if (start >= end) return;
    int len = end - start, nc = generatedAudio.getNumChannels();
    juce::AudioBuffer<float> trimmed (nc, len);
    for (int ch = 0; ch < nc; ++ch)
        trimmed.copyFrom (ch, 0, generatedAudio, ch, start, len);
    generatedAudio = std::move (trimmed);
    playbackPosition.store (0);
    trimStartBeats = 0.0f;
    trimEndBeats = -1.0f;
    { juce::ScopedLock lock (stateLock);
      statusMessage = "Trimmed to " + juce::String ((float) len / (float) currentSampleRate, 3) + "s"; }
}

// ---------------------------------------------------------------------------
// Preset / Export
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::savePreset (const juce::File& file)
{
    juce::MemoryBlock block;
    getStateInformation (block);
    file.replaceWithText (juce::String::fromUTF8 (
        static_cast<const char*> (block.getData()), (int) block.getSize()));
}

void MLXAudioGenProcessor::loadPreset (const juce::File& file)
{
    auto text = file.loadFileAsString();
    if (text.isNotEmpty())
        setStateInformation (text.toRawUTF8(), text.getNumBytesAsUTF8());
}

void MLXAudioGenProcessor::exportAudio (const juce::File& file)
{
    if (! hasAudio.load() || generatedAudio.getNumSamples() == 0) return;
    file.deleteFile();
    auto stream = file.createOutputStream();
    if (! stream) return;
    juce::WavAudioFormat wav;
    auto writer = std::unique_ptr<juce::AudioFormatWriter> (
        wav.createWriterFor (stream.release(), currentSampleRate,
                             (unsigned int) generatedAudio.getNumChannels(), 32, {}, 0));
    if (writer)
        writer->writeFromAudioSampleBuffer (generatedAudio, 0, generatedAudio.getNumSamples());
}

// ---------------------------------------------------------------------------
// Keep / Discard workflow
// ---------------------------------------------------------------------------

juce::File MLXAudioGenProcessor::writeTempAudio()
{
    if (! hasAudio.load() || generatedAudio.getNumSamples() == 0)
        return {};

    auto tempDir = juce::File::getSpecialLocation (juce::File::tempDirectory)
                       .getChildFile ("mlx-audiogen");
    tempDir.createDirectory();

    auto ts = juce::Time::getCurrentTime().formatted ("%Y%m%d_%H%M%S");
    auto name = instanceName.replaceCharacter (' ', '_') + "_" + ts + ".wav";
    auto file = tempDir.getChildFile (name);

    exportAudio (file);
    lastTempFile = file;
    return file;
}

juce::File MLXAudioGenProcessor::keepAudio()
{
    if (! hasAudio.load()) return {};

    juce::File dest;

    if (exportFolder.isNotEmpty())
    {
        auto dir = juce::File (exportFolder);
        if (dir.isDirectory())
        {
            auto ts = juce::Time::getCurrentTime().formatted ("%Y%m%d_%H%M%S");
            auto name = instanceName.replaceCharacter (' ', '_') + "_" + ts + ".wav";
            dest = dir.getChildFile (name);
        }
    }

    if (dest == juce::File())
    {
        // Fall back to temp directory
        dest = writeTempAudio();
    }
    else
    {
        exportAudio (dest);
    }

    pendingDecision.store (false);

    {
        juce::ScopedLock lock (stateLock);
        statusMessage = "Saved: " + dest.getFileName();
    }

    return dest;
}

void MLXAudioGenProcessor::discardAudio()
{
    playing.store (false);
    hasAudio.store (false);
    playbackPosition.store (0);
    generatedAudio.setSize (0, 0);
    pendingDecision.store (false);

    // Clean up temp file
    if (lastTempFile.existsAsFile())
        lastTempFile.deleteFile();

    {
        juce::ScopedLock lock (stateLock);
        statusMessage = "Discarded";
    }
}

// ---------------------------------------------------------------------------
// State (APVTS handles automatable params; we add non-automatable here)
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    // Add non-automatable state
    state.setProperty ("instanceName", instanceName, nullptr);
    state.setProperty ("prompt", prompt, nullptr);
    state.setProperty ("negativePrompt", negativePrompt, nullptr);
    state.setProperty ("exportFolder", exportFolder, nullptr);
    state.setProperty ("keySignature", keySignature, nullptr);
    state.setProperty ("serverUrl", httpClient.getBaseUrl(), nullptr);

    auto xml = state.createXml();
    if (xml)
        copyXmlToBinary (*xml, destData);
}

juce::AudioProcessorEditor* MLXAudioGenProcessor::createEditor()
{
    return new MLXAudioGenEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MLXAudioGenProcessor();
}

void MLXAudioGenProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary (data, sizeInBytes);
    if (! xml) return;

    auto state = juce::ValueTree::fromXml (*xml);
    if (state.isValid())
    {
        apvts.replaceState (state);
        instanceName = state.getProperty ("instanceName", "MLX AudioGen").toString();
        prompt = state.getProperty ("prompt", "").toString();
        negativePrompt = state.getProperty ("negativePrompt", "").toString();
        exportFolder = state.getProperty ("exportFolder", "").toString();
        keySignature = state.getProperty ("keySignature", "").toString();
        auto url = state.getProperty ("serverUrl", "").toString();
        if (url.isNotEmpty()) httpClient.setBaseUrl (url);
    }
}
