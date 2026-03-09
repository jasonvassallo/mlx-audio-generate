#include "PluginProcessor.h"
#include "PluginEditor.h"

// ---------------------------------------------------------------------------
// Background generation thread
// ---------------------------------------------------------------------------

class GenerationThread : public juce::Thread
{
public:
    GenerationThread (MLXAudioGenProcessor& p)
        : juce::Thread ("MLX Generation"), processor (p)
    {
    }

    void run() override
    {
        processor.runGeneration();
    }

private:
    MLXAudioGenProcessor& processor;
};

// ---------------------------------------------------------------------------
// Processor lifecycle
// ---------------------------------------------------------------------------

MLXAudioGenProcessor::MLXAudioGenProcessor()
    : AudioProcessor (BusesProperties()
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
}

MLXAudioGenProcessor::~MLXAudioGenProcessor()
{
    stopTimer();
    if (generationThread && generationThread->isThreadRunning())
        generationThread->stopThread (5000);
}

void MLXAudioGenProcessor::prepareToPlay (double sampleRate, int /*samplesPerBlock*/)
{
    currentSampleRate = sampleRate;

    // Auto-launch server on first load (runs in background)
    if (! serverLauncher.isServerAlive())
    {
        // Launch on a background thread to avoid blocking the audio thread
        auto* launcher = &serverLauncher;
        auto* self = this;
        juce::Thread::launch ([launcher, self]
        {
            launcher->ensureServerRunning();
            juce::ScopedLock lock (self->stateLock);
            self->statusMessage = launcher->getStatus();
        });
    }
}

void MLXAudioGenProcessor::releaseResources()
{
}

// ---------------------------------------------------------------------------
// Audio processing — plays back generated audio
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& /*midi*/)
{
    buffer.clear();

    if (! hasAudio || generatedAudio.getNumSamples() == 0)
        return;

    const int numChannels = juce::jmin (buffer.getNumChannels(),
                                         generatedAudio.getNumChannels());
    const int numSamples = buffer.getNumSamples();
    const int totalSamples = generatedAudio.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        int remaining = totalSamples - playbackPosition;
        int toCopy = juce::jmin (numSamples, remaining);

        if (toCopy > 0)
        {
            buffer.copyFrom (ch, 0,
                             generatedAudio, ch, playbackPosition, toCopy);
        }
    }

    playbackPosition += numSamples;
    if (playbackPosition >= totalSamples)
        playbackPosition = 0; // Loop
}

// ---------------------------------------------------------------------------
// Generation control
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::triggerGeneration()
{
    if (generating.load())
        return;

    if (prompt.isEmpty())
    {
        juce::ScopedLock lock (stateLock);
        lastError = "Prompt is required";
        return;
    }

    // Ensure server is running before generating
    if (! serverLauncher.isServerAlive())
    {
        juce::ScopedLock lock (stateLock);
        lastError = serverLauncher.getStatus();
        if (lastError.isEmpty())
            lastError = "Server not running. Waiting for auto-start...";
        return;
    }

    generating.store (true);
    progress.store (0.0f);

    {
        juce::ScopedLock lock (stateLock);
        statusMessage = "Submitting...";
        lastError = {};
    }

    // Launch background thread
    generationThread = std::make_unique<GenerationThread> (*this);
    generationThread->startThread();
}

void MLXAudioGenProcessor::runGeneration()
{
    // Build JSON request body
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("model", modelType);
    obj->setProperty ("prompt", prompt);
    obj->setProperty ("seconds", (double) seconds);

    if (modelType == "musicgen")
    {
        obj->setProperty ("temperature", (double) temperature);
        obj->setProperty ("top_k", topK);
        obj->setProperty ("guidance_coef", (double) guidanceCoef);
    }
    else
    {
        obj->setProperty ("steps", steps);
        obj->setProperty ("cfg_scale", (double) cfgScale);
        obj->setProperty ("sampler", sampler);
        if (negativePrompt.isNotEmpty())
            obj->setProperty ("negative_prompt", negativePrompt);
    }

    if (seed >= 0)
        obj->setProperty ("seed", seed);

    juce::var json (obj);
    auto jsonBody = juce::JSON::toString (json);

    // Submit to server
    auto jobId = httpClient.submitGeneration (jsonBody);

    if (jobId.isEmpty())
    {
        juce::ScopedLock lock (stateLock);
        lastError = "Failed to connect to server. Is mlx-audiogen-app running?";
        statusMessage = "Error";
        generating.store (false);
        return;
    }

    {
        juce::ScopedLock lock (stateLock);
        statusMessage = "Generating...";
    }

    // Poll until done (up to 10 minutes)
    const int maxPolls = 1200; // 10 min at 500ms interval
    for (int i = 0; i < maxPolls; ++i)
    {
        if (juce::Thread::currentThreadShouldExit())
        {
            generating.store (false);
            return;
        }

        juce::Thread::sleep (500);

        auto statusJson = httpClient.fetchStatus (jobId);
        if (statusJson.isEmpty())
            continue;

        auto parsed = juce::JSON::parse (statusJson);
        if (auto* statusObj = parsed.getDynamicObject())
        {
            auto status = statusObj->getProperty ("status").toString();
            auto progressVal = (float) statusObj->getProperty ("progress");
            progress.store (progressVal);

            if (status == "done")
            {
                // Download audio
                {
                    juce::ScopedLock lock (stateLock);
                    statusMessage = "Downloading audio...";
                }

                auto wavData = httpClient.downloadAudio (jobId);
                if (wavData.getSize() > 0)
                {
                    // Load WAV into AudioBuffer
                    auto inputStream = std::make_unique<juce::MemoryInputStream> (
                        wavData, false);

                    juce::WavAudioFormat wavFormat;
                    auto reader = std::unique_ptr<juce::AudioFormatReader> (
                        wavFormat.createReaderFor (inputStream.release(), true));

                    if (reader != nullptr)
                    {
                        generatedAudio.setSize (
                            (int) reader->numChannels,
                            (int) reader->lengthInSamples);
                        reader->read (&generatedAudio, 0,
                                      (int) reader->lengthInSamples, 0, true, true);
                        playbackPosition = 0;
                        hasAudio = true;

                        {
                            juce::ScopedLock lock (stateLock);
                            statusMessage = "Ready — audio loaded";
                        }
                    }
                    else
                    {
                        juce::ScopedLock lock (stateLock);
                        lastError = "Failed to decode WAV data";
                        statusMessage = "Error";
                    }
                }
                else
                {
                    juce::ScopedLock lock (stateLock);
                    lastError = "Failed to download audio";
                    statusMessage = "Error";
                }

                progress.store (1.0f);
                generating.store (false);
                return;
            }
            else if (status == "error")
            {
                auto errorMsg = statusObj->getProperty ("error").toString();
                juce::ScopedLock lock (stateLock);
                lastError = errorMsg.isNotEmpty() ? errorMsg : "Generation failed";
                statusMessage = "Error";
                generating.store (false);
                return;
            }

            // Still running — update status
            {
                juce::ScopedLock lock (stateLock);
                statusMessage = juce::String ("Generating... ")
                                + juce::String ((int) (progressVal * 100)) + "%";
            }
        }
    }

    // Timeout
    {
        juce::ScopedLock lock (stateLock);
        lastError = "Generation timed out (10 minutes)";
        statusMessage = "Error";
    }
    generating.store (false);
}

// ---------------------------------------------------------------------------
// Status accessors
// ---------------------------------------------------------------------------

juce::String MLXAudioGenProcessor::getStatusMessage() const
{
    juce::ScopedLock lock (stateLock);
    return statusMessage;
}

juce::String MLXAudioGenProcessor::getLastError() const
{
    juce::ScopedLock lock (stateLock);
    return lastError;
}

// ---------------------------------------------------------------------------
// Timer — triggers UI repaints during generation
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::timerCallback()
{
    // Editor polls this — nothing needed here
}

// ---------------------------------------------------------------------------
// State save/restore (DAW session persistence)
// ---------------------------------------------------------------------------

void MLXAudioGenProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("model", modelType);
    obj->setProperty ("prompt", prompt);
    obj->setProperty ("negativePrompt", negativePrompt);
    obj->setProperty ("seconds", (double) seconds);
    obj->setProperty ("temperature", (double) temperature);
    obj->setProperty ("topK", topK);
    obj->setProperty ("guidanceCoef", (double) guidanceCoef);
    obj->setProperty ("steps", steps);
    obj->setProperty ("cfgScale", (double) cfgScale);
    obj->setProperty ("sampler", sampler);
    obj->setProperty ("seed", seed);
    obj->setProperty ("serverUrl", httpClient.getBaseUrl());

    juce::var json (obj);
    auto text = juce::JSON::toString (json);
    destData.replaceAll (text.toRawUTF8(), text.getNumBytesAsUTF8());
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
    auto text = juce::String::fromUTF8 (static_cast<const char*> (data), sizeInBytes);
    auto parsed = juce::JSON::parse (text);

    if (auto* obj = parsed.getDynamicObject())
    {
        modelType = obj->getProperty ("model").toString();
        prompt = obj->getProperty ("prompt").toString();
        negativePrompt = obj->getProperty ("negativePrompt").toString();
        seconds = (float) obj->getProperty ("seconds");
        temperature = (float) obj->getProperty ("temperature");
        topK = (int) obj->getProperty ("topK");
        guidanceCoef = (float) obj->getProperty ("guidanceCoef");
        steps = (int) obj->getProperty ("steps");
        cfgScale = (float) obj->getProperty ("cfgScale");
        sampler = obj->getProperty ("sampler").toString();
        seed = (int) obj->getProperty ("seed");

        auto serverUrl = obj->getProperty ("serverUrl").toString();
        if (serverUrl.isNotEmpty())
            httpClient.setBaseUrl (serverUrl);
    }
}
