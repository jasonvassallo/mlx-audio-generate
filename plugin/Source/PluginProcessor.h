#pragma once

#include "HttpClient.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>

/**
 * MLX AudioGen plugin processor.
 *
 * This is a synthesizer plugin that generates audio via the mlx-audiogen
 * HTTP server. Generation happens asynchronously on a background thread;
 * completed audio is loaded into a buffer and played back through the
 * DAW's audio bus.
 *
 * Architecture:
 *   User clicks Generate → background thread POSTs to server →
 *   polls /api/status → downloads WAV → loads into AudioBuffer →
 *   processBlock() reads from buffer → DAW output
 */
class MLXAudioGenProcessor : public juce::AudioProcessor,
                              private juce::Timer
{
public:
    MLXAudioGenProcessor();
    ~MLXAudioGenProcessor() override;

    // --- AudioProcessor interface ---
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "MLX AudioGen"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    // --- Generation control ---

    /** Trigger generation with current parameters. Thread-safe. */
    void triggerGeneration();

    /** Is a generation currently in progress? */
    bool isGenerating() const { return generating.load(); }

    /** Get current progress (0.0 to 1.0). */
    float getProgress() const { return progress.load(); }

    /** Get status message for UI display. */
    juce::String getStatusMessage() const;

    /** Get last error message, empty if no error. */
    juce::String getLastError() const;

    // --- Parameters ---
    juce::String prompt;
    juce::String negativePrompt;
    juce::String modelType { "musicgen" };
    float seconds { 5.0f };
    float temperature { 1.0f };
    int topK { 250 };
    float guidanceCoef { 3.0f };
    int steps { 8 };
    float cfgScale { 6.0f };
    juce::String sampler { "euler" };
    int seed { -1 }; // -1 = random

    HttpClient httpClient;

    /** Run the generation loop — called from background thread. */
    void runGeneration();

private:
    void timerCallback() override;

    // Playback state
    juce::AudioBuffer<float> generatedAudio;
    int playbackPosition { 0 };
    bool hasAudio { false };
    double currentSampleRate { 44100.0 };

    // Generation state (thread-safe)
    std::atomic<bool> generating { false };
    std::atomic<float> progress { 0.0f };
    juce::String statusMessage;
    juce::String lastError;
    juce::CriticalSection stateLock;

    // Background thread
    std::unique_ptr<juce::Thread> generationThread;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MLXAudioGenProcessor)
};
