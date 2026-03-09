#pragma once

#include "HttpClient.h"
#include "ServerLauncher.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>

/**
 * MLX AudioGen plugin processor.
 *
 * Synthesizer plugin that generates audio via the mlx-audiogen HTTP server.
 * Supports BPM sync with DAW, key signature awareness, MIDI trigger,
 * and looped playback with bar-aligned trim.
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
    void triggerGeneration();
    bool isGenerating() const { return generating.load(); }
    float getProgress() const { return progress.load(); }
    juce::String getStatusMessage() const;
    juce::String getLastError() const;
    void runGeneration();

    // --- Playback control ---
    void togglePlayback();
    void stopPlayback();
    bool isPlaying() const { return playing.load(); }
    bool hasAudioLoaded() const { return hasAudio.load(); }
    float getPlaybackProgress() const;

    // --- Audio buffer access (for waveform drawing) ---
    const juce::AudioBuffer<float>& getGeneratedAudio() const { return generatedAudio; }

    // --- Instance identity (for multi-instance) ---
    juce::String instanceName { "MLX AudioGen" };

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
    int seed { -1 };

    // --- DAW integration (Phase 4b) ---
    bool useDawBpm { true };        // Auto-read BPM from DAW
    float manualBpm { 120.0f };     // Manual BPM when not synced
    int bars { 4 };                 // Duration in bars (when using bar mode)
    bool useBarsMode { false };     // true = use bars+BPM, false = use seconds
    juce::String keySignature;      // e.g. "A minor", "C major" — appended to prompt
    bool midiTrigger { false };     // Generate on MIDI note-on
    bool looping { true };          // Loop playback

    /** Get effective BPM (from DAW or manual setting). */
    float getEffectiveBpm() const;

    /** Get effective duration in seconds (accounting for bar mode). */
    float getEffectiveSeconds() const;

    HttpClient httpClient;
    ServerLauncher serverLauncher;

private:
    void timerCallback() override;

    /** Build the full prompt including key signature suffix. */
    juce::String buildFullPrompt() const;

    // Playback state
    juce::AudioBuffer<float> generatedAudio;
    std::atomic<int> playbackPosition { 0 };
    std::atomic<bool> hasAudio { false };
    std::atomic<bool> playing { false };
    double currentSampleRate { 44100.0 };
    float dawBpm { 120.0f };

    // Generation state
    std::atomic<bool> generating { false };
    std::atomic<float> progress { 0.0f };
    juce::String statusMessage;
    juce::String lastError;
    juce::CriticalSection stateLock;

    std::unique_ptr<juce::Thread> generationThread;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MLXAudioGenProcessor)
};
