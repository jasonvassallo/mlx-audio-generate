#pragma once

#include "HttpClient.h"
#include "ServerLauncher.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

/**
 * MLX AudioGen plugin processor.
 *
 * All generation parameters are exposed via AudioProcessorValueTreeState
 * (APVTS) for Push 2 compatibility, DAW automation, and MIDI controller
 * mapping. Push 2 automatically discovers parameters and maps them to
 * its knobs and display.
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

    // --- APVTS: exposes parameters to DAW, Push 2, MIDI controllers ---
    juce::AudioProcessorValueTreeState apvts;
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // --- Generation control ---
    void triggerGeneration();
    bool isGenerating() const { return generating.load(); }
    float getProgress() const { return progress.load(); }
    juce::String getStatusMessage() const;
    juce::String getLastError() const;
    void runGeneration();

    // --- Playback ---
    void togglePlayback();
    void stopPlayback();
    bool isPlaying() const { return playing.load(); }
    bool hasAudioLoaded() const { return hasAudio.load(); }
    float getPlaybackProgress() const;
    const juce::AudioBuffer<float>& getGeneratedAudio() const { return generatedAudio; }

    // --- Non-automatable state (not in APVTS) ---
    juce::String instanceName { "MLX AudioGen" };
    juce::String prompt;
    juce::String negativePrompt;
    juce::String exportFolder;
    juce::String keySignature;

    // --- Beat-grid trimmer ---
    float trimStartBeats { 0.0f };
    float trimEndBeats { -1.0f };
    int getTrimStartSamples() const;
    int getTrimEndSamples() const;
    float getSixteenthNoteSamples() const;
    float getTotalBeats() const;
    void applyTrim();

    /** Get effective BPM (from DAW or manual setting). */
    float getEffectiveBpm() const;
    /** Get effective duration in seconds (accounting for bar mode). */
    float getEffectiveSeconds() const;

    // --- Keep / Discard workflow ---
    /** Save current audio to export folder (or temp dir). Returns the file path. */
    juce::File keepAudio();
    /** Discard current audio — clears the buffer. */
    void discardAudio();
    /** Whether audio is pending keep/discard decision. */
    bool isPendingDecision() const { return pendingDecision.load(); }

    // --- Drag-and-drop ---
    /** Write current audio to a temp file for drag operations. Returns path. */
    juce::File writeTempAudio();

    // --- Preset / Export ---
    void savePreset (const juce::File& file);
    void loadPreset (const juce::File& file);
    void exportAudio (const juce::File& file);

    HttpClient httpClient;
    ServerLauncher serverLauncher;

private:
    void timerCallback() override;
    juce::String buildFullPrompt() const;

    void updateEffectsParameters();
    void applyEffects (juce::AudioBuffer<float>& buffer);

    // Playback state
    juce::AudioBuffer<float> generatedAudio;
    std::atomic<int> playbackPosition { 0 };
    std::atomic<bool> hasAudio { false };
    std::atomic<bool> playing { false };
    std::atomic<bool> pendingDecision { false };
    double currentSampleRate { 44100.0 };
    float dawBpm { 120.0f };

    // Generation state
    std::atomic<bool> generating { false };
    std::atomic<float> progress { 0.0f };
    juce::String statusMessage;
    juce::String lastError;
    juce::CriticalSection stateLock;

    std::unique_ptr<juce::Thread> generationThread;

    // DSP
    juce::dsp::DelayLine<float> delayLine { 48000 };
    juce::dsp::Reverb reverb;
    juce::File lastTempFile;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MLXAudioGenProcessor)
};
