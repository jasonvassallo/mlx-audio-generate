#pragma once

#include "PluginProcessor.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

/**
 * MLX AudioGen plugin editor — DAW-embedded UI.
 *
 * Dark theme matching the Web UI aesthetic. Contains:
 *   - Model selector (MusicGen / Stable Audio)
 *   - Text prompt input
 *   - Duration slider
 *   - Model-specific parameter controls
 *   - Generate button with progress bar
 *   - Status display
 */
class MLXAudioGenEditor : public juce::AudioProcessorEditor,
                           private juce::Timer
{
public:
    explicit MLXAudioGenEditor (MLXAudioGenProcessor&);
    ~MLXAudioGenEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void updateUIState();
    void onGenerateClicked();

    MLXAudioGenProcessor& audioGenProcessor;

    // UI Components
    juce::ComboBox modelSelector;
    juce::TextEditor promptInput;
    juce::Slider durationSlider;
    juce::Label durationLabel;

    // MusicGen params
    juce::Slider temperatureSlider;
    juce::Slider topKSlider;
    juce::Slider guidanceSlider;

    // Stable Audio params
    juce::Slider stepsSlider;
    juce::Slider cfgScaleSlider;
    juce::ComboBox samplerSelector;

    // Controls
    juce::TextButton generateButton { "Generate" };
    juce::Label statusLabel;
    juce::Label errorLabel;

    // Progress bar
    float displayProgress { 0.0f };

    // Dark theme colours
    static constexpr juce::uint32 bgColour       = 0xFF0A0A0A;
    static constexpr juce::uint32 panelColour     = 0xFF111111;
    static constexpr juce::uint32 borderColour    = 0xFF2A2A2A;
    static constexpr juce::uint32 textColour      = 0xFFE8E8E8;
    static constexpr juce::uint32 dimTextColour   = 0xFF888888;
    static constexpr juce::uint32 accentColour    = 0xFFFF6B35;
    static constexpr juce::uint32 errorColourVal  = 0xFFF87171;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MLXAudioGenEditor)
};
