#include "PluginEditor.h"

// ---------------------------------------------------------------------------
// Helper: style a slider for the dark theme
// ---------------------------------------------------------------------------

static void styleSlider (juce::Slider& slider, double min, double max,
                          double interval, double defaultVal)
{
    slider.setRange (min, max, interval);
    slider.setValue (defaultVal);
    slider.setSliderStyle (juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 55, 20);
    slider.setColour (juce::Slider::backgroundColourId,
                      juce::Colour (0xFF2A2A2A));
    slider.setColour (juce::Slider::thumbColourId,
                      juce::Colour (0xFFFF6B35));
    slider.setColour (juce::Slider::trackColourId,
                      juce::Colour (0xFFFF6B35).withAlpha (0.5f));
    slider.setColour (juce::Slider::textBoxTextColourId,
                      juce::Colour (0xFFE8E8E8));
    slider.setColour (juce::Slider::textBoxOutlineColourId,
                      juce::Colour (0xFF2A2A2A));
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

MLXAudioGenEditor::MLXAudioGenEditor (MLXAudioGenProcessor& p)
    : AudioProcessorEditor (&p), audioGenProcessor (p)
{
    setSize (480, 560);

    // Model selector
    modelSelector.addItem ("MusicGen", 1);
    modelSelector.addItem ("Stable Audio", 2);
    modelSelector.setSelectedId (audioGenProcessor.modelType == "stable_audio" ? 2 : 1);
    modelSelector.onChange = [this]
    {
        audioGenProcessor.modelType = modelSelector.getSelectedId() == 2
                                  ? "stable_audio"
                                  : "musicgen";
        updateUIState();
    };
    addAndMakeVisible (modelSelector);

    // Prompt input
    promptInput.setMultiLine (true);
    promptInput.setReturnKeyStartsNewLine (false);
    promptInput.setTextToShowWhenEmpty (
        "Describe the audio you want to generate...",
        juce::Colour (dimTextColour));
    promptInput.setColour (juce::TextEditor::backgroundColourId,
                            juce::Colour (0xFF1A1A1A));
    promptInput.setColour (juce::TextEditor::textColourId,
                            juce::Colour (textColour));
    promptInput.setColour (juce::TextEditor::outlineColourId,
                            juce::Colour (borderColour));
    promptInput.setColour (juce::TextEditor::focusedOutlineColourId,
                            juce::Colour (accentColour));
    promptInput.setText (audioGenProcessor.prompt);
    promptInput.onTextChange = [this]
    {
        audioGenProcessor.prompt = promptInput.getText();
    };
    addAndMakeVisible (promptInput);

    // Duration
    styleSlider (durationSlider, 0.5, 60.0, 0.5, audioGenProcessor.seconds);
    durationSlider.onValueChange = [this]
    {
        audioGenProcessor.seconds = (float) durationSlider.getValue();
    };
    addAndMakeVisible (durationSlider);

    durationLabel.setText ("Duration (s)", juce::dontSendNotification);
    durationLabel.setColour (juce::Label::textColourId,
                              juce::Colour (dimTextColour));
    addAndMakeVisible (durationLabel);

    // MusicGen sliders
    styleSlider (temperatureSlider, 0.1, 2.0, 0.05, audioGenProcessor.temperature);
    temperatureSlider.onValueChange = [this]
    {
        audioGenProcessor.temperature = (float) temperatureSlider.getValue();
    };
    addAndMakeVisible (temperatureSlider);

    styleSlider (topKSlider, 1, 500, 1, audioGenProcessor.topK);
    topKSlider.onValueChange = [this]
    {
        audioGenProcessor.topK = (int) topKSlider.getValue();
    };
    addAndMakeVisible (topKSlider);

    styleSlider (guidanceSlider, 0, 10, 0.1, audioGenProcessor.guidanceCoef);
    guidanceSlider.onValueChange = [this]
    {
        audioGenProcessor.guidanceCoef = (float) guidanceSlider.getValue();
    };
    addAndMakeVisible (guidanceSlider);

    // Stable Audio sliders
    styleSlider (stepsSlider, 1, 100, 1, audioGenProcessor.steps);
    stepsSlider.onValueChange = [this]
    {
        audioGenProcessor.steps = (int) stepsSlider.getValue();
    };
    addAndMakeVisible (stepsSlider);

    styleSlider (cfgScaleSlider, 0, 15, 0.1, audioGenProcessor.cfgScale);
    cfgScaleSlider.onValueChange = [this]
    {
        audioGenProcessor.cfgScale = (float) cfgScaleSlider.getValue();
    };
    addAndMakeVisible (cfgScaleSlider);

    samplerSelector.addItem ("Euler (fast)", 1);
    samplerSelector.addItem ("RK4 (accurate)", 2);
    samplerSelector.setSelectedId (audioGenProcessor.sampler == "rk4" ? 2 : 1);
    samplerSelector.onChange = [this]
    {
        audioGenProcessor.sampler = samplerSelector.getSelectedId() == 2 ? "rk4" : "euler";
    };
    addAndMakeVisible (samplerSelector);

    // Generate button
    generateButton.setColour (juce::TextButton::buttonColourId,
                               juce::Colour (accentColour));
    generateButton.setColour (juce::TextButton::textColourOffId,
                               juce::Colour (0xFF0A0A0A));
    generateButton.onClick = [this] { onGenerateClicked(); };
    addAndMakeVisible (generateButton);

    // Status label
    statusLabel.setColour (juce::Label::textColourId,
                            juce::Colour (dimTextColour));
    statusLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (statusLabel);

    // Error label
    errorLabel.setColour (juce::Label::textColourId,
                           juce::Colour (errorColourVal));
    errorLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (errorLabel);

    updateUIState();
    startTimerHz (10); // 10 Hz UI refresh during generation
}

MLXAudioGenEditor::~MLXAudioGenEditor()
{
    stopTimer();
}

// ---------------------------------------------------------------------------
// Paint
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (bgColour));

    // Draw progress bar
    if (audioGenProcessor.isGenerating())
    {
        auto bounds = getLocalBounds().removeFromBottom (4);
        g.setColour (juce::Colour (borderColour));
        g.fillRect (bounds);
        g.setColour (juce::Colour (accentColour));
        g.fillRect (bounds.removeFromLeft (
            (int) (bounds.getWidth() * displayProgress)));
    }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::resized()
{
    auto area = getLocalBounds().reduced (16);

    // Model selector
    modelSelector.setBounds (area.removeFromTop (28));
    area.removeFromTop (8);

    // Prompt
    promptInput.setBounds (area.removeFromTop (70));
    area.removeFromTop (8);

    // Duration
    auto durRow = area.removeFromTop (24);
    durationLabel.setBounds (durRow.removeFromLeft (90));
    durationSlider.setBounds (durRow);
    area.removeFromTop (8);

    bool isMusicGen = audioGenProcessor.modelType != "stable_audio";

    // MusicGen params
    if (isMusicGen)
    {
        auto makeRow = [&] (juce::Slider& slider)
        {
            auto row = area.removeFromTop (24);
            row.removeFromLeft (90);
            slider.setBounds (row);
            area.removeFromTop (4);
        };

        makeRow (temperatureSlider);
        makeRow (topKSlider);
        makeRow (guidanceSlider);
    }
    else
    {
        auto makeRow = [&] (juce::Slider& slider)
        {
            auto row = area.removeFromTop (24);
            row.removeFromLeft (90);
            slider.setBounds (row);
            area.removeFromTop (4);
        };

        makeRow (stepsSlider);
        makeRow (cfgScaleSlider);

        auto samplerRow = area.removeFromTop (24);
        samplerRow.removeFromLeft (90);
        samplerSelector.setBounds (samplerRow);
        area.removeFromTop (4);
    }

    // Generate button
    area.removeFromTop (8);
    generateButton.setBounds (area.removeFromTop (36));
    area.removeFromTop (8);

    // Status
    statusLabel.setBounds (area.removeFromTop (20));
    errorLabel.setBounds (area.removeFromTop (20));
}

// ---------------------------------------------------------------------------
// UI update
// ---------------------------------------------------------------------------

void MLXAudioGenEditor::updateUIState()
{
    bool isMusicGen = audioGenProcessor.modelType != "stable_audio";

    temperatureSlider.setVisible (isMusicGen);
    topKSlider.setVisible (isMusicGen);
    guidanceSlider.setVisible (isMusicGen);

    stepsSlider.setVisible (! isMusicGen);
    cfgScaleSlider.setVisible (! isMusicGen);
    samplerSelector.setVisible (! isMusicGen);

    resized();
}

void MLXAudioGenEditor::timerCallback()
{
    bool gen = audioGenProcessor.isGenerating();
    displayProgress = audioGenProcessor.getProgress();

    generateButton.setEnabled (! gen);
    generateButton.setButtonText (gen ? "Generating..." : "Generate");

    statusLabel.setText (audioGenProcessor.getStatusMessage(),
                          juce::dontSendNotification);

    auto err = audioGenProcessor.getLastError();
    errorLabel.setText (err, juce::dontSendNotification);
    errorLabel.setVisible (err.isNotEmpty());

    repaint();
}

void MLXAudioGenEditor::onGenerateClicked()
{
    audioGenProcessor.triggerGeneration();
}
