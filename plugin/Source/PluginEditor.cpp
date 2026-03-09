#include "PluginEditor.h"

static void styleSlider (juce::Slider& s, juce::Slider::TextEntryBoxPosition tp = juce::Slider::TextBoxRight)
{
    s.setSliderStyle (juce::Slider::LinearHorizontal);
    s.setTextBoxStyle (tp, false, 50, 18);
    s.setColour (juce::Slider::backgroundColourId,    juce::Colour (0xFF2A2A2A));
    s.setColour (juce::Slider::thumbColourId,          juce::Colour (0xFFFF6B35));
    s.setColour (juce::Slider::trackColourId,          juce::Colour (0xFFFF6B35).withAlpha (0.5f));
    s.setColour (juce::Slider::textBoxTextColourId,    juce::Colour (0xFFE8E8E8));
    s.setColour (juce::Slider::textBoxOutlineColourId, juce::Colour (0xFF2A2A2A));
}

static void styleToggle (juce::ToggleButton& t) {
    t.setColour (juce::ToggleButton::textColourId, juce::Colour (0xFF888888));
    t.setColour (juce::ToggleButton::tickColourId, juce::Colour (0xFFFF6B35));
}

static void styleCombo (juce::ComboBox& c) {
    c.setColour (juce::ComboBox::backgroundColourId, juce::Colour (0xFF1A1A1A));
    c.setColour (juce::ComboBox::textColourId, juce::Colour (0xFFE8E8E8));
    c.setColour (juce::ComboBox::outlineColourId, juce::Colour (0xFF2A2A2A));
}

static void styleBtn (juce::TextButton& b) {
    b.setColour (juce::TextButton::buttonColourId, juce::Colour (0xFF1A1A1A));
    b.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFFE8E8E8));
}

static const juce::StringArray KEY_OPTIONS = {
    "", "C major", "C minor", "C# major", "C# minor",
    "D major", "D minor", "Eb major", "Eb minor",
    "E major", "E minor", "F major", "F minor",
    "F# major", "F# minor", "G major", "G minor",
    "Ab major", "Ab minor", "A major", "A minor",
    "Bb major", "Bb minor", "B major", "B minor"
};

MLXAudioGenEditor::MLXAudioGenEditor (MLXAudioGenProcessor& p)
    : AudioProcessorEditor (&p), proc (p)
{
    setSize (520, 950);

    // Instance name
    instanceNameInput.setColour (juce::TextEditor::backgroundColourId, juce::Colour (bgColour));
    instanceNameInput.setColour (juce::TextEditor::textColourId, juce::Colour (textColour));
    instanceNameInput.setColour (juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
    instanceNameInput.setFont (juce::Font (14.0f, juce::Font::bold));
    instanceNameInput.setText (proc.instanceName);
    instanceNameInput.onTextChange = [this] { proc.instanceName = instanceNameInput.getText(); };
    addAndMakeVisible (instanceNameInput);

    // Model
    modelSelector.addItem ("MusicGen", 1);
    modelSelector.addItem ("Stable Audio", 2);
    styleCombo (modelSelector);
    addAndMakeVisible (modelSelector);
    modelAttach = std::make_unique<ComboAttach> (proc.apvts, "model", modelSelector);
    modelSelector.onChange = [this] { updateUIState(); };

    // Prompt
    promptInput.setMultiLine (true);
    promptInput.setReturnKeyStartsNewLine (false);
    promptInput.setTextToShowWhenEmpty ("Describe the audio...", juce::Colour (dimTextColour));
    promptInput.setColour (juce::TextEditor::backgroundColourId, juce::Colour (surfaceColour));
    promptInput.setColour (juce::TextEditor::textColourId, juce::Colour (textColour));
    promptInput.setColour (juce::TextEditor::outlineColourId, juce::Colour (borderColour));
    promptInput.setColour (juce::TextEditor::focusedOutlineColourId, juce::Colour (accentColour));
    promptInput.setText (proc.prompt);
    promptInput.onTextChange = [this] { proc.prompt = promptInput.getText(); };
    addAndMakeVisible (promptInput);

    // Key
    keySelector.addItem ("(no key)", 1);
    for (int i = 1; i < KEY_OPTIONS.size(); ++i)
        keySelector.addItem (KEY_OPTIONS[i], i + 1);
    int ki = KEY_OPTIONS.indexOf (proc.keySignature);
    keySelector.setSelectedId (ki >= 0 ? ki + 1 : 1);
    keySelector.onChange = [this] {
        int idx = keySelector.getSelectedId() - 1;
        proc.keySignature = (idx > 0 && idx < KEY_OPTIONS.size()) ? KEY_OPTIONS[idx] : "";
    };
    styleCombo (keySelector);
    addAndMakeVisible (keySelector);

    // Duration controls
    styleToggle (barsModeToggle);
    addAndMakeVisible (barsModeToggle);
    barsModeAttach = std::make_unique<ButtonAttach> (proc.apvts, "barsMode", barsModeToggle);
    barsModeToggle.onClick = [this] { updateUIState(); };

    styleSlider (durationSlider); addAndMakeVisible (durationSlider);
    durationAttach = std::make_unique<SliderAttach> (proc.apvts, "seconds", durationSlider);
    durationLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    durationLabel.setFont (juce::Font (11.0f));
    addAndMakeVisible (durationLabel);

    styleSlider (barsSlider); addAndMakeVisible (barsSlider);
    barsAttach = std::make_unique<SliderAttach> (proc.apvts, "bars", barsSlider);
    barsLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    barsLabel.setFont (juce::Font (11.0f));
    addAndMakeVisible (barsLabel);

    // BPM
    styleToggle (dawBpmToggle); addAndMakeVisible (dawBpmToggle);
    dawBpmAttach = std::make_unique<ButtonAttach> (proc.apvts, "dawBpm", dawBpmToggle);
    dawBpmToggle.onClick = [this] { updateUIState(); };

    styleSlider (bpmSlider); addAndMakeVisible (bpmSlider);
    bpmAttach = std::make_unique<SliderAttach> (proc.apvts, "manualBpm", bpmSlider);
    bpmLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    addAndMakeVisible (bpmLabel);
    bpmDisplay.setColour (juce::Label::textColourId, juce::Colour (accentColour));
    bpmDisplay.setFont (juce::Font (13.0f, juce::Font::bold));
    bpmDisplay.setJustificationType (juce::Justification::centredRight);
    addAndMakeVisible (bpmDisplay);

    // MusicGen params
    styleSlider (temperatureSlider); addAndMakeVisible (temperatureSlider);
    tempAttach = std::make_unique<SliderAttach> (proc.apvts, "temperature", temperatureSlider);

    styleSlider (topKSlider); addAndMakeVisible (topKSlider);
    topKAttach = std::make_unique<SliderAttach> (proc.apvts, "topK", topKSlider);

    styleSlider (guidanceSlider); addAndMakeVisible (guidanceSlider);
    guidanceAttach = std::make_unique<SliderAttach> (proc.apvts, "guidance", guidanceSlider);

    // Stable Audio
    styleSlider (stepsSlider); addAndMakeVisible (stepsSlider);
    stepsAttach = std::make_unique<SliderAttach> (proc.apvts, "steps", stepsSlider);

    styleSlider (cfgScaleSlider); addAndMakeVisible (cfgScaleSlider);
    cfgAttach = std::make_unique<SliderAttach> (proc.apvts, "cfgScale", cfgScaleSlider);

    samplerSelector.addItem ("Euler", 1);
    samplerSelector.addItem ("RK4", 2);
    styleCombo (samplerSelector);
    addAndMakeVisible (samplerSelector);
    samplerAttach = std::make_unique<ComboAttach> (proc.apvts, "sampler", samplerSelector);

    // Seed
    styleSlider (seedSlider); addAndMakeVisible (seedSlider);
    seedAttach = std::make_unique<SliderAttach> (proc.apvts, "seed", seedSlider);

    // Transport
    styleToggle (loopToggle); addAndMakeVisible (loopToggle);
    loopAttach = std::make_unique<ButtonAttach> (proc.apvts, "loop", loopToggle);

    styleToggle (midiTriggerToggle); addAndMakeVisible (midiTriggerToggle);
    midiAttach = std::make_unique<ButtonAttach> (proc.apvts, "midiTrigger", midiTriggerToggle);

    generateButton.setColour (juce::TextButton::buttonColourId, juce::Colour (accentColour));
    generateButton.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFF0A0A0A));
    generateButton.onClick = [this] { onGenerateClicked(); };
    addAndMakeVisible (generateButton);

    styleBtn (playButton); playButton.onClick = [this] { proc.togglePlayback(); };
    addAndMakeVisible (playButton);
    styleBtn (stopButton); stopButton.onClick = [this] { proc.stopPlayback(); };
    addAndMakeVisible (stopButton);

    // Keep / Discard
    keepButton.setColour (juce::TextButton::buttonColourId, juce::Colour (successColour));
    keepButton.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFF0A0A0A));
    keepButton.onClick = [this] { proc.keepAudio(); };
    addAndMakeVisible (keepButton);

    discardButton.setColour (juce::TextButton::buttonColourId, juce::Colour (errorColourVal));
    discardButton.setColour (juce::TextButton::textColourOffId, juce::Colour (0xFF0A0A0A));
    discardButton.onClick = [this] { proc.discardAudio(); };
    addAndMakeVisible (discardButton);

    // Drag to DAW
    styleBtn (dragButton);
    dragButton.onClick = [this] {
        auto file = proc.writeTempAudio();
        if (file.existsAsFile())
        {
            performExternalDragDropOfFiles (
                juce::StringArray { file.getFullPathName() }, false, this);
        }
    };
    addAndMakeVisible (dragButton);

    // Output gain + loop crossfade
    styleSlider (outputGainSlider); addAndMakeVisible (outputGainSlider);
    gainAttach = std::make_unique<SliderAttach> (proc.apvts, "outputGain", outputGainSlider);

    styleSlider (loopFadeSlider); addAndMakeVisible (loopFadeSlider);
    fadeAttach = std::make_unique<SliderAttach> (proc.apvts, "loopFade", loopFadeSlider);

    // Effects
    styleToggle (fxToggle); addAndMakeVisible (fxToggle);
    fxAttach = std::make_unique<ButtonAttach> (proc.apvts, "fxEnabled", fxToggle);
    fxToggle.onClick = [this] { updateUIState(); };

    styleSlider (compThresholdSlider); addAndMakeVisible (compThresholdSlider);
    compTAttach = std::make_unique<SliderAttach> (proc.apvts, "compThreshold", compThresholdSlider);
    styleSlider (compRatioSlider); addAndMakeVisible (compRatioSlider);
    compRAttach = std::make_unique<SliderAttach> (proc.apvts, "compRatio", compRatioSlider);
    styleSlider (delayTimeSlider); addAndMakeVisible (delayTimeSlider);
    delayTAttach = std::make_unique<SliderAttach> (proc.apvts, "delayTime", delayTimeSlider);
    styleSlider (delayMixSlider); addAndMakeVisible (delayMixSlider);
    delayMxAttach = std::make_unique<SliderAttach> (proc.apvts, "delayMix", delayMixSlider);
    styleSlider (reverbSizeSlider); addAndMakeVisible (reverbSizeSlider);
    revSizeAttach = std::make_unique<SliderAttach> (proc.apvts, "reverbSize", reverbSizeSlider);
    styleSlider (reverbMixSlider); addAndMakeVisible (reverbMixSlider);
    revMixAttach = std::make_unique<SliderAttach> (proc.apvts, "reverbMix", reverbMixSlider);

    // Trim
    styleSlider (trimStartSlider); trimStartSlider.setRange (0, 32, 0.25);
    trimStartSlider.onValueChange = [this] { proc.trimStartBeats = (float) trimStartSlider.getValue(); };
    addAndMakeVisible (trimStartSlider);
    styleSlider (trimEndSlider); trimEndSlider.setRange (0, 32, 0.25);
    trimEndSlider.onValueChange = [this] {
        float v = (float) trimEndSlider.getValue();
        proc.trimEndBeats = v > 0.0f ? v : -1.0f;
    };
    addAndMakeVisible (trimEndSlider);
    styleBtn (trimButton); trimButton.onClick = [this] { proc.applyTrim(); };
    addAndMakeVisible (trimButton);
    trimInfoLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    trimInfoLabel.setFont (juce::Font (10.0f));
    addAndMakeVisible (trimInfoLabel);

    // Preset / Export
    styleBtn (savePresetButton);
    savePresetButton.onClick = [this] {
        auto c = std::make_shared<juce::FileChooser> ("Save Preset", juce::File(), "*.mlxpreset");
        c->launchAsync (juce::FileBrowserComponent::saveMode, [this, c] (const auto& fc) {
            auto f = fc.getResult();
            if (f != juce::File()) proc.savePreset (f.withFileExtension ("mlxpreset"));
        });
    };
    addAndMakeVisible (savePresetButton);

    styleBtn (loadPresetButton);
    loadPresetButton.onClick = [this] {
        auto c = std::make_shared<juce::FileChooser> ("Load Preset", juce::File(), "*.mlxpreset");
        c->launchAsync (juce::FileBrowserComponent::openMode, [this, c] (const auto& fc) {
            auto f = fc.getResult();
            if (f != juce::File()) { proc.loadPreset (f); promptInput.setText (proc.prompt); updateUIState(); }
        });
    };
    addAndMakeVisible (loadPresetButton);

    styleBtn (exportAudioButton);
    exportAudioButton.onClick = [this] {
        if (proc.exportFolder.isNotEmpty()) {
            auto dir = juce::File (proc.exportFolder);
            if (dir.isDirectory()) {
                auto ts = juce::Time::getCurrentTime().formatted ("%Y%m%d_%H%M%S");
                proc.exportAudio (dir.getChildFile (proc.instanceName.replaceCharacter (' ', '_') + "_" + ts + ".wav"));
                return;
            }
        }
        auto c = std::make_shared<juce::FileChooser> ("Export", juce::File(), "*.wav");
        c->launchAsync (juce::FileBrowserComponent::saveMode, [this, c] (const auto& fc) {
            auto f = fc.getResult();
            if (f != juce::File()) proc.exportAudio (f.withFileExtension ("wav"));
        });
    };
    addAndMakeVisible (exportAudioButton);

    styleBtn (setFolderButton);
    setFolderButton.onClick = [this] {
        auto c = std::make_shared<juce::FileChooser> ("Export Folder",
            proc.exportFolder.isNotEmpty() ? juce::File (proc.exportFolder) : juce::File());
        c->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
            [this, c] (const auto& fc) {
                auto f = fc.getResult();
                if (f != juce::File() && f.isDirectory()) {
                    proc.exportFolder = f.getFullPathName();
                    folderLabel.setText (f.getFileName(), juce::dontSendNotification);
                }
            });
    };
    addAndMakeVisible (setFolderButton);

    folderLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    folderLabel.setFont (juce::Font (10.0f));
    folderLabel.setText (proc.exportFolder.isNotEmpty()
        ? juce::File (proc.exportFolder).getFileName() : "(ask each time)",
        juce::dontSendNotification);
    addAndMakeVisible (folderLabel);

    statusLabel.setColour (juce::Label::textColourId, juce::Colour (dimTextColour));
    statusLabel.setFont (juce::Font (11.0f));
    statusLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (statusLabel);
    errorLabel.setColour (juce::Label::textColourId, juce::Colour (errorColourVal));
    errorLabel.setFont (juce::Font (11.0f));
    errorLabel.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (errorLabel);

    updateUIState();
    startTimerHz (15);
}

MLXAudioGenEditor::~MLXAudioGenEditor() { stopTimer(); }

// ---------------------------------------------------------------------------

void MLXAudioGenEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (bgColour));
    g.setColour (juce::Colour (borderColour));
    g.drawHorizontalLine (waveformBounds.getY() - 4, 14.0f, (float) getWidth() - 14.0f);
    drawWaveform (g, waveformBounds);
    if (proc.isGenerating()) {
        auto bar = getLocalBounds().removeFromBottom (3);
        g.setColour (juce::Colour (borderColour)); g.fillRect (bar);
        g.setColour (juce::Colour (accentColour));
        g.fillRect (bar.removeFromLeft ((int) (bar.getWidth() * displayProgress)));
    }
}

void MLXAudioGenEditor::drawWaveform (juce::Graphics& g, juce::Rectangle<int> bounds)
{
    g.setColour (juce::Colour (panelColour));
    g.fillRoundedRectangle (bounds.toFloat(), 4.0f);

    const auto& audio = proc.getGeneratedAudio();
    if (audio.getNumSamples() == 0) {
        g.setColour (juce::Colour (dimTextColour).withAlpha (0.3f));
        g.drawText ("No audio", bounds, juce::Justification::centred);
        return;
    }

    const float* samples = audio.getReadPointer (0);
    const int numSamples = audio.getNumSamples();
    const float w = (float) bounds.getWidth();
    const float h = (float) bounds.getHeight();
    const float midY = (float) bounds.getCentreY();

    // Waveform
    g.setColour (juce::Colour (accentColour).withAlpha (0.7f));
    juce::Path path;
    for (int x = 0; x < (int) w; ++x) {
        int s0 = (int) ((float) x / w * numSamples);
        int s1 = juce::jmin ((int) ((float) (x + 1) / w * numSamples), numSamples);
        float pk = 0.0f;
        for (int s = s0; s < s1; ++s) pk = juce::jmax (pk, std::abs (samples[s]));
        float y = pk * h * 0.45f;
        float px = (float) bounds.getX() + (float) x;
        if (x == 0) path.startNewSubPath (px, midY - y);
        else path.lineTo (px, midY - y);
    }
    for (int x = (int) w - 1; x >= 0; --x) {
        int s0 = (int) ((float) x / w * numSamples);
        int s1 = juce::jmin ((int) ((float) (x + 1) / w * numSamples), numSamples);
        float pk = 0.0f;
        for (int s = s0; s < s1; ++s) pk = juce::jmax (pk, std::abs (samples[s]));
        float y = pk * h * 0.45f;
        path.lineTo ((float) bounds.getX() + (float) x, midY + y);
    }
    path.closeSubPath();
    g.fillPath (path);

    // Beat grid
    float totalBeats = proc.getTotalBeats();
    if (totalBeats > 0.0f) {
        float s16total = totalBeats * 4.0f;
        for (int s = 0; s <= (int) s16total; ++s) {
            float frac = (float) s / s16total;
            int lx = bounds.getX() + (int) (frac * w);
            if (s % 16 == 0) g.setColour (juce::Colour (textColour).withAlpha (0.4f));
            else if (s % 4 == 0) g.setColour (juce::Colour (dimTextColour).withAlpha (0.25f));
            else g.setColour (juce::Colour (dimTextColour).withAlpha (0.1f));
            g.drawVerticalLine (lx, (float) bounds.getY(), (float) bounds.getBottom());
        }
    }

    // Trim region
    int ts = proc.getTrimStartSamples(), te = proc.getTrimEndSamples();
    if (ts > 0 || (te < numSamples && te > 0)) {
        float sf = (float) ts / (float) numSamples, ef = (float) te / (float) numSamples;
        g.setColour (juce::Colour (bgColour).withAlpha (0.6f));
        if (ts > 0) g.fillRect (bounds.getX(), bounds.getY(), (int) (sf * w), bounds.getHeight());
        if (te < numSamples) g.fillRect (bounds.getX() + (int) (ef * w), bounds.getY(), (int) ((1.0f - ef) * w), bounds.getHeight());
        g.setColour (juce::Colour (successColour));
        g.drawVerticalLine (bounds.getX() + (int) (sf * w), (float) bounds.getY(), (float) bounds.getBottom());
        g.drawVerticalLine (bounds.getX() + (int) (ef * w), (float) bounds.getY(), (float) bounds.getBottom());
    }

    // Playback position
    if (proc.hasAudioLoaded()) {
        int lx = bounds.getX() + (int) (proc.getPlaybackProgress() * w);
        g.setColour (juce::Colour (textColour));
        g.drawVerticalLine (lx, (float) bounds.getY(), (float) bounds.getBottom());
    }
}

// ---------------------------------------------------------------------------

void MLXAudioGenEditor::resized()
{
    auto area = getLocalBounds().reduced (14);
    const int rh = 22, gap = 4, lw = 70;

    instanceNameInput.setBounds (area.removeFromTop (20)); area.removeFromTop (3);
    auto topRow = area.removeFromTop (26);
    modelSelector.setBounds (topRow.removeFromLeft (topRow.getWidth() / 2 - 2));
    topRow.removeFromLeft (4);
    keySelector.setBounds (topRow);
    area.removeFromTop (gap);

    promptInput.setBounds (area.removeFromTop (50)); area.removeFromTop (gap);

    // Duration
    auto dr = area.removeFromTop (rh);
    barsModeToggle.setBounds (dr.removeFromLeft (55));
    if (barsModeToggle.getToggleState()) { barsLabel.setBounds (dr.removeFromLeft (35)); barsSlider.setBounds (dr); }
    else { durationLabel.setBounds (dr.removeFromLeft (25)); durationSlider.setBounds (dr); }
    area.removeFromTop (gap);

    // BPM
    auto br = area.removeFromTop (rh);
    dawBpmToggle.setBounds (br.removeFromLeft (85));
    bpmDisplay.setBounds (br.removeFromRight (60));
    if (! dawBpmToggle.getToggleState()) { bpmLabel.setBounds (br.removeFromLeft (30)); bpmSlider.setBounds (br); }
    area.removeFromTop (gap);

    bool mg = modelSelector.getSelectedId() != 2;
    auto sr = [&] (juce::Slider& s) { auto r = area.removeFromTop (rh); r.removeFromLeft (lw); s.setBounds (r); area.removeFromTop (3); };
    if (mg) { sr (temperatureSlider); sr (topKSlider); sr (guidanceSlider); }
    else { sr (stepsSlider); sr (cfgScaleSlider); auto r = area.removeFromTop (rh); r.removeFromLeft (lw); samplerSelector.setBounds (r); area.removeFromTop (3); }

    auto seedR = area.removeFromTop (rh); seedR.removeFromLeft (lw); seedSlider.setBounds (seedR); area.removeFromTop (gap);
    auto optR = area.removeFromTop (rh);
    midiTriggerToggle.setBounds (optR.removeFromLeft (110));
    loopToggle.setBounds (optR.removeFromLeft (70));
    area.removeFromTop (gap);

    generateButton.setBounds (area.removeFromTop (30)); area.removeFromTop (gap);
    auto tr = area.removeFromTop (26);
    playButton.setBounds (tr.removeFromLeft (60)); tr.removeFromLeft (3);
    stopButton.setBounds (tr.removeFromLeft (50)); tr.removeFromLeft (3);
    keepButton.setBounds (tr.removeFromLeft (55)); tr.removeFromLeft (3);
    discardButton.setBounds (tr.removeFromLeft (60)); tr.removeFromLeft (3);
    dragButton.setBounds (tr);
    area.removeFromTop (gap);

    // Output gain + loop crossfade
    auto gainRow = area.removeFromTop (rh); gainRow.removeFromLeft (lw);
    outputGainSlider.setBounds (gainRow); area.removeFromTop (3);
    auto fadeRow = area.removeFromTop (rh); fadeRow.removeFromLeft (lw);
    loopFadeSlider.setBounds (fadeRow); area.removeFromTop (gap);

    // Trim
    auto t1 = area.removeFromTop (rh); t1.removeFromLeft (lw); trimStartSlider.setBounds (t1); area.removeFromTop (3);
    auto t2 = area.removeFromTop (rh); t2.removeFromLeft (lw); trimEndSlider.setBounds (t2); area.removeFromTop (3);
    auto t3 = area.removeFromTop (18);
    trimButton.setBounds (t3.removeFromLeft (50)); t3.removeFromLeft (4);
    trimInfoLabel.setBounds (t3); area.removeFromTop (gap);

    // FX
    fxToggle.setBounds (area.removeFromTop (rh)); area.removeFromTop (3);
    bool fx = fxToggle.getToggleState();
    if (fx) { sr (compThresholdSlider); sr (compRatioSlider); sr (delayTimeSlider); sr (delayMixSlider); sr (reverbSizeSlider); sr (reverbMixSlider); }

    // Preset row
    auto pr = area.removeFromTop (22);
    int bw = (pr.getWidth() - 6) / 3;
    savePresetButton.setBounds (pr.removeFromLeft (bw)); pr.removeFromLeft (3);
    loadPresetButton.setBounds (pr.removeFromLeft (bw)); pr.removeFromLeft (3);
    exportAudioButton.setBounds (pr); area.removeFromTop (3);
    auto fr = area.removeFromTop (18);
    setFolderButton.setBounds (fr.removeFromLeft (55)); fr.removeFromLeft (4);
    folderLabel.setBounds (fr); area.removeFromTop (gap);

    statusLabel.setBounds (area.removeFromTop (14));
    errorLabel.setBounds (area.removeFromTop (14));
    area.removeFromTop (gap);

    waveformBounds = area;
}

void MLXAudioGenEditor::updateUIState()
{
    bool mg = modelSelector.getSelectedId() != 2;
    temperatureSlider.setVisible (mg); topKSlider.setVisible (mg); guidanceSlider.setVisible (mg);
    stepsSlider.setVisible (! mg); cfgScaleSlider.setVisible (! mg); samplerSelector.setVisible (! mg);
    durationSlider.setVisible (! barsModeToggle.getToggleState());
    durationLabel.setVisible (! barsModeToggle.getToggleState());
    barsSlider.setVisible (barsModeToggle.getToggleState());
    barsLabel.setVisible (barsModeToggle.getToggleState());
    bpmSlider.setVisible (! dawBpmToggle.getToggleState());
    bpmLabel.setVisible (! dawBpmToggle.getToggleState());
    bool fx = fxToggle.getToggleState();
    compThresholdSlider.setVisible (fx); compRatioSlider.setVisible (fx);
    delayTimeSlider.setVisible (fx); delayMixSlider.setVisible (fx);
    reverbSizeSlider.setVisible (fx); reverbMixSlider.setVisible (fx);
    resized();
}

void MLXAudioGenEditor::timerCallback()
{
    displayProgress = proc.getProgress();
    bool gen = proc.isGenerating();
    generateButton.setEnabled (! gen);
    generateButton.setButtonText (gen ? juce::String ("Generating ") + juce::String ((int) (displayProgress * 100)) + "%" : "Generate");
    bool audio = proc.hasAudioLoaded();
    bool pending = proc.isPendingDecision();
    playButton.setEnabled (audio);
    stopButton.setEnabled (audio);
    exportAudioButton.setEnabled (audio && ! pending);
    keepButton.setVisible (pending);
    discardButton.setVisible (pending);
    keepButton.setEnabled (pending);
    discardButton.setEnabled (pending);
    dragButton.setEnabled (audio);
    playButton.setButtonText (proc.isPlaying() ? "Pause" : "Play");
    statusLabel.setText (proc.getStatusMessage(), juce::dontSendNotification);
    bpmDisplay.setText (juce::String ((int) proc.getEffectiveBpm()) + " BPM", juce::dontSendNotification);

    if (proc.hasAudioLoaded()) {
        float tb = proc.getTotalBeats();
        trimStartSlider.setRange (0, (double) tb, 0.25);
        trimEndSlider.setRange (0, (double) tb, 0.25);
        if (trimEndSlider.getValue() == 0.0) trimEndSlider.setValue (tb, juce::dontSendNotification);
        float dur = ((float) trimEndSlider.getValue() - (float) trimStartSlider.getValue()) * 60.0f / proc.getEffectiveBpm();
        trimInfoLabel.setText (juce::String ((float) trimStartSlider.getValue(), 2) + " → "
            + juce::String ((float) trimEndSlider.getValue(), 2) + " beats (" + juce::String (dur, 3) + "s)",
            juce::dontSendNotification);
    }
    trimButton.setEnabled (proc.hasAudioLoaded());

    auto err = proc.getLastError();
    errorLabel.setText (err, juce::dontSendNotification);
    errorLabel.setVisible (err.isNotEmpty());
    repaint();
}

void MLXAudioGenEditor::onGenerateClicked() { proc.triggerGeneration(); }
