import { useCallback, useEffect, useState } from "react";
import { useStore } from "./store/useStore";
import { useServerHeartbeat } from "./hooks/useServerHeartbeat";
import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import PromptInput from "./components/PromptInput";
import ParameterPanel from "./components/ParameterPanel";
import GenerateButton from "./components/GenerateButton";
import EnhancePreview from "./components/EnhancePreview";
import HistoryPanel from "./components/HistoryPanel";
import SettingsPanel from "./components/SettingsPanel";
import LLMSettingsPanel from "./components/LLMSettingsPanel";
import ServerPanel from "./components/ServerPanel";
import TransportBar from "./components/TransportBar";
import TabBar from "./components/TabBar";
import SuggestPanel from "./components/SuggestPanel";
import QueuePanel from "./components/QueuePanel";
import TrainPanel from "./components/TrainPanel";
import { LibrarySidebar, LibraryTrackTable } from "./components/LibraryPanel";
import MetadataEditor from "./components/MetadataEditor";

const TABS = [
  { id: "generate", label: "Generate" },
  { id: "suggest", label: "Suggest" },
  { id: "train", label: "Train" },
  { id: "library", label: "Library" },
  { id: "settings", label: "Settings" },
];

export default function App() {
  const loadModels = useStore((s) => s.loadModels);
  const loadHistory = useStore((s) => s.loadHistory);
  const loadSettings = useStore((s) => s.loadSettings);
  const loadTags = useStore((s) => s.loadTags);
  const modelsLoading = useStore((s) => s.modelsLoading);
  const modelsError = useStore((s) => s.modelsError);
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const serverUrl = useStore((s) => s.serverUrl);
  const connected = useServerHeartbeat();

  // MetadataEditor modal state
  const selectedTrackIds = useStore((s) => s.selectedTrackIds);
  const libraryTracks = useStore((s) => s.libraryTracks);
  const activeSourceId = useStore((s) => s.activeSourceId);
  const [showMetadataEditor, setShowMetadataEditor] = useState(false);

  useEffect(() => {
    loadModels();
    loadHistory();
    loadSettings();
    loadTags();
  }, [loadModels, loadHistory, loadSettings, loadTags]);

  // Global keyboard shortcuts
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const isInput =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable;

      // Tab switching: 1/2/3/4/5 (only when not typing in input)
      if (!isInput && !e.metaKey && !e.ctrlKey) {
        if (e.key === "1") {
          e.preventDefault();
          setActiveTab("generate");
          return;
        }
        if (e.key === "2") {
          e.preventDefault();
          setActiveTab("suggest");
          return;
        }
        if (e.key === "3") {
          e.preventDefault();
          setActiveTab("train");
          return;
        }
        if (e.key === "4") {
          e.preventDefault();
          setActiveTab("library");
          return;
        }
        if (e.key === "5") {
          e.preventDefault();
          setActiveTab("settings");
          return;
        }
      }

      // Escape: dismiss enhance preview or metadata editor
      if (e.key === "Escape") {
        if (showMetadataEditor) {
          e.preventDefault();
          setShowMetadataEditor(false);
          return;
        }
        const { enhanceResult, clearEnhanceResult } = useStore.getState();
        if (enhanceResult) {
          e.preventDefault();
          clearEnhanceResult();
          return;
        }
      }

      // Space: play/pause most recent audio (only when not in input)
      if (e.key === " " && !isInput) {
        e.preventDefault();
        // Find the first audio element in history and toggle
        const audioEls = document.querySelectorAll("audio");
        if (audioEls.length > 0) {
          const audio = audioEls[0] as HTMLAudioElement;
          if (audio.paused) audio.play().catch(() => {});
          else audio.pause();
        }
      }
    },
    [setActiveTab, showMetadataEditor],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // "Train on These" opens the MetadataEditor with selected tracks
  const handleTrainOnThese = useCallback(() => {
    if (selectedTrackIds.size > 0 && activeSourceId) {
      setShowMetadataEditor(true);
    }
  }, [selectedTrackIds, activeSourceId]);

  // Selected tracks for MetadataEditor
  const selectedTracks = libraryTracks.filter((t) =>
    selectedTrackIds.has(t.track_id),
  );

  return (
    <div className="flex h-screen flex-col bg-surface-0">
      {/* Server disconnected banner */}
      {!connected && (
        <div className="bg-error/90 text-surface-0 px-4 py-2 text-center text-sm font-medium">
          {serverUrl ? (
            <>
              Remote server unreachable ({serverUrl}) —{" "}
              <button
                onClick={() => setActiveTab("settings")}
                className="underline hover:text-surface-0/80"
              >
                check Settings
              </button>
            </>
          ) : (
            <>
              Server disconnected — restart with{" "}
              <code className="bg-surface-0/20 px-1 rounded">
                mlx-audiogen-app
              </code>
            </>
          )}
        </div>
      )}
      <Header />

      <main className="flex flex-1 overflow-hidden">
        {/* Left panel: Controls */}
        <div className="flex w-80 shrink-0 flex-col border-r border-border bg-surface-1">
          <TabBar
            active={activeTab}
            tabs={TABS}
            onChange={(id) =>
              setActiveTab(id as typeof activeTab)
            }
          />

          {activeTab === "generate" && (
            <div className="flex flex-1 flex-col overflow-hidden">
              {/* Scrollable controls */}
              <div className="flex-1 space-y-5 overflow-y-auto p-5 pb-3">
                {modelsLoading && (
                  <div className="text-xs text-text-muted">Loading models...</div>
                )}
                {modelsError && (
                  <div className="rounded border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
                    Failed to connect to server: {modelsError}
                  </div>
                )}

                <ModelSelector />
                <PromptInput />
                <EnhancePreview />
                <ParameterPanel />
              </div>

              {/* Pinned at bottom: Generate + Queue */}
              <div className="shrink-0 space-y-3 border-t border-border p-5 pt-4">
                <GenerateButton />
                <QueuePanel />
              </div>
            </div>
          )}

          {activeTab === "suggest" && (
            <div className="flex-1 overflow-y-auto p-5">
              <SuggestPanel />
            </div>
          )}

          {activeTab === "train" && (
            <div className="flex-1 overflow-y-auto p-5">
              <TrainPanel />
            </div>
          )}

          {activeTab === "library" && <LibrarySidebar />}

          {activeTab === "settings" && (
            <div className="flex-1 space-y-5 overflow-y-auto p-5">
              <ServerPanel />
              <div className="border-t border-border pt-4">
                <SettingsPanel />
              </div>
              <div className="border-t border-border pt-4">
                <LLMSettingsPanel />
              </div>
            </div>
          )}
        </div>

        {/* Right panel: History or Library Track Table */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {activeTab === "library" ? (
            <LibraryTrackTable onTrainOnThese={handleTrainOnThese} />
          ) : (
            <div className="flex-1 overflow-y-auto p-5">
              <HistoryPanel />
            </div>
          )}
        </div>
      </main>

      {/* Transport bar: global playback controls */}
      <TransportBar connected={connected} />

      {/* MetadataEditor modal */}
      {showMetadataEditor && activeSourceId && selectedTracks.length > 0 && (
        <MetadataEditor
          tracks={selectedTracks}
          sourceId={activeSourceId}
          onClose={() => setShowMetadataEditor(false)}
        />
      )}
    </div>
  );
}
