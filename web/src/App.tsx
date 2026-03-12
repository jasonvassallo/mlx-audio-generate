import { useEffect } from "react";
import { useStore } from "./store/useStore";
import { useServerHeartbeat } from "./hooks/useServerHeartbeat";
import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import PromptInput from "./components/PromptInput";
import ParameterPanel from "./components/ParameterPanel";
import GenerateButton from "./components/GenerateButton";
import EnhancePreview from "./components/EnhancePreview";
import HistoryPanel from "./components/HistoryPanel";
import AudioDeviceSelector from "./components/AudioDeviceSelector";
import SettingsPanel from "./components/SettingsPanel";
import LLMSettingsPanel from "./components/LLMSettingsPanel";
import TabBar from "./components/TabBar";
import SuggestPanel from "./components/SuggestPanel";

const TABS = [
  { id: "generate", label: "Generate" },
  { id: "suggest", label: "Suggest" },
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
  const connected = useServerHeartbeat();

  useEffect(() => {
    loadModels();
    loadHistory();
    loadSettings();
    loadTags();
  }, [loadModels, loadHistory, loadSettings, loadTags]);

  return (
    <div className="flex h-screen flex-col bg-surface-0">
      {/* Server disconnected banner */}
      {!connected && (
        <div className="bg-error/90 text-surface-0 px-4 py-2 text-center text-sm font-medium">
          Server disconnected — restart with{" "}
          <code className="bg-surface-0/20 px-1 rounded">mlx-audiogen-app</code>
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
              setActiveTab(id as "generate" | "suggest" | "settings")
            }
          />

          <div className="flex flex-1 flex-col gap-5 overflow-y-auto p-5">
            {activeTab === "generate" && (
              <>
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
                <GenerateButton />
              </>
            )}

            {activeTab === "suggest" && <SuggestPanel />}

            {activeTab === "settings" && <LLMSettingsPanel />}
          </div>

          {/* Bottom section: always visible */}
          <div className="space-y-4 border-t border-border p-5">
            <SettingsPanel />
            <AudioDeviceSelector />
          </div>
        </div>

        {/* Right panel: History / Output */}
        <div className="flex flex-1 flex-col overflow-y-auto p-5">
          <HistoryPanel />
        </div>
      </main>
    </div>
  );
}
