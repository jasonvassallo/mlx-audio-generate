#pragma once

#include <juce_core/juce_core.h>

/**
 * Automatically launches the mlx-audiogen server if it isn't running.
 *
 * On first use, writes a config file to ~/.mlx-audiogen/config.json
 * with the project path and uv executable location. The plugin reads
 * this config on load and spawns "uv run mlx-audiogen-app" as a
 * detached background process.
 *
 * The server auto-discovers all converted models in ./converted/ and
 * starts on port 8420. No manual terminal commands needed.
 */
class ServerLauncher
{
public:
    ServerLauncher();

    /**
     * Ensure the server is running. If not, launch it automatically.
     * @return true if server is alive (or was just launched), false on failure.
     */
    bool ensureServerRunning();

    /** Get human-readable status for UI display. */
    juce::String getStatus() const { return status; }

    /** Check if server is currently reachable. */
    bool isServerAlive();

private:
    /** Find the config file (~/.mlx-audiogen/config.json). */
    juce::File getConfigFile();

    /** Load or create the config file. Returns project directory path. */
    juce::String loadProjectPath();

    /** Find the uv executable on the system. */
    juce::String findUvExecutable();

    /** Launch the server process. */
    bool launchServer (const juce::String& uvPath,
                       const juce::String& projectPath);

    juce::String status;
    bool serverLaunched { false };
};
