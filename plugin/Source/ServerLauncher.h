#pragma once

#include <juce_core/juce_core.h>

/**
 * Automatically launches the mlx-audiogen server if it isn't running.
 *
 * Supports both local (via uv) and remote (via Cloudflare Tunnel)
 * server connections. Local is always preferred; remote is used as
 * fallback when local launch fails.
 *
 * Config file: ~/.mlx-audiogen/config.json
 */
class ServerLauncher
{
public:
    ServerLauncher();

    /** Connection mode determined by ensureServerRunning / recheckConnection. */
    enum class ConnectionMode { Local, Remote, Disconnected };

    /**
     * Ensure a server is available. Tries local first, falls back to remote.
     * @return true if any server (local or remote) is reachable.
     */
    bool ensureServerRunning();

    /** Get human-readable status for UI display. */
    juce::String getStatus() const { return status; }

    /** Check if local server is currently reachable. */
    bool isServerAlive();

    /** Current connection mode. */
    ConnectionMode getConnectionMode() const { return connectionMode; }

    /** Re-check server availability and update connection mode. */
    ConnectionMode recheckConnection();

    /** Remote server config from ~/.mlx-audiogen/config.json. */
    juce::String getRemoteUrl() const { return remoteUrl; }
    juce::String getCfClientId() const { return cfClientId; }
    juce::String getCfClientSecret() const { return cfClientSecret; }

    /** Validate a path is safe (no traversal, non-empty). */
    static bool isPathSafe (const juce::String& path);

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

    /** Load remote_url, cf_client_id, cf_client_secret from config.json. */
    void loadRemoteConfig();

    /** Check if the remote server is reachable (with auth headers). */
    bool isRemoteServerAlive();

    juce::String status;
    bool serverLaunched { false };
    ConnectionMode connectionMode { ConnectionMode::Disconnected };

    // Remote server config
    juce::String remoteUrl;
    juce::String cfClientId;
    juce::String cfClientSecret;
};
