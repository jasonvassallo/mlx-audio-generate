#include "ServerLauncher.h"
#include <sys/stat.h>

ServerLauncher::ServerLauncher()
{
    loadRemoteConfig();
}

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

bool ServerLauncher::isPathSafe (const juce::String& path)
{
    // Reject paths containing traversal components
    if (path.contains (".."))
        return false;
    // Reject empty paths
    if (path.isEmpty())
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// Remote config
// ---------------------------------------------------------------------------

void ServerLauncher::loadRemoteConfig()
{
    auto configFile = getConfigFile();
    if (! configFile.existsAsFile())
        return;

    auto text = configFile.loadFileAsString();
    auto parsed = juce::JSON::parse (text);

    if (parsed.isVoid())
    {
        DBG ("ServerLauncher: failed to parse config.json");
        return;
    }

    if (auto* obj = parsed.getDynamicObject())
    {
        remoteUrl = obj->getProperty ("remote_url").toString();
        cfClientId = obj->getProperty ("cf_client_id").toString();
        cfClientSecret = obj->getProperty ("cf_client_secret").toString();

        // Validate credentials: must be non-trivial length if present
        if (cfClientId.isNotEmpty() && cfClientId.length() < 8)
        {
            DBG ("ServerLauncher: cf_client_id too short, ignoring");
            cfClientId = {};
        }
        if (cfClientSecret.isNotEmpty() && cfClientSecret.length() < 8)
        {
            DBG ("ServerLauncher: cf_client_secret too short, ignoring");
            cfClientSecret = {};
        }
    }
}

bool ServerLauncher::isRemoteServerAlive()
{
    if (remoteUrl.isEmpty())
        return false;

    juce::URL url (remoteUrl + "/api/health");

    juce::String extraHeaders;
    if (cfClientId.isNotEmpty() && cfClientSecret.isNotEmpty())
        extraHeaders = "CF-Access-Client-Id: " + cfClientId
                     + "\r\nCF-Access-Client-Secret: " + cfClientSecret;

    int statusCode = 0;
    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (5000)
                       .withExtraHeaders (extraHeaders)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (&statusCode)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);
    if (stream == nullptr)
        return false;

    auto response = stream->readEntireStreamAsString();
    return statusCode == 200 && response.contains ("ok");
}

ServerLauncher::ConnectionMode ServerLauncher::recheckConnection()
{
    if (isServerAlive())
    {
        connectionMode = ConnectionMode::Local;
        status = "Server connected (local)";
    }
    else if (isRemoteServerAlive())
    {
        connectionMode = ConnectionMode::Remote;
        status = "Server connected (remote)";
    }
    else
    {
        connectionMode = ConnectionMode::Disconnected;
        status = "Disconnected";
    }
    return connectionMode;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

bool ServerLauncher::ensureServerRunning()
{
    // Already running locally?
    if (isServerAlive())
    {
        connectionMode = ConnectionMode::Local;
        status = "Server connected (local)";
        return true;
    }

    // Try to launch local server (if not already attempted)
    if (! serverLaunched)
    {
        status = "Starting server...";

        auto uvPath = findUvExecutable();
        if (uvPath.isNotEmpty())
        {
            auto projectPath = loadProjectPath();
            if (projectPath.isNotEmpty())
            {
                if (launchServer (uvPath, projectPath))
                {
                    serverLaunched = true;
                    status = "Server starting...";

                    // Wait up to 60 seconds for server to become alive
                    // (first load can take 30+ seconds for large models)
                    for (int i = 0; i < 120; ++i)
                    {
                        juce::Thread::sleep (500);

                        if (isServerAlive())
                        {
                            connectionMode = ConnectionMode::Local;
                            status = "Server connected (local)";
                            return true;
                        }

                        status = juce::String ("Starting server... (")
                               + juce::String (i / 2) + "s)";
                    }
                }
            }
        }
    }
    else
    {
        // Already tried launching — check if it came up
        if (isServerAlive())
        {
            connectionMode = ConnectionMode::Local;
            status = "Server connected (local)";
            return true;
        }
    }

    // Local failed — try remote server as fallback
    if (isRemoteServerAlive())
    {
        connectionMode = ConnectionMode::Remote;
        status = "Server connected (remote)";
        return true;
    }

    connectionMode = ConnectionMode::Disconnected;
    if (! serverLaunched)
        status = "No server available";
    else
        status = "Local server failed, remote unavailable";

    return false;
}

bool ServerLauncher::isServerAlive()
{
    juce::URL url ("http://127.0.0.1:8420/api/health");

    int statusCode = 0;
    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (2000)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (&statusCode)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);
    if (stream == nullptr)
        return false;

    auto response = stream->readEntireStreamAsString();
    return statusCode == 200 && response.contains ("ok");
}

// ---------------------------------------------------------------------------
// Config file management
// ---------------------------------------------------------------------------

juce::File ServerLauncher::getConfigFile()
{
    auto configDir = juce::File::getSpecialLocation (
                         juce::File::userHomeDirectory)
                         .getChildFile (".mlx-audiogen");
    configDir.createDirectory();
    return configDir.getChildFile ("config.json");
}

juce::String ServerLauncher::loadProjectPath()
{
    auto configFile = getConfigFile();

    // Try to read existing config
    if (configFile.existsAsFile())
    {
        auto text = configFile.loadFileAsString();
        auto parsed = juce::JSON::parse (text);

        if (! parsed.isVoid())
        {
            if (auto* obj = parsed.getDynamicObject())
            {
                auto path = obj->getProperty ("project_path").toString();
                if (isPathSafe (path) && juce::File (path).isDirectory())
                    return path;
            }
        }
        else
        {
            DBG ("ServerLauncher: failed to parse config.json in loadProjectPath");
        }
    }

    // Auto-detect: look for common locations relative to plugin binary
    juce::StringArray searchPaths;

    // The plugin is typically at ~/Library/Audio/Plug-Ins/...
    // The project is likely in ~/Documents/Code/mlx-audiogen
    auto home = juce::File::getSpecialLocation (juce::File::userHomeDirectory);
    searchPaths.add (home.getChildFile ("Documents/Code/mlx-audiogen").getFullPathName());
    searchPaths.add (home.getChildFile ("code/mlx-audiogen").getFullPathName());
    searchPaths.add (home.getChildFile ("projects/mlx-audiogen").getFullPathName());
    searchPaths.add (home.getChildFile ("dev/mlx-audiogen").getFullPathName());

    for (auto& path : searchPaths)
    {
        if (! isPathSafe (path))
            continue;

        auto dir = juce::File (path);
        // Verify it's the right project by checking for pyproject.toml
        if (dir.isDirectory() && dir.getChildFile ("pyproject.toml").existsAsFile())
        {
            // Save config for future loads (preserve remote fields)
            auto* obj = new juce::DynamicObject();
            obj->setProperty ("project_path", path);
            obj->setProperty ("uv_path", findUvExecutable());

            if (remoteUrl.isNotEmpty())
                obj->setProperty ("remote_url", remoteUrl);
            if (cfClientId.isNotEmpty())
                obj->setProperty ("cf_client_id", cfClientId);
            if (cfClientSecret.isNotEmpty())
                obj->setProperty ("cf_client_secret", cfClientSecret);

            juce::var json (obj);
            configFile.replaceWithText (juce::JSON::toString (json));
            ::chmod (configFile.getFullPathName().toRawUTF8(), 0600);

            return path;
        }
    }

    // Last resort: write a template config for the user to fill in
    if (! configFile.existsAsFile())
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("project_path", "/path/to/mlx-audiogen");
        obj->setProperty ("uv_path", "uv");
        obj->setProperty ("remote_url", "");
        obj->setProperty ("cf_client_id", "");
        obj->setProperty ("cf_client_secret", "");

        juce::var json (obj);
        configFile.replaceWithText (juce::JSON::toString (json));
        ::chmod (configFile.getFullPathName().toRawUTF8(), 0600);
    }

    return {};
}

// ---------------------------------------------------------------------------
// uv executable discovery
// ---------------------------------------------------------------------------

juce::String ServerLauncher::findUvExecutable()
{
    // Check common locations
    juce::StringArray candidates = {
        "/opt/homebrew/bin/uv",                     // Homebrew (Apple Silicon)
        "/usr/local/bin/uv",                        // Homebrew (Intel) or manual
        juce::File::getSpecialLocation (juce::File::userHomeDirectory)
            .getChildFile (".cargo/bin/uv").getFullPathName(), // Cargo install
        juce::File::getSpecialLocation (juce::File::userHomeDirectory)
            .getChildFile (".local/bin/uv").getFullPathName(), // astral.sh installer
        "uv"                                        // On PATH
    };

    for (auto& candidate : candidates)
    {
        if (juce::File (candidate).existsAsFile())
            return candidate;
    }

    // Try which
    juce::ChildProcess which;
    if (which.start ("which uv"))
    {
        which.waitForProcessToFinish (2000);
        auto path = which.readAllProcessOutput().trim();
        if (path.isNotEmpty() && juce::File (path).existsAsFile())
            return path;
    }

    return {};
}

// ---------------------------------------------------------------------------
// Server process launch
// ---------------------------------------------------------------------------

bool ServerLauncher::launchServer (const juce::String& uvPath,
                                    const juce::String& projectPath)
{
    // Validate paths before shell interpolation
    if (! isPathSafe (projectPath) || ! isPathSafe (uvPath))
    {
        DBG ("ServerLauncher: unsafe path detected, refusing to launch");
        return false;
    }

    // Launch: uv run mlx-audiogen-app
    // Detached so it survives the plugin being unloaded

    // Create a launcher script that sets up the environment properly
    auto launchScript = getConfigFile().getSiblingFile ("launch-server.sh");

    // Use single-quoted paths to prevent shell injection,
    // with proper escaping of any single quotes in the path itself
    auto safeProject = projectPath.replace ("'", "'\\''");
    auto safeUv = uvPath.replace ("'", "'\\''");

    juce::String script;
    script += "#!/bin/bash\n";
    script += "cd '" + safeProject + "'\n";
    script += "'" + safeUv + "' run mlx-audiogen-app > ~/.mlx-audiogen/server.log 2>&1 &\n";
    script += "echo $! > ~/.mlx-audiogen/server.pid\n";

    launchScript.replaceWithText (script);
    launchScript.setExecutePermission (true);

    juce::ChildProcess launcher;
    return launcher.start (launchScript.getFullPathName());
}
