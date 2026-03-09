#include "ServerLauncher.h"

ServerLauncher::ServerLauncher()
{
}

bool ServerLauncher::ensureServerRunning()
{
    // Already running?
    if (isServerAlive())
    {
        status = "Server connected";
        return true;
    }

    // Already tried to launch this session
    if (serverLaunched)
    {
        status = "Waiting for server...";
        return isServerAlive();
    }

    status = "Starting server...";

    // Find uv executable
    auto uvPath = findUvExecutable();
    if (uvPath.isEmpty())
    {
        status = "Error: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh";
        return false;
    }

    // Find project directory
    auto projectPath = loadProjectPath();
    if (projectPath.isEmpty())
    {
        status = "Error: Project not configured. See ~/.mlx-audiogen/config.json";
        return false;
    }

    // Launch server
    if (! launchServer (uvPath, projectPath))
    {
        status = "Error: Failed to start server";
        return false;
    }

    serverLaunched = true;
    status = "Server starting...";

    // Wait up to 60 seconds for server to become alive
    // (first load can take 30+ seconds for large models)
    for (int i = 0; i < 120; ++i)
    {
        juce::Thread::sleep (500);

        if (isServerAlive())
        {
            status = "Server connected";
            return true;
        }

        status = juce::String ("Starting server... (") + juce::String (i / 2) + "s)";
    }

    status = "Error: Server failed to start within 60 seconds";
    return false;
}

bool ServerLauncher::isServerAlive()
{
    juce::URL url ("http://127.0.0.1:8420/api/health");

    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (2000)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (nullptr)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);
    if (stream == nullptr)
        return false;

    auto response = stream->readEntireStreamAsString();
    return response.contains ("ok");
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

        if (auto* obj = parsed.getDynamicObject())
        {
            auto path = obj->getProperty ("project_path").toString();
            if (juce::File (path).isDirectory())
                return path;
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
        auto dir = juce::File (path);
        // Verify it's the right project by checking for pyproject.toml
        if (dir.isDirectory() && dir.getChildFile ("pyproject.toml").existsAsFile())
        {
            // Save config for future loads
            auto* obj = new juce::DynamicObject();
            obj->setProperty ("project_path", path);
            obj->setProperty ("uv_path", findUvExecutable());

            juce::var json (obj);
            configFile.replaceWithText (juce::JSON::toString (json));

            return path;
        }
    }

    // Last resort: write a template config for the user to fill in
    if (! configFile.existsAsFile())
    {
        auto* obj = new juce::DynamicObject();
        obj->setProperty ("project_path", "/path/to/mlx-audiogen");
        obj->setProperty ("uv_path", "uv");

        juce::var json (obj);
        configFile.replaceWithText (juce::JSON::toString (json));
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
    // Launch: uv run mlx-audiogen-app
    // Detached so it survives the plugin being unloaded
    auto projectDir = juce::File (projectPath);

    // Create a launcher script that sets up the environment properly
    auto launchScript = getConfigFile().getSiblingFile ("launch-server.sh");

    juce::String script;
    script += "#!/bin/bash\n";
    script += "cd \"" + projectPath + "\"\n";
    script += "\"" + uvPath + "\" run mlx-audiogen-app > ~/.mlx-audiogen/server.log 2>&1 &\n";
    script += "echo $! > ~/.mlx-audiogen/server.pid\n";

    launchScript.replaceWithText (script);
    launchScript.setExecutePermission (true);

    juce::ChildProcess launcher;
    return launcher.start (launchScript.getFullPathName());
}
