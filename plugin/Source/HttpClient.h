#pragma once

#include <juce_core/juce_core.h>

/**
 * Lightweight HTTP client for talking to the mlx-audiogen-server.
 *
 * All methods are synchronous — call them from a background thread.
 * The server runs on localhost:8420 by default.
 */
class HttpClient
{
public:
    HttpClient (const juce::String& baseUrl = "http://127.0.0.1:8420");

    /** Health check — returns true if server is reachable. */
    bool isServerAlive();

    /** List available models. Returns JSON array string. */
    juce::String fetchModels();

    /**
     * Submit a generation request.
     * @param jsonBody  Full GenerateRequest JSON string.
     * @return Job ID string, or empty on failure.
     */
    juce::String submitGeneration (const juce::String& jsonBody);

    /**
     * Poll job status.
     * @param jobId  The job ID returned by submitGeneration.
     * @return JSON string of JobInfo, or empty on failure.
     */
    juce::String fetchStatus (const juce::String& jobId);

    /**
     * Download generated audio as WAV bytes.
     * @param jobId  The job ID of a completed generation.
     * @return WAV file data, or empty block on failure.
     */
    juce::MemoryBlock downloadAudio (const juce::String& jobId);

    /** Get/set the server base URL. */
    juce::String getBaseUrl() const { return baseUrl; }
    void setBaseUrl (const juce::String& url) { baseUrl = url; }

private:
    juce::String baseUrl;

    /** Perform a GET request, return response body. */
    juce::String doGet (const juce::String& path, int timeoutMs = 5000);

    /** Perform a POST request with JSON body, return response body. */
    juce::String doPost (const juce::String& path,
                         const juce::String& jsonBody,
                         int timeoutMs = 5000);
};
