#pragma once

#include <juce_core/juce_core.h>

/**
 * HTTP client for the mlx-audiogen-server.
 *
 * All methods are synchronous — call them from a background thread.
 * Supports both local (default localhost:8420) and remote servers
 * (with Cloudflare Access service token authentication).
 */
class HttpClient
{
public:
    HttpClient (const juce::String& baseUrl = "http://127.0.0.1:8420");

    /** Health check — returns true if server is reachable and responds 200 with "ok". */
    bool isServerAlive();

    /** List available models. Returns JSON array string, or empty on failure. */
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

    /** Get/set the server base URL (thread-safe). */
    juce::String getBaseUrl() const { juce::ScopedLock l (configLock); return baseUrl; }
    void setBaseUrl (const juce::String& url) { juce::ScopedLock l (configLock); baseUrl = url; }

    /** Set Cloudflare Access service token for remote server auth (thread-safe). */
    void setServiceToken (const juce::String& clientId, const juce::String& clientSecret);
    bool hasServiceToken() const { juce::ScopedLock l (configLock); return cfClientId.isNotEmpty() && cfClientSecret.isNotEmpty(); }

    // Timeout constants (milliseconds)
    static constexpr int healthTimeoutMs    = 3000;   // Health check
    static constexpr int defaultTimeoutMs   = 5000;   // Status polls, model list
    static constexpr int submitTimeoutMs    = 10000;  // Generation submit
    static constexpr int downloadTimeoutMs  = 60000;  // Audio download (large files)

    /** Safely parse JSON, returning juce::var() on failure. Logs errors to DBG. */
    static juce::var safeJsonParse (const juce::String& text, const juce::String& context);

private:
    mutable juce::CriticalSection configLock;
    juce::String baseUrl;
    juce::String cfClientId;
    juce::String cfClientSecret;

    /**
     * Perform a GET request, return response body.
     * @param statusCode  If non-null, receives the HTTP status code.
     */
    juce::String doGet (const juce::String& path, int timeoutMs = defaultTimeoutMs,
                        int* statusCode = nullptr);

    /**
     * Perform a POST request with JSON body, return response body.
     * @param statusCode  If non-null, receives the HTTP status code.
     */
    juce::String doPost (const juce::String& path,
                         const juce::String& jsonBody,
                         int timeoutMs = defaultTimeoutMs,
                         int* statusCode = nullptr);

    /** Build Cloudflare Access auth headers (must be called under configLock). */
    juce::String getAuthHeaders() const;

    /** Validate that a health response indicates a live server. */
    static bool isValidHealthResponse (const juce::String& body, int statusCode);
};
