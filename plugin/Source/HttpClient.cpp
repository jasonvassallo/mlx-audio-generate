#include "HttpClient.h"

HttpClient::HttpClient (const juce::String& url)
    : baseUrl (url)
{
}

void HttpClient::setServiceToken (const juce::String& clientId, const juce::String& clientSecret)
{
    juce::ScopedLock l (configLock);
    cfClientId = clientId;
    cfClientSecret = clientSecret;
}

juce::String HttpClient::getAuthHeaders() const
{
    // Caller must hold configLock.
    if (cfClientId.isNotEmpty() && cfClientSecret.isNotEmpty())
        return "CF-Access-Client-Id: " + cfClientId
             + "\r\nCF-Access-Client-Secret: " + cfClientSecret;
    return {};
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

bool HttpClient::isValidHealthResponse (const juce::String& body, int statusCode)
{
    return statusCode == 200 && body.isNotEmpty() && body.contains ("ok");
}

juce::var HttpClient::safeJsonParse (const juce::String& text, const juce::String& context)
{
    if (text.isEmpty())
        return {};

    auto result = juce::JSON::parse (text);
    if (result.isVoid())
    {
        DBG ("JSON parse error (" + context + "): " + text.substring (0, 200));
        return {};
    }
    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool HttpClient::isServerAlive()
{
    int statusCode = 0;
    auto response = doGet ("/api/health", healthTimeoutMs, &statusCode);
    return isValidHealthResponse (response, statusCode);
}

juce::String HttpClient::fetchModels()
{
    int statusCode = 0;
    auto response = doGet ("/api/models", defaultTimeoutMs, &statusCode);
    if (statusCode != 200)
        return {};
    return response;
}

juce::String HttpClient::submitGeneration (const juce::String& jsonBody)
{
    int statusCode = 0;
    auto response = doPost ("/api/generate", jsonBody, submitTimeoutMs, &statusCode);
    if (response.isEmpty() || statusCode < 200 || statusCode >= 300)
        return {};

    auto json = safeJsonParse (response, "submitGeneration");
    if (auto* obj = json.getDynamicObject())
        return obj->getProperty ("id").toString();

    return {};
}

juce::String HttpClient::fetchStatus (const juce::String& jobId)
{
    int statusCode = 0;
    auto response = doGet ("/api/status/" + jobId, defaultTimeoutMs, &statusCode);
    if (statusCode != 200)
        return {};
    return response;
}

juce::MemoryBlock HttpClient::downloadAudio (const juce::String& jobId)
{
    juce::String currentBase, authHeaders;
    {
        juce::ScopedLock l (configLock);
        currentBase = baseUrl;
        authHeaders = getAuthHeaders();
    }

    juce::URL url (currentBase + "/api/audio/" + jobId);

    int statusCode = 0;
    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (downloadTimeoutMs)
                       .withExtraHeaders (authHeaders)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (&statusCode)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);

    if (stream == nullptr || statusCode != 200)
    {
        DBG ("Audio download failed: HTTP " + juce::String (statusCode));
        return {};
    }

    juce::MemoryBlock block;
    stream->readIntoMemoryBlock (block);
    return block;
}

// ---------------------------------------------------------------------------
// Private HTTP helpers
// ---------------------------------------------------------------------------

juce::String HttpClient::doGet (const juce::String& path, int timeoutMs, int* statusCode)
{
    juce::String currentBase, authHeaders;
    {
        juce::ScopedLock l (configLock);
        currentBase = baseUrl;
        authHeaders = getAuthHeaders();
    }

    juce::URL url (currentBase + path);

    int localStatus = 0;
    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (timeoutMs)
                       .withExtraHeaders (authHeaders)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (&localStatus)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);

    if (statusCode != nullptr)
        *statusCode = localStatus;

    if (stream == nullptr)
        return {};

    return stream->readEntireStreamAsString();
}

juce::String HttpClient::doPost (const juce::String& path,
                                  const juce::String& jsonBody,
                                  int timeoutMs,
                                  int* statusCode)
{
    juce::String currentBase, cfHeaders;
    {
        juce::ScopedLock l (configLock);
        currentBase = baseUrl;
        cfHeaders = getAuthHeaders();
    }

    juce::URL url (currentBase + path);

    // POST with JSON body
    url = url.withPOSTData (jsonBody);

    juce::String extraHeaders = "Content-Type: application/json";
    if (cfHeaders.isNotEmpty())
        extraHeaders += "\r\n" + cfHeaders;

    int localStatus = 0;
    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (timeoutMs)
                       .withExtraHeaders (extraHeaders)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (&localStatus)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);

    if (statusCode != nullptr)
        *statusCode = localStatus;

    if (stream == nullptr)
        return {};

    return stream->readEntireStreamAsString();
}
