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

bool HttpClient::isServerAlive()
{
    auto response = doGet ("/api/health");
    return response.isNotEmpty() && response.contains ("ok");
}

juce::String HttpClient::fetchModels()
{
    return doGet ("/api/models");
}

juce::String HttpClient::submitGeneration (const juce::String& jsonBody)
{
    auto response = doPost ("/api/generate", jsonBody);
    if (response.isEmpty())
        return {};

    // Parse {"id": "abc123", "status": "queued"}
    auto json = juce::JSON::parse (response);
    if (auto* obj = json.getDynamicObject())
        return obj->getProperty ("id").toString();

    return {};
}

juce::String HttpClient::fetchStatus (const juce::String& jobId)
{
    return doGet ("/api/status/" + jobId);
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

    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (10000)
                       .withExtraHeaders (authHeaders)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (nullptr)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);

    if (stream == nullptr)
        return {};

    juce::MemoryBlock block;
    stream->readIntoMemoryBlock (block);
    return block;
}

// ---------------------------------------------------------------------------
// Private HTTP helpers
// ---------------------------------------------------------------------------

juce::String HttpClient::doGet (const juce::String& path, int timeoutMs)
{
    juce::String currentBase, authHeaders;
    {
        juce::ScopedLock l (configLock);
        currentBase = baseUrl;
        authHeaders = getAuthHeaders();
    }

    juce::URL url (currentBase + path);

    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (timeoutMs)
                       .withExtraHeaders (authHeaders)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (nullptr)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);

    if (stream == nullptr)
        return {};

    return stream->readEntireStreamAsString();
}

juce::String HttpClient::doPost (const juce::String& path,
                                  const juce::String& jsonBody,
                                  int timeoutMs)
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

    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (timeoutMs)
                       .withExtraHeaders (extraHeaders)
                       .withResponseHeaders (nullptr)
                       .withStatusCode (nullptr)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);

    if (stream == nullptr)
        return {};

    return stream->readEntireStreamAsString();
}
