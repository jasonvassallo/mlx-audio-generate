#include "HttpClient.h"

HttpClient::HttpClient (const juce::String& url)
    : baseUrl (url)
{
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
    juce::URL url (baseUrl + "/api/audio/" + jobId);

    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (5000)
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
    juce::URL url (baseUrl + path);

    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (timeoutMs)
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
    juce::URL url (baseUrl + path);

    // POST with JSON body
    url = url.withPOSTData (jsonBody);

    auto options = juce::URL::InputStreamOptions (juce::URL::ParameterHandling::inAddress)
                       .withConnectionTimeoutMs (timeoutMs)
                       .withExtraHeaders ("Content-Type: application/json")
                       .withResponseHeaders (nullptr)
                       .withStatusCode (nullptr)
                       .withNumRedirectsToFollow (0);

    auto stream = url.createInputStream (options);

    if (stream == nullptr)
        return {};

    return stream->readEntireStreamAsString();
}
