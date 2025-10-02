#include "tg.h"

void Tg::sendMessage(const std::string &message)
{
    if (botToken.empty() || chatId.empty())
    {
        return;
    }

    try
    {
        // Escape special characters for JSON
        std::string escapedMessage = message;
        size_t pos = 0;
        while ((pos = escapedMessage.find("\"", pos)) != std::string::npos)
        {
            escapedMessage.replace(pos, 1, "\\\"");
            pos += 2;
        }
        while ((pos = escapedMessage.find("\n", pos)) != std::string::npos)
        {
            escapedMessage.replace(pos, 1, "\\n");
            pos += 2;
        }

        std::string url = "https://api.telegram.org/bot" + botToken + "/sendMessage";
        std::string data = "{\"chat_id\":\"" + chatId + "\",\"text\":\"" + escapedMessage + "\"}";

        // Build curl command
        std::string curlCmd = "curl -s -X POST -H \"Content-Type: application/json\" -d '" + data + "' " + url + " > /dev/null 2>&1 &";

        // Execute asynchronously in background
        int result = system(curlCmd.c_str());
        (void)result; // Suppress unused variable warning
    }
    catch (...)
    {
        // Silently ignore any errors during telegram notification
        // This ensures the main program continues even if telegram is unavailable
    }
}

std::string Tg::headerSuffix()
{
    if (instanceName.empty())
    {
        return "";
    }
    return " [" + instanceName + "]";
}

std::string Tg::keyFound(const std::string &body)
{
    return "🔑 KEY FOUND" + headerSuffix() + "\\n\\n" + body;
}

std::string Tg::messageEnd()
{
    return "⏹️ SEARCH COMPLETED" + headerSuffix() + "\\n\\n";
}

std::string Tg::messageError(const std::string &body)
{
    return "❗ ERROR OCCURRED" + headerSuffix() + "\\n\\n" + body;
}
