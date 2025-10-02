#ifndef _TG_H
#define _TG_H

#include <string>

class Tg
{
private:
    std::string botToken;
    std::string chatId;
    std::string instanceName;

    std::string headerSuffix();

public:
    Tg(const std::string &botToken, const std::string &chatId, const std::string &instanceName) : botToken(botToken), chatId(chatId), instanceName(instanceName) {};
    void sendMessage(const std::string &message);

    std::string keyFound(const std::string &body);
    std::string messageEnd();
    std::string messageError(const std::string &body);
};

#endif