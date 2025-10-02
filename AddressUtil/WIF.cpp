#include "AddressUtil.h"
#include "CryptoUtil.h"
#include <cstdio>
#include <cstring>

// Функция для создания WIF из приватного ключа
std::string WIF::privateKeyToWIF(const secp256k1::uint256 &privateKey, bool compressed)
{
    if (compressed)
    {
        return privateKeyToWIFCompressed(privateKey);
    }
    else
    {
        return privateKeyToWIFUncompressed(privateKey);
    }
}

// Функция для создания неcжатого WIF
std::string WIF::privateKeyToWIFUncompressed(const secp256k1::uint256 &privateKey)
{
    // Структура: [0x80] + [32-byte private key] + [4-byte checksum]
    // Общая длина: 37 байт
    unsigned char data[37];

    // 1. Добавляем version byte (0x80 для mainnet)
    data[0] = 0x80;

    // 2. Добавляем приватный ключ (32 байта)
    // Получаем данные из uint256 в big-endian формате
    unsigned int words[8];
    privateKey.exportWords(words, 8, secp256k1::uint256::BigEndian);

    // Конвертируем 32-bit words в байты
    for (int i = 0; i < 8; i++)
    {
        data[1 + i * 4] = (words[i] >> 24) & 0xFF;
        data[1 + i * 4 + 1] = (words[i] >> 16) & 0xFF;
        data[1 + i * 4 + 2] = (words[i] >> 8) & 0xFF;
        data[1 + i * 4 + 3] = words[i] & 0xFF;
    }

    // 3. Вычисляем checksum (двойной SHA256 первых 33 байт)
    unsigned int hash1[8], hash2[8];
    unsigned int msg[16] = {0};

    // Подготавливаем данные для SHA256 (33 байта = 264 бита)
    // Копируем 33 байта в первые 8.25 слов
    for (int i = 0; i < 8; i++)
    {
        msg[i] = (data[i * 4] << 24) |
                 (data[i * 4 + 1] << 16) |
                 (data[i * 4 + 2] << 8) |
                 data[i * 4 + 3];
    }
    // Последний байт в 9-м слове + паддинг
    msg[8] = (data[32] << 24) | 0x00800000; // data[32] + padding 0x80

    // Устанавливаем длину данных в битах (33 * 8 = 264)
    msg[15] = 33 * 8;

    // Первый SHA256
    crypto::sha256Init(hash1);
    crypto::sha256(msg, hash1);

    // Подготавливаем для второго SHA256 (32 байта = 256 бит)
    memset(msg, 0, 16 * sizeof(unsigned int));
    for (int i = 0; i < 8; i++)
    {
        msg[i] = hash1[i];
    }
    msg[8] = 0x80000000; // паддинг
    msg[15] = 256;       // длина в битах

    // Второй SHA256
    crypto::sha256Init(hash2);
    crypto::sha256(msg, hash2);

    // 4. Добавляем первые 4 байта checksum
    data[33] = (hash2[0] >> 24) & 0xFF;
    data[34] = (hash2[0] >> 16) & 0xFF;
    data[35] = (hash2[0] >> 8) & 0xFF;
    data[36] = hash2[0] & 0xFF;

    // 5. Кодируем в Base58 напрямую из массива байтов
    return Base58::toBase58(data, 37);
}

// Функция для создания сжатого WIF
std::string WIF::privateKeyToWIFCompressed(const secp256k1::uint256 &privateKey)
{
    // Структура: [0x80] + [32-byte private key] + [0x01] + [4-byte checksum]
    // Общая длина: 38 байт
    unsigned char data[38];

    // 1. Добавляем version byte (0x80 для mainnet)
    data[0] = 0x80;

    // 2. Добавляем приватный ключ (32 байта)
    // Получаем данные из uint256 в big-endian формате
    unsigned int words[8];
    privateKey.exportWords(words, 8, secp256k1::uint256::BigEndian);

    // Конвертируем 32-bit words в байты
    for (int i = 0; i < 8; i++)
    {
        data[1 + i * 4] = (words[i] >> 24) & 0xFF;
        data[1 + i * 4 + 1] = (words[i] >> 16) & 0xFF;
        data[1 + i * 4 + 2] = (words[i] >> 8) & 0xFF;
        data[1 + i * 4 + 3] = words[i] & 0xFF;
    }

    // 3. Добавляем compression flag
    data[33] = 0x01;

    // 4. Вычисляем checksum (двойной SHA256 первых 34 байт)
    unsigned int hash1[8], hash2[8];
    unsigned int msg[16] = {0};

    // Подготавливаем данные для SHA256 (34 байта = 272 бита)
    // Копируем 34 байта в первые 8.5 слов
    for (int i = 0; i < 8; i++)
    {
        msg[i] = (data[i * 4] << 24) |
                 (data[i * 4 + 1] << 16) |
                 (data[i * 4 + 2] << 8) |
                 data[i * 4 + 3];
    }
    // Последние 2 байта в 9-м слове + паддинг
    msg[8] = (data[32] << 24) | (data[33] << 16) | 0x00008000; // data[32], data[33] + padding 0x80

    // Устанавливаем длину данных в битах (34 * 8 = 272)
    msg[15] = 34 * 8;

    // Первый SHA256
    crypto::sha256Init(hash1);
    crypto::sha256(msg, hash1);

    // Подготавливаем для второго SHA256 (32 байта = 256 бит)
    memset(msg, 0, 16 * sizeof(unsigned int));
    for (int i = 0; i < 8; i++)
    {
        msg[i] = hash1[i];
    }
    msg[8] = 0x80000000; // паддинг
    msg[15] = 256;       // длина в битах

    // Второй SHA256
    crypto::sha256Init(hash2);
    crypto::sha256(msg, hash2);

    // 5. Добавляем первые 4 байта checksum
    data[34] = (hash2[0] >> 24) & 0xFF;
    data[35] = (hash2[0] >> 16) & 0xFF;
    data[36] = (hash2[0] >> 8) & 0xFF;
    data[37] = hash2[0] & 0xFF;

    // 6. Кодируем в Base58 напрямую из массива байтов
    return Base58::toBase58(data, 38);
}