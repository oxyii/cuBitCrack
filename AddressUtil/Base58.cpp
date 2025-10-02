#include <map>
#include <vector>
#include <string>
#include "CryptoUtil.h"

#include "AddressUtil.h"


static const std::string BASE58_STRING = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

struct Base58Map {
	static std::map<char, int> createBase58Map()
	{
		std::map<char, int> m;
		for(int i = 0; i < 58; i++) {
			m[BASE58_STRING[i]] = i;
		}

		return m;
	}

	static std::map<char, int> myMap;
};

std::map<char, int> Base58Map::myMap = Base58Map::createBase58Map();



/**
 * Converts a base58 string to uint256
 */
secp256k1::uint256 Base58::toBigInt(const std::string &s)
{
	secp256k1::uint256 value;

	for(unsigned int i = 0; i < s.length(); i++) {
		value = value.mul(58);

		int c = Base58Map::myMap[s[i]];
		value = value.add(c);
	}

	return value;
}

void Base58::toHash160(const std::string &s, unsigned int hash[5])
{
	secp256k1::uint256 value = toBigInt(s);
	unsigned int words[6];

	value.exportWords(words, 6, secp256k1::uint256::BigEndian);

	// Extract words, ignore checksum
	for(int i = 0; i < 5; i++) {
		hash[i] = words[i];
	}
}

bool Base58::isBase58(std::string s)
{
	for(unsigned int i = 0; i < s.length(); i++) {
		if(BASE58_STRING.find(s[i]) < 0) {
			return false;
		}
	}

	return true;
}

std::string Base58::toBase58(const secp256k1::uint256 &x)
{
	std::string s;

	secp256k1::uint256 value = x;

	while(!value.isZero()) {
		secp256k1::uint256 digit = value.mod(58);
		int digitInt = digit.toInt32();

		s = BASE58_STRING[digitInt] + s;

		value = value.div(58);
	}

	return s;
}

std::string Base58::toBase58(const unsigned char *data, size_t length)
{
	// Подсчет ведущих нулей
	size_t zeros = 0;
	while (zeros < length && data[zeros] == 0)
	{
		zeros++;
	}

	// Выделяем место для результата
	size_t size = length * 138 / 100 + 1; // Максимальный размер в base58
	std::vector<unsigned char> b58(size, 0);

	// Процессируем байты
	for (size_t i = zeros; i < length; i++)
	{
		int carry = data[i];
		// Проходим через все цифры base58 от младших к старшим
		for (size_t j = 0; j < size; j++)
		{
			carry += 256 * b58[size - 1 - j];
			b58[size - 1 - j] = carry % 58;
			carry /= 58;
		}
	}

	// Пропускаем ведущие нули в результате base58
	size_t start = size;
	for (size_t i = 0; i < size; i++)
	{
		if (b58[i] != 0)
		{
			start = i;
			break;
		}
	}

	// Строим итоговую строку
	std::string result;
	result.reserve(zeros + (size - start));

	// Добавляем '1' для каждого ведущего нуля в исходных данных
	result.assign(zeros, '1');

	// Добавляем остальные символы base58
	for (size_t i = start; i < size; i++)
	{
		result += BASE58_STRING[b58[i]];
	}

	return result;
}

void Base58::getMinMaxFromPrefix(const std::string &prefix, secp256k1::uint256 &minValueOut, secp256k1::uint256 &maxValueOut)
{
	secp256k1::uint256 minValue = toBigInt(prefix);
	secp256k1::uint256 maxValue = minValue;
	int exponent = 1;

	// 2^192
	unsigned int expWords[] = { 0, 0, 0, 0, 0, 0, 1, 0 };

	secp256k1::uint256 exp(expWords);

	// Find the smallest 192-bit number that starts with the prefix. That is, the prefix multiplied
	// by some power of 58
	secp256k1::uint256 nextValue = minValue.mul(58);

	while(nextValue.cmp(exp) < 0) {
		exponent++;
		minValue = nextValue;
		nextValue = nextValue.mul(58);
	}

	secp256k1::uint256 diff = secp256k1::uint256(58).pow(exponent - 1).sub(1);

	maxValue = minValue.add(diff);

	if(maxValue.cmp(exp) > 0) {
		maxValue = exp.sub(1);
	}

	minValueOut = minValue;
	maxValueOut = maxValue;
}