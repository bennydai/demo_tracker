#ifndef data_utils_H
#define data_utils_H

#include <functional>
#include <utility>

struct PairHash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const
    {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ (hash2 << 1); // Combine hashes
    }
};

#endif // data_utils_H
