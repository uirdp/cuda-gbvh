#pragma once
#include <unordered_set>
#include <cstdint>
enum DirtyKeyType : uint8_t
{
    DIRTY_KEY_GRID = 0,
    DIRTY_KEY_LEAF = 1
};

struct DirtyKey
{
    uint64_t code;
    uint64_t leaf_code;
    uint8_t bits;
    uint8_t type;

    bool operator==(const DirtyKey &rhs) const
    {
        return code == rhs.code &&
               leaf_code == rhs.leaf_code &&
               bits == rhs.bits &&
               type == rhs.type;
    }
};

using GPU_DirtyKey = DirtyKey;

struct DirtyKeyHash
{
    size_t operator()(const DirtyKey &k) const
    {
        size_t h0 = std::hash<uint64_t>()(k.code);
        size_t h1 = std::hash<int>()((int)k.bits);
        size_t h2 = std::hash<uint64_t>()(k.leaf_code);
        size_t h3 = std::hash<int>()((int)k.type);

        return h0 ^ (h1 << 1) ^ (h2 << 2) ^ (h3 << 3);
    }
};

using DirtyKeySet = std::unordered_set<DirtyKey, DirtyKeyHash>;

struct NodeKey
{
    uint64_t code;
    uint8_t bits;

    bool operator==(const NodeKey &other) const
    {
        return code == other.code && bits == other.bits;
    }
};

struct NodeKeyHash
{
    std::size_t operator()(const NodeKey &k) const
    {
        return std::hash<uint64_t>()(k.code) ^ (std::hash<uint8_t>()(k.bits) << 1);
    }
};