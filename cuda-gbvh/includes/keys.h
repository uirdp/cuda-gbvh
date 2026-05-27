#pragma once
#include <unordered_set>
#include <cstdint>

struct DirtyKey {
    uint64_t code;
    uint8_t bits;

    bool operator==(const DirtyKey& other) const {
        return code == other.code && bits == other.bits;
    }
};

struct DirtyKeyHash{
    size_t operator()(const DirtyKey& k) const {
        return std::hash<uint64_t>()(k.code) ^ (std::hash<uint8_t>()(k.bits) << 1);
    }
};

using DirtyKeySet = std::unordered_set<DirtyKey, DirtyKeyHash>;

struct NodeKey {
    uint64_t code;
    uint8_t bits;

    bool operator==(const NodeKey& other) const {
        return code == other.code && bits == other.bits;
    }
};

struct NodeKeyHash {
    std::size_t operator()(const NodeKey& k) const {
        return std::hash<uint64_t>()(k.code) ^ (std::hash<uint8_t>()(k.bits) << 1);
    }
};