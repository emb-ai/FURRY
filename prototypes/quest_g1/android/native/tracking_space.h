#pragma once
#include <cstdint>
#include <set>

// Caller serializes access. OpenXR announces changes before their effective time.
class TrackingSpaceChanges {
    std::set<int64_t> pending;
public:
    void Schedule(int64_t time){pending.insert(time);}
    bool Advance(int64_t time){
        auto end=pending.upper_bound(time);
        bool changed=end!=pending.begin();
        pending.erase(pending.begin(),end);
        return changed;
    }
};
