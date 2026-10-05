#include "tracking_space.h"
#include <cassert>

int main(){
    TrackingSpaceChanges changes;
    assert(!changes.Advance(100));
    changes.Schedule(200);
    changes.Schedule(200);
    changes.Schedule(300);
    assert(!changes.Advance(199));
    assert(changes.Advance(200));
    assert(!changes.Advance(200));
    assert(!changes.Advance(299));
    assert(changes.Advance(301));
    assert(!changes.Advance(302));
    // Late event delivery and multiple changes during a render gap.
    changes.Schedule(500);
    changes.Schedule(400);
    assert(changes.Advance(600));
    assert(!changes.Advance(601));
}
