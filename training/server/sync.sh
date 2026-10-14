#!/bin/bash
# Push git-tracked training code and server.env to the GPU server. Never deletes remote files.
set -euo pipefail
cd "$(dirname "$0")/.."
source server/server.env
ssh "$FURRY_HOST" "mkdir -p $FURRY_ROOT/code/server"
git ls-files -z quest_contact_clearance server | rsync -a --from0 --files-from=- ./ "$FURRY_HOST:$FURRY_ROOT/code/"
rsync -a server/server.env "$FURRY_HOST:$FURRY_ROOT/code/server/"
git rev-parse HEAD | ssh "$FURRY_HOST" "cat > $FURRY_ROOT/code/REVISION"
