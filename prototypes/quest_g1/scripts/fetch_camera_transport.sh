#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
revision=514a0b968503d758a2954ff9016289f41f489616
if [ ! -d vendor/IXWebSocket/.git ]; then
    git init vendor/IXWebSocket
    git -C vendor/IXWebSocket remote add origin https://github.com/machinezone/IXWebSocket.git
fi
if ! git -C vendor/IXWebSocket cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C vendor/IXWebSocket fetch --depth 1 origin "$revision"
fi
git -C vendor/IXWebSocket checkout --detach "$revision"
