#!/bin/bash
set -euo pipefail

usage() {
    echo "usage: sudo $0 <runner-uid> <runner-gid>" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
[[ $EUID -eq 0 ]] || {
    echo "provision_cgroup.sh must run as root (normally through sudo)" >&2
    exit 1
}
[[ $1 =~ ^[0-9]+$ && $2 =~ ^[0-9]+$ ]] || usage

readonly RUNNER_UID=$1
readonly RUNNER_GID=$2
readonly CGROUP_MOUNT=/sys/fs/cgroup
readonly AUTOKERNEL_CGROUP=/sys/fs/cgroup/autokernel

readonly CGROUP2_SUPER_MAGIC=63677270
[[ $(stat -f -c %t "$CGROUP_MOUNT") == "$CGROUP2_SUPER_MAGIC" ]] || {
    echo "$CGROUP_MOUNT is not a cgroup-v2 mount" >&2
    exit 1
}

mkdir -p "$AUTOKERNEL_CGROUP"
chown "$RUNNER_UID:$RUNNER_GID" "$AUTOKERNEL_CGROUP"
chmod 0700 "$AUTOKERNEL_CGROUP"

echo "AutoKernel cgroup delegation ready: $AUTOKERNEL_CGROUP uid=$RUNNER_UID gid=$RUNNER_GID"
