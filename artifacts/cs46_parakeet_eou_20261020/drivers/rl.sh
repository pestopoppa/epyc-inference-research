#!/bin/bash
# build-role claim on ONE region (q0); blocks until granted. All paths absolute.
exec /workspace/repos/epyc-orchestrator/scripts/region-lock run --regions "${CS46_REGION:-q0}" --role build --tag cs-46-parakeet-eou -- "$@"
