#!/bin/bash
set -euo pipefail
R=/mnt/raid0/llm/tmp/cs-46-20261020/rust
export RUSTUP_HOME=$R/rustup CARGO_HOME=$R/cargo PATH=$R/cargo/bin:$PATH
export CARGO_TARGET_DIR=$R/target
cd $R/parakeet-rs
cargo build --release -j16 --example eou_dump --no-default-features --features cpu,api-28,ort/tracing,ort/download-binaries,ort/copy-dylibs,ort/tls-rustls
