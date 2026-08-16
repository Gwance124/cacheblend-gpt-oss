#!/usr/bin/env bash

export CACHEBLEND_P7_RUN_BASE_DIR="${CACHEBLEND_P7_RUN_BASE_DIR:-/mnt/nvme2/mlee/cacheblend-gpt-oss-artifacts/browsecomp-append-only-cacheblend-selective-20260816}"
export CACHEBLEND_G3_RUN_POINTER="${CACHEBLEND_G3_RUN_POINTER:-/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-selective-browsecomp-append-only-20260816.current}"
export CACHEBLEND_P7_VALIDATE_SELECTIVE=yes

bash /mnt/nvme2/mlee/cacheblend-gpt-oss/local-m85-p7-browsecomp.sh
true
