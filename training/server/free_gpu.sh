#!/bin/bash
# Print UUIDs of idle H100s: no compute processes, <100 MiB used, 0% utilisation. The server is shared: never take a busy GPU.
busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
nvidia-smi --query-gpu=uuid,name,memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' -v busy="$busy" '$2 ~ /H100/ && $3 < 100 && $4 == 0 && index(busy, $1) == 0 {print $1}'
