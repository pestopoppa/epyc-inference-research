#!/bin/bash
set -uo pipefail
cd /mnt/raid0/llm/tmp/gpu_coresidency
IMGDIR=/mnt/raid0/llm/epyc-inference-research/test_images/vl_rubric
IMGS=(chart_bar.png code_python.png diagram_flowchart.png doc_invoice.png math_equation.png blueprint.png scientific_figure.png puzzle_grid.png)
i=0
while [ ! -f /mnt/raid0/llm/tmp/gpu_coresidency/.stop_load ]; do
  img="${IMGS[$((i % ${#IMGS[@]}))]}"
  python3 vision_query.py "$IMGDIR/$img" "Describe this image in detail, including all text, numbers and structural relationships." 2>&1 | tail -1
  i=$((i+1))
done
echo "vision_loop finished after $i queries"
