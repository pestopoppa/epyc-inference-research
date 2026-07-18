# Existing ODL/PaddleOCR Comparison

Schema: `odl_bench.existing_comparison.v1`
GT: `/mnt/raid0/llm/opendataloader-bench/demo_data/omnidocbench_demo/OmniDocBench_demo.json`

| engine | structural Edit_dist (lower) | table TEDS (higher) | reading-order Edit_dist (lower) | latency_ms median (lower) | sources |
| --- | ---: | ---: | ---: | ---: | --- |
| paddleocr_vl_default | 0.343019 | 0 | 0.337318 | 2918.78 | 1 |
| paddleocr_vl_html_tables | 0.429062 | 0 | 0.285753 | 3245.60 | 2 |
| paddleocr_vl_postprocessed | 0.34354 | 0.0583333 | 0.350138 |  | 3 |
| opendataloader_end2end | 0.356126 | 0.783813 | 0.216996 |  | 4 |

## Sources

1. `row_set` `paddleocr_vl_default` `/mnt/raid0/llm/tmp/odl-paddleocr-vl-demo-20260717T200212Z/model_gated_row_set.json`
2. `row_set` `paddleocr_vl_html_tables` `/mnt/raid0/llm/tmp/odl-paddleocr-vl-htmltables-20260717T201106Z/model_gated_row_set.json`
3. `metric_result` `paddleocr_vl_postprocessed` `/mnt/raid0/llm/opendataloader-bench/result/paddleocr_vl_1_6_postprocessed_20260717T211432Z_quick_match_metric_result.json`
4. `metric_result` `opendataloader_end2end` `/mnt/raid0/llm/opendataloader-bench/result/end2end_quick_match_metric_result.json`
