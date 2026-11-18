# R22-5 boundary matrix coverage (source preparation)

The experimental `test-backend-ops` case set `r22-5-v1` contains the lengths
33, 127, 128, 129, 512, 1023, 1024, 2047 and 2048. MUL_MAT and MUL_MAT_ID
exercise both output-row and token axes with F32 and Q8_0 inputs (36 cases each).
GATED_DELTA_NET exercises sequence tokens; RMS_NORM and RMS_NORM_MUL_ADD exercise
row width; FLASH_ATTN_EXT exercises KV length (nine cases each). Each selected
binary must produce the case-set marker and the entire expected non-skipped case
tally. A seeded GPU reference also requires a native reference metric for each.

The graph scheduler additionally covers ADD, MUL, SCALE, CLAMP, CONT, CPY,
CONCAT, GLU, UNARY, SUM_ROWS, GET_ROWS, SET_ROWS, ROPE, SOFT_MAX, ARGSORT,
SUB, DIV, SQR, SQRT, LOG, SIN, COS, FILL, DUP and TOPK_MOE. These operations
have no reviewed boundary mapping in this packet. Explicit
`boundary_matrix_correctness` requests for them refuse with `oracle_unavailable`;
legacy ordinary correctness checks retain their existing behavior. No universal
boundary coverage is claimed. The special custom scoped corpora also retain their
existing behavior. Physical CPU and GPU execution and numerical tolerance still
require validation before this source is deployed.
