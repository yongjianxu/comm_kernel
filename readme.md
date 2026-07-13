# AllToAllv Kernel and Benchmark Requirements

## Objective

Develop and benchmark an AllToAllv device kernel that transfers a variable
number of tokens between ranks.

The implementation may use either:

- C++/CUDA with the NCCL Device API and GIN
- Python CuTeDSL with the nccl4py Device API bindings.
- NCCL_TESTS_ALLTOALLV_MATRIX_FILE to pass the token num matrix to app



## Reference Implementations

- [nccl-tests AllToAll](https://github.com/NVIDIA/nccl-tests/blob/master/src/alltoall.cu):
  reference for C++ Device API requirements, `GinAlltoAllKernel`, GIN
  connection setup, signal waiting, flushing, barriers, kernel specialization,
  and `-D` device-implementation dispatch.
- [nccl-tests AllToAllv](https://github.com/NVIDIA/nccl-tests/blob/master/src/alltoallv.cu):
  reference for matrix-file parsing, asymmetric peer sizes, packed send/receive
  offsets, expected-data generation, bandwidth calculation, and test-harness
  integration.
- [nccl4py CuTeDSL GIN example](https://github.com/NVIDIA/nccl/blob/master/bindings/nccl4py/examples/cute/main.py):
  reference for MPI/NCCL initialization, symmetric window registration,
  `ncclDevComm` creation, CuTeDSL `Gin.put`, completion signals, validation,
  and resource cleanup.
## Traffic Matrix Semantics

NCCL_TESTS_ALLTOALLV_MATRIX_FILE
64 66
512 516
4096 5000

`NCCL_TESTS_ALLTOALLV_MATRIX_FILE` shall specify **token counts**, rather than
raw byte counts:

```text
matrix[src_rank][dst_rank] = number of tokens sent from src_rank to dst_rank
```

Each token is exactly:

```text
7168 bytes (7 KiB)
```

Therefore:

```text
message_bytes[src][dst] = token_count[src][dst] * 7168
```

## Required Two-Rank Benchmark Cases

The supplied values are interpreted as three independent, asymmetric,
two-rank benchmark cases:

| Case | Rank 0 → Rank 1 | Rank 1 → Rank 0 |
|---|---:|---:|
| 1 | 64 tokens / 448 KiB | 66 tokens / 462 KiB |
| 2 | 512 tokens / 3.5 MiB | 516 tokens / 3612 KiB |
| 3 | 4096 tokens / 28 MiB | 5000 tokens / 35000 KiB |



## Benchmark and Validation

Run all three matrix cases with exactly two ranks and report:

- Transfer time.



