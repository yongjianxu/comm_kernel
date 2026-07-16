# AllToAllv Kernel and Benchmark Requirements

## Objective

Implement and benchmark a **device-initiated AllToAllv kernel** that
transfers a variable number of tokens between **two ranks**, where each rank
is one NVIDIA DGX Spark (GB10) and the two machines are connected through
their ConnectX-7 200 GbE ports (RoCE v2).

"Device-initiated" means a CUDA kernel drives the communication: it posts the
transfers, observes their completion, and consumes the received tokens —
without returning to the host between operations. This is the communication
pattern of MoE expert-parallel dispatch/combine (cf. DeepEP, NCCL Device API
GIN): each token is 7168 bytes — one fp8 DeepSeek-V3 hidden vector.

## Objective

Develop and benchmark an AllToAllv device kernel that transfers a variable
number of tokens between ranks.

## Traffic Matrix Semantics

The values define three independent two-rank matrix files. Self-traffic is
zero, and the off-diagonal entries specify the token counts exchanged between
rank 0 and rank 1.

### Case 1

```text
0 64
66 0
```

### Case 2

```text
0 512
516 0
```

### Case 3

```text
0 4096
5000 0
```


## Benchmark and Validation

Run all three matrix cases with exactly two ranks and report:

- Transfer time.


