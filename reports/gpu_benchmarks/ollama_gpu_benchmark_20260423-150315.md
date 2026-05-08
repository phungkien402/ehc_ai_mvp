# GPU Benchmark Report

- Timestamp: 20260423-150315
- Host: ehcaihelpdeskserver.tail4517fa.ts.net
- Model: qwen3.5:9b
- Base URL: http://127.0.0.1:11434
- Samples: 3
- Warmup runs: 1
- Prompt tokens: 21
- Requested output tokens: 128

## Summary

- Avg latency: 2.370 s
- Min latency: 2.306 s
- Max latency: 2.468 s
- P50 latency: 2.335 s
- P95 latency: 2.455 s
- Stddev: 0.086 s
- Avg eval rate: 76.34 tok/s
- Avg prompt eval rate: 624.81 tok/s

## nvitop Snapshot

```
Thu Apr 23 15:03:17 2026
+=============================================================================+
| NVITOP 1.6.2      Driver Version: 570.211.01      CUDA Driver Version: 12.8 |
|-------------------------------+----------------------+----------------------|
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  ..la V100-SXM2-16GB  Off | 00000000:04:00.0 Off |                    0 |
| N/A   44C   P0     69W / 300W |   9319MiB / 16384MiB |     48%      Default |
|-------------------------------+----------------------+----------------------|
|   1  ..la V100-SXM2-16GB  Off | 00000000:82:00.0 Off |                    0 |
| N/A   40C   P0     58W / 300W |  309.1MiB / 16384MiB |      0%      Default |
+===============================+======================+======================+
[ CPU: | 1.6%            UPTIME: 1:19:36 ]  ( Load Average:  0.27  0.32  0.38 )
[ MEM: || 4.6%             USED: 5.82GiB ]  [ SWP: | 0.0%                     ]

+=============================================================================+
| Processes:                                    phungkien@ehcaihelpdeskserver |
| GPU     PID      USER  GPU-MEM %SM %GMBW  %CPU  %MEM  TIME  COMMAND         |
|=============================================================================|
|   0   18830 C phungk+  9316MiB  76    53   0.0   1.2  5:39  /usr/local/bi.. |
|-----------------------------------------------------------------------------|
|   1   18830 C phungk+ 306.0MiB   0     0   0.0   1.2  5:39  /usr/local/bi.. |
+=============================================================================+
```

## Runs

| Run | Elapsed (s) | Total (s) | Load (s) | Prompt tokens | Eval tokens | Eval rate (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.335 | 2.332 | 0.474 | 38 | 128 | 77.25 |
| 2 | 2.468 | 2.465 | 0.578 | 38 | 128 | 76.29 |
| 3 | 2.306 | 2.303 | 0.410 | 38 | 128 | 75.49 |

## Notes

- This report measures Ollama inference latency and GPU residency, not gaming or graphics benchmarks.
- For repeatability, use the same prompt, model, and token budget when comparing runs.
