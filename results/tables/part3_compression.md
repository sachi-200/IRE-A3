| m | Variant | Bytes/vector | Build time (s) | Best test recall | QPS @ 0.90 (nprobe, test recall) | QPS @ 0.95 (nprobe, test recall) |
|---|---|---|---|---|---|---|
| 8 | ivfpq | 18.3 | 112.3 | 0.4171 | not reached | not reached |
| 8 | ivfpq+rerank100 | 530.3 | 112.7 | 0.8909 | not reached | not reached |
| 16 | ivfpq | 26.3 | 106.8 | 0.5957 | not reached | not reached |
| 16 | ivfpq+rerank100 | 538.3 | 107.2 | 0.9855 | 2748.4 (32, 0.926) | 1736.6 (64, 0.967) |
| 32 | ivfpq | 42.3 | 106.1 | 0.7462 | not reached | not reached |
| 32 | ivfpq+rerank100 | 554.3 | 106.5 | 0.9941 | 2017.4 (32, 0.933) | 1297.1 (64, 0.977) |
| 64 | ivfpq | 74.3 | 112.7 | 0.8850 | not reached | not reached |
| 64 | ivfpq+rerank100 | 586.3 | 113.1 | 0.9942 | 1355.3 (32, 0.933) | 769.3 (64, 0.977) |
| - | IVF-Flat (same run) | 522.1 | 87.7 | 0.9942 | 1876.4 (32, 0.933) | 1113.8 (64, 0.977) |
