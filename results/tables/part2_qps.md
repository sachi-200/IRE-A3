| Dataset | Index | QPS @ 0.90 | setting (test recall) | QPS @ 0.95 | setting (test recall) |
|---|---|---|---|---|---|
| sift | Exact (Flat) | 21.2 | recall 1.0 | 21.2 | recall 1.0 |
| sift | HNSW | 8580.1 | M=16,efC=200, efSearch=32 (0.918) | 5099.4 | M=16,efC=200, efSearch=64 (0.971) |
| sift | IVF-Flat | 2330.0 | nlist=4096, nprobe=32 (0.933) | 1361.5 | nlist=16384, nprobe=128 (0.973) |
| sift | IVF-PQ | not reached | max tune recall 0.884 | not reached | max tune recall 0.884 |
| glove | Exact (Flat) | 21.7 | recall 1.0 | 21.7 | recall 1.0 |
| glove | HNSW | 1281.9 | M=32,efC=200, efSearch=192 (0.918) | 649.9 | M=32,efC=200, efSearch=384 (0.955) |
| glove | IVF-Flat | 581.3 | nlist=16384, nprobe=512 (0.923) | 320.6 | nlist=16384, nprobe=1024 (0.956) |
| glove | IVF-PQ | not reached | max tune recall 0.870 | not reached | max tune recall 0.870 |
