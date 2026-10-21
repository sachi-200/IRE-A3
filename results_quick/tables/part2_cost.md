| Dataset | Index | Setting | Build time (s) | Index size (MB) | Bytes/vector | Test recall | QPS |
|---|---|---|---|---|---|---|---|
| sift | Exact (Flat) | - | 0.02 | 51.2 | 512.0 | 1.0000 | 340.3 |
| sift | HNSW | M=32,efC=200, efSearch=32 | 12.5 | 78.4 | 784.2 | 0.9720 | 12765.2 |
| sift | IVF-Flat | nlist=1024, nprobe=32 | 1.6 | 52.5 | 525.3 | 0.9652 | 8596.5 |
| sift | IVF-PQ | nlist=256,m=64,nbits=8, nprobe=128 (best; 0.95 not reached) | 7.8 | 7.5 | 74.6 | 0.8912 | 307.4 |
| glove | Exact (Flat) | - | 0.03 | 40.0 | 400.0 | 1.0000 | 389.3 |
| glove | HNSW | M=32,efC=200, efSearch=128 | 25.7 | 67.2 | 672.2 | 0.9510 | 1951.6 |
| glove | IVF-Flat | nlist=1024, nprobe=256 | 1.9 | 41.2 | 412.2 | 0.9646 | 1345.4 |
| glove | IVF-PQ | nlist=256,m=50,nbits=8, nprobe=128 (best; 0.95 not reached) | 7.2 | 6.0 | 60.1 | 0.8624 | 397.1 |
