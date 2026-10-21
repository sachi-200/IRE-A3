| Dataset | Index | Setting | Build time (s) | Index size (MB) | Bytes/vector | Test recall | QPS |
|---|---|---|---|---|---|---|---|
| sift | Exact (Flat) | - | 0.40 | 512.0 | 512.0 | 1.0000 | 21.2 |
| sift | HNSW | M=16,efC=200, efSearch=64 | 143.5 | 656.3 | 656.3 | 0.9712 | 5099.4 |
| sift | IVF-Flat | nlist=16384, nprobe=128 | 199.6 | 528.5 | 528.5 | 0.9731 | 1361.5 |
| sift | IVF-PQ | nlist=4096,m=64,nbits=8, nprobe=512 (best; 0.95 not reached) | 51.9 | 74.3 | 74.3 | 0.8850 | 130.4 |
| glove | Exact (Flat) | - | 0.31 | 473.4 | 400.0 | 1.0000 | 21.7 |
| glove | HNSW | M=32,efC=200, efSearch=384 | 270.0 | 795.5 | 672.2 | 0.9546 | 649.9 |
| glove | IVF-Flat | nlist=16384, nprobe=1024 | 203.7 | 489.6 | 413.6 | 0.9562 | 320.6 |
| glove | IVF-PQ | nlist=4096,m=50,nbits=8, nprobe=1024 (best; 0.95 not reached) | 46.3 | 70.4 | 59.5 | 0.8675 | 79.2 |
