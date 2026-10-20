| Dataset | Index | QPS @ 0.90 | setting (test recall) | QPS @ 0.95 | setting (test recall) |
|---|---|---|---|---|---|
| sift | Exact (Flat) | 340.3 | recall 1.0 | 340.3 | recall 1.0 |
| sift | HNSW | 18982.0 | M=32,efC=200, efSearch=16 (0.912) | 12765.2 | M=32,efC=200, efSearch=32 (0.972) |
| sift | IVF-Flat | 6027.8 | nlist=1024, nprobe=32 (0.965) | 6027.8 | nlist=1024, nprobe=32 (0.965) |
| sift | IVF-PQ | not reached | max tune recall 0.889 | not reached | max tune recall 0.889 |
| glove | Exact (Flat) | 389.3 | recall 1.0 | 389.3 | recall 1.0 |
| glove | HNSW | 3582.8 | M=16,efC=200, efSearch=128 (0.901) | 1951.6 | M=32,efC=200, efSearch=128 (0.951) |
| glove | IVF-Flat | 2311.0 | nlist=1024, nprobe=128 (0.918) | 1345.4 | nlist=1024, nprobe=256 (0.965) |
| glove | IVF-PQ | not reached | max tune recall 0.861 | not reached | max tune recall 0.861 |
