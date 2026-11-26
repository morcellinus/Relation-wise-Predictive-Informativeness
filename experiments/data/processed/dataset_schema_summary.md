# RPI Processed Dataset Schema Summary

Source files: `experiments/data/processed/{dataset}/{dataset}.pt`. Target node types follow `experiments/realworld/compute_rpi.py` and `experiments/realworld_alignment/run_backbone_alignment.py`. Counts and feature shapes were read directly from the processed `HeteroData` objects.

Feature type labels are empirical: `one-hot` means exactly one nonzero binary entry per sampled row; `multi-hot/sparse binary` means binary sparse features with possibly multiple active entries; `rich/dense numeric` means non-binary dense numeric features; `poor/no feature` would mean no stored `x` tensor.

## Predefined Target-to-Target Meta-Paths

Predefined meta-paths were checked against `DATASET_META` in `experiments/realworld_alignment/run_backbone_alignment.py`, then verified against the edge types stored in each processed `HeteroData` object. All six datasets have at least one predefined meta-path that starts and ends with the target node type. `aminer` has one additional predefined meta-path that is target-to-target by schema, but not valid for the checked-in processed file because the required `reference` node/relation types are absent.

| dataset | target node type | predefined meta-paths | start/end target? | valid in processed data? |
|---|---|---|---|---|
| `acm` | `paper` | (`paper`, `to`, `author`) -> (`author`, `to`, `paper`)<br>(`paper`, `to`, `subject`) -> (`subject`, `to`, `paper`) | yes, all | yes, all |
| `aminer` | `paper` | (`paper`, `to`, `author`) -> (`author`, `to`, `paper`)<br>(`paper`, `to`, `reference`) -> (`reference`, `to`, `paper`) | yes, all | partial: author path valid; reference path missing from processed data |
| `dblp` | `author` | (`author`, `to`, `paper`) -> (`paper`, `to`, `author`)<br>(`author`, `to`, `paper`) -> (`paper`, `to`, `term`) -> (`term`, `to`, `paper`) -> (`paper`, `to`, `author`)<br>(`author`, `to`, `paper`) -> (`paper`, `to`, `venue`) -> (`venue`, `to`, `paper`) -> (`paper`, `to`, `author`) | yes, all | yes, all |
| `freebase` | `movie` | (`movie`, `to`, `actor`) -> (`actor`, `to`, `movie`)<br>(`movie`, `to`, `director`) -> (`director`, `to`, `movie`)<br>(`movie`, `to`, `writer`) -> (`writer`, `to`, `movie`) | yes, all | yes, all |
| `imdb` | `movie` | (`movie`, `>actorh`, `actor`) -> (`actor`, `to`, `movie`)<br>(`movie`, `to`, `director`) -> (`director`, `to`, `movie`) | yes, all | yes, all |
| `yelp` | `business` | (`business`, `to`, `user`) -> (`user`, `to`, `business`)<br>(`business`, `to`, `service`) -> (`service`, `to`, `business`)<br>(`business`, `to`, `location`) -> (`location`, `to`, `business`) | yes, all | yes, all |

## acm

- Target node type: `paper`
- Other node types: `author`, `subject`, `term`

### Node Counts and Features

| node type | target? | # nodes | feature dim | feature dtype | feature type | feature stats |
|---|---:|---:|---:|---|---|---|
| `paper` | yes | 3,025 | 1902 | `float32` | multi-hot/sparse binary | avg nnz/row=88.00, density=0.0463 |
| `author` | no | 5,959 | 1902 | `float32` | multi-hot/sparse binary | avg nnz/row=167.30, density=0.0880 |
| `subject` | no | 56 | 1902 | `float32` | multi-hot/sparse binary | avg nnz/row=477.32, density=0.2510 |
| `term` | no | 1,902 | - | `-` | poor/no feature |  |

### Edge Counts by Relation Type

| relation type | # edges |
|---|---:|
| (`paper`, `cite`, `paper`) | 5,343 |
| (`paper`, `ref`, `paper`) | 5,343 |
| (`paper`, `to`, `author`) | 9,949 |
| (`author`, `to`, `paper`) | 9,949 |
| (`paper`, `to`, `subject`) | 3,025 |
| (`subject`, `to`, `paper`) | 3,025 |
| (`paper`, `to`, `term`) | 255,619 |
| (`term`, `to`, `paper`) | 255,619 |

## aminer

- Target node type: `paper`
- Other node types: `author`, `venue`

### Node Counts and Features

| node type | target? | # nodes | feature dim | feature dtype | feature type | feature stats |
|---|---:|---:|---:|---|---|---|
| `paper` | yes | 127,623 | - | `-` | poor/no feature |  |
| `author` | no | 164,473 | - | `-` | poor/no feature |  |
| `venue` | no | 101 | - | `-` | poor/no feature |  |

### Edge Counts by Relation Type

| relation type | # edges |
|---|---:|
| (`paper`, `to`, `author`) | 355,072 |
| (`author`, `to`, `paper`) | 355,072 |
| (`paper`, `to`, `venue`) | 127,623 |
| (`venue`, `to`, `paper`) | 127,623 |

## dblp

- Target node type: `author`
- Other node types: `paper`, `term`, `venue`

### Node Counts and Features

| node type | target? | # nodes | feature dim | feature dtype | feature type | feature stats |
|---|---:|---:|---:|---|---|---|
| `author` | yes | 4,057 | 334 | `float32` | multi-hot/sparse binary | avg nnz/row=12.03, density=0.0360 |
| `paper` | no | 14,328 | 4231 | `float32` | sparse numeric | avg nnz/row=6.08, density=0.0014 |
| `term` | no | 7,723 | 50 | `float32` | rich/dense numeric | avg nnz/row=50.00, density=1.0000 |
| `venue` | no | 20 | - | `-` | poor/no feature |  |

### Edge Counts by Relation Type

| relation type | # edges |
|---|---:|
| (`author`, `to`, `paper`) | 19,645 |
| (`paper`, `to`, `term`) | 85,810 |
| (`paper`, `to`, `venue`) | 14,328 |
| (`paper`, `to`, `author`) | 19,645 |
| (`term`, `to`, `paper`) | 85,810 |
| (`venue`, `to`, `paper`) | 14,328 |

## freebase

- Target node type: `movie`
- Other node types: `actor`, `director`, `writer`

### Node Counts and Features

| node type | target? | # nodes | feature dim | feature dtype | feature type | feature stats |
|---|---:|---:|---:|---|---|---|
| `movie` | yes | 3,492 | 128 | `float32` | rich/dense numeric | avg nnz/row=128.00, density=1.0000 |
| `actor` | no | 33,401 | 128 | `float32` | rich/dense numeric | avg nnz/row=128.00, density=1.0000 |
| `director` | no | 2,502 | 128 | `float32` | rich/dense numeric | avg nnz/row=128.00, density=1.0000 |
| `writer` | no | 4,459 | 128 | `float32` | rich/dense numeric | avg nnz/row=128.00, density=1.0000 |

### Edge Counts by Relation Type

| relation type | # edges |
|---|---:|
| (`movie`, `to`, `actor`) | 65,341 |
| (`actor`, `to`, `movie`) | 65,341 |
| (`movie`, `to`, `director`) | 3,762 |
| (`director`, `to`, `movie`) | 3,762 |
| (`movie`, `to`, `writer`) | 6,414 |
| (`writer`, `to`, `movie`) | 6,414 |

## imdb

- Target node type: `movie`
- Other node types: `director`, `actor`, `keyword`

### Node Counts and Features

| node type | target? | # nodes | feature dim | feature dtype | feature type | feature stats |
|---|---:|---:|---:|---|---|---|
| `movie` | yes | 4,932 | 3489 | `float32` | sparse numeric | avg nnz/row=26.34, density=0.0076 |
| `director` | no | 2,393 | 3341 | `float32` | sparse numeric | avg nnz/row=11.78, density=0.0035 |
| `actor` | no | 6,124 | 3341 | `float32` | sparse numeric | avg nnz/row=13.83, density=0.0041 |
| `keyword` | no | 7,971 | - | `-` | poor/no feature |  |

### Edge Counts by Relation Type

| relation type | # edges |
|---|---:|
| (`movie`, `to`, `director`) | 4,932 |
| (`director`, `to`, `movie`) | 4,932 |
| (`movie`, `>actorh`, `actor`) | 14,779 |
| (`actor`, `to`, `movie`) | 14,779 |
| (`movie`, `to`, `keyword`) | 23,610 |
| (`keyword`, `to`, `movie`) | 23,610 |

## yelp

- Target node type: `business`
- Other node types: `user`, `service`, `location`

### Node Counts and Features

| node type | target? | # nodes | feature dim | feature dtype | feature type | feature stats |
|---|---:|---:|---:|---|---|---|
| `business` | yes | 2,614 | 82 | `float32` | rich/dense numeric | avg nnz/row=13.60, density=0.1658 |
| `user` | no | 1,286 | 82 | `float32` | rich/dense numeric | avg nnz/row=31.76, density=0.3873 |
| `service` | no | 4 | 82 | `float32` | rich/dense numeric | avg nnz/row=74.25, density=0.9055 |
| `location` | no | 9 | 82 | `float32` | rich/dense numeric | avg nnz/row=61.56, density=0.7507 |

### Edge Counts by Relation Type

| relation type | # edges |
|---|---:|
| (`business`, `to`, `service`) | 2,614 |
| (`service`, `to`, `business`) | 2,614 |
| (`business`, `to`, `location`) | 2,614 |
| (`location`, `to`, `business`) | 2,614 |
| (`business`, `to`, `user`) | 30,838 |
| (`user`, `to`, `business`) | 30,838 |
