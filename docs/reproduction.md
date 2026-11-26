# Reproduce the paper

This guide covers the main-paper experiments for [Beyond Homophily](../README.md), including the HCSBM generator, HGNN alignment, and fixed-budget pruning.

The [supplementary material](WSDM2027-supplementary.pdf) contains the additional proofs, experimental details, ablations, validation analyses, and fixed-budget retraining protocol.

## Main-paper artifacts

| Artifact | Experiment | Generated output |
| --- | --- | --- |
| Figure 3 | HCSBM one-hop accuracy, RPI, and target representations | `experiments/results/main/figures/figure_3.{png,pdf}` |
| Figure 4 | HCSBM reinforcement and dilution | `experiments/results/main/figures/figure_4.{png,pdf}` |
| Figure 5 | HCSBM first-hop-only and second-hop-only typed paths | `experiments/results/main/figures/figure_5.{png,pdf}` |
| Figure 6 | Relation-level operational RPI distributions | `experiments/results/main/figures/figure_6.{png,pdf}` |
| Figure 7 | Dataset-level RPI(t), MHR, and CHR with split standard deviations | `experiments/results/main/figures/figure_7.{png,pdf}` |
| Figure 8 | Diagnostic-score alignment with HGNN gains and split standard deviations | `experiments/results/main/figures/figure_8.{png,pdf}` |
| Table 2 | Destination-type aggregation robustness | `experiments/results/main/tables/table_2/table_2.tex` |
| Table 3 | Fixed 1% R-GCN pruning with full-graph and global-random controls | `experiments/results/main/tables/table_3/table_3.tex` |

Figures 1 and 2 are conceptual illustrations and do not require experiment code. Table 1 reports statistics of the bundled datasets.

## Environment

The reproduced environment uses Python 3.11, PyTorch 2.4.0, PyG 2.6.1, NumPy 1.26.4, SciPy 1.13.1, pandas 2.3.2, scikit-learn 1.7.1, and Matplotlib 3.10.8.

From the repository root, create a virtual environment and install the pinned experiment dependencies. Install Git LFS before fetching the data.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/requirements.txt
```

For full reproduction, fetch the complete graph bundle (about 770 MB) and verify it:

```bash
git lfs install
git lfs pull --include="experiments/data/processed/**"
python experiments/scripts/verify_bundle.py
```

`experiments/scripts/verify_bundle.py` loads all eleven graph files and verifies all 330 dataset/split/model configurations. The expected final status is `ok`.

## Reproduction

Run every main-paper experiment and build all artifacts:

```bash
bash experiments/scripts/run_all.sh
```

For a long run, start a tmux session from the repository root:

```bash
tmux new -s rpi-reproduction
source .venv/bin/activate
mkdir -p experiments/results
bash experiments/scripts/run_all.sh 2>&1 | tee experiments/results/reproduction.log
```

The three experiment groups can also be reproduced independently:

```bash
bash experiments/scripts/run_hcsbm.sh
bash experiments/scripts/run_realworld.sh
bash experiments/scripts/run_pruning.sh
```

The HCSBM command generates fresh synthetic graphs, trains SimpleHeteroSAGE over five seeds, writes raw summaries, and exports Figures 3-5. The real-world command computes operational RPI, MHR, and CHR on five stratified 60/20/20 splits; trains MLP, HAN, HetGNN, R-GCN, RSHN, and Simple-HGN; and exports Figures 6-8 and Table 2. The pruning command computes split-specific operational RPI, performs all paired R-GCN retraining runs, and exports Table 3.

Successful runs are resumable. Raw CSV files remain under the corresponding experiment directory, and artifact builders consume those newly generated files rather than bundled model outputs.

## Datasets

Git LFS stores processed PyG `HeteroData` files under `experiments/data/processed/` for ACM, AIFB, AM, AMiner, BGS, DBLP, Freebase, IMDB, LIAR, MUTAG, and Yelp. No symlinks or external dataset paths are used at runtime. Target node types are defined in `experiments/realworld/compute_rpi.py` and `experiments/realworld_alignment/run_backbone_alignment.py`. Every real-world command regenerates the same stratified masks from seed 42 with split offsets of 1009.

## Output layout

```text
experiments/results/
  hcsbm/                    HCSBM runs and raw summaries
  realworld_alignment/      metrics, model runs, and alignment tables
  table_3_fixed_1pct/       operational RPI and paired pruning runs
  main/
    figures/                Figures 3-8
    tables/table_2/         aggregation CSVs and Table 2 LaTeX
    tables/table_3/         paired cases, seed-level values, CSV, Markdown, and LaTeX
```
