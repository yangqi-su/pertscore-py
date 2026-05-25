# pertscore-py

`pertscore-py` calculates perturbation scores from `.h5ad` perturb-seq data using the streamed `exact_fast` PS score implementation.

## What The PS Score Does

For each perturbation, the method first identifies target genes, fits perturbation effect vectors from normalized expression, and then scores each perturbed cell against the effect vector for its observed perturbation. Scores are bounded between `0` and `scale_factor`, then scaled to `0-1` by default.

In the default `target_mode="union_deg"`, target genes are selected per perturbation by comparing perturbed cells against control cells with streamed Welch t-statistics. Genes can first be filtered by absolute log2 fold change with `logfc_threshold`, then ranked by absolute Welch t-score. Wilcoxon is not used because it depends on cell-level ranks, which are much harder to compute cleanly while streaming backed `.h5ad` chunks. Welch ranking only needs sums, squared sums, and counts, so it fits the large-file streaming design.

After target genes are selected, the method solves two main optimization problems:

- Ridge regression estimates each perturbation effect vector, called `beta`, on the union target gene set.
- In multilabel mode, bounded quadratic optimization assigns per-perturbation scores for cells with multiple active perturbations. Small active sets are solved by active-set enumeration; larger active sets use L-BFGS-B.

The implementation is designed for large `.h5ad` files. It streams expression chunks from disk, avoids materializing a full cell-by-perturbation score matrix, and writes a long CSV table of per-cell scores.

## Input Expectations

- Input is an `.h5ad` file or an in-memory `AnnData` object.
- Expression should be raw/count-like, nonnegative values in `adata.X` or a selected `adata.layers[...]` layer.
- The method internally applies library-size normalization to `target_sum=10000` followed by `log1p`.
- If your `adata.X` is already log-normalized, put raw counts in a layer and pass that layer name.
- Multilabel perturbations are represented with `+`, for example `GENE1+GENE2`.

## Install Locally

From this repository:

```bash
pip install -e .
```

## CLI Usage

Single-label perturbations, using counts in `adata.X`:

```bash
ps_score_exact_fast \
  --dataset-path input.h5ad \
  --output-dir ps_out \
  --mode single \
  --perturb-column perturbation \
  --ctrl-name control \
  --target-mode union_deg \
  --target-gene-max 500 \
  --logfc-threshold 0.1 \
  --clip-quantile 0.95 \
  --chunk-size 8192 \
  --progress
```

If raw counts are in a layer:

```bash
ps_score_exact_fast \
  --dataset-path input.h5ad \
  --output-dir ps_out \
  --mode single \
  --perturb-column perturbation \
  --ctrl-name control \
  --layer counts
```

Multilabel perturbations:

```bash
ps_score_exact_fast \
  --dataset-path input.h5ad \
  --output-dir ps_out \
  --mode multilabel \
  --perturb-column perturbation \
  --ctrl-name control
```

Outputs:

```text
ps_out/ps-score-exact-fast.csv
ps_out/ps-score-exact-fast-manifest.json
```

Progress bars and phase messages are off by default. Add `--progress` to show when the program is finding target genes, estimating beta, and scoring cells.

## Background Correction

Background correction is turned on by passing an obs column with background groups, usually a cell type, batch, or other grouping where each group should have its own control baseline:

```bash
ps_score_exact_fast \
  --dataset-path input.h5ad \
  --output-dir ps_out \
  --mode single \
  --perturb-column perturbation \
  --ctrl-name control \
  --background-cluster-column cell_type \
  --progress
```

With background correction, control cells are averaged within each background cluster. Perturbation effects are then estimated after subtracting the matched cluster control baseline. Each modeled background cluster must contain control cells; otherwise the run fails instead of silently borrowing another baseline.

## Python API Usage

```python
from pertscore import run_ps_score_exact_fast

manifest = run_ps_score_exact_fast(
    "input.h5ad",
    output_dir="ps_out",
    mode="single",
    perturb_column="perturbation",
    ctrl_name="control",
    layer=None,
    show_progress=True,
)
```

For in-memory use without writing files:

```python
from pertscore import run_ps_score_exact_fast

result = run_ps_score_exact_fast(
    "input.h5ad",
    output_dir=None,
    mode="single",
    perturb_column="perturbation",
    ctrl_name="control",
)

scores = result.scores
metadata = result.metadata
```

Background correction from Python:

```python
from pertscore import run_ps_score_exact_fast

manifest = run_ps_score_exact_fast(
    "input.h5ad",
    output_dir="ps_out",
    mode="single",
    perturb_column="perturbation",
    ctrl_name="control",
    background_cluster_column="cell_type",
    show_progress=True,
)
```

## Main Defaults

- `mode="single"`
- `target_mode="union_deg"`
- `target_gene_max=500`
- `logfc_threshold=0.1`
- `clip_quantile=None` unless supplied on the CLI
- `chunk_size=8192`
- `target_sum=10000`
- `lr_lambda=0.01`
- `score_lambda=0.0`
- `scale_factor=3.0`
- `scale_score=True`
- `show_progress=False`

## Notes

- Use `target_mode="hvg"` only when `adata.var["highly_variable"]` is already set.
- Use `mode="multilabel"` only when one cell may contain multiple perturbation tokens.
- The current parser uses `+` as the multilabel delimiter.
