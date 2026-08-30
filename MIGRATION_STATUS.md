# Vit-Head migration status

Migration audit date: 2026-08-30.

## Active project roots

- `BLB/`
- `BOLT/`
- `IRON/`
- `OURS/`

Source files in these roots no longer reference
`MIND_code/Pencil/{BLB,BOLT,IRON,OURS}`. The BOLT virtual environment,
IRON runtime symlinks, and the `bolt_env` and `ours-head-hetero` container
definitions were also migrated to `Vit-Head` paths.

## Preserved assets

- `BOLT/checkpoints/vit_b16_hf/`
- `BOLT/checkpoints/deit_tiny_patch16_224/`
- `OURS/checkpoints/official-blb-mrpc/`
- `migration-archives/blb-ours-legacy-history.bundle`
- `migration-archives/blb-*-uncommitted-source.patch`
- `migration-archives/legacy-blb-top-README.md`

`migration-archives/checkpoints.sha256` records all 1,041 migrated checkpoint
files. Verify it from the repository root with:

```bash
sha256sum -c --quiet migration-archives/checkpoints.sha256
```

## Intentionally abandoned

Old project-level `outputs/` and `results/` directories, copied worktree build
directories, Python bytecode caches, and logs stored only in superseded Docker
containers are not part of the migration.

## Historical audit note

The legacy BLB formal-audit scripts now accept the migrated roots and find the
migrated checkpoints. They still report pre-existing algorithm-acceptance
blockers (execution-depth/oracle/preprocess gates); those are model-validation
issues, not migration-path failures.
