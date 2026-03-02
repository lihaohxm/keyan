# Grouping Consistency Notes

Before:
- Urgent/normal grouping drifted across files (fixed-index fallbacks like first users, hard-ratio defaults, and direct profile field slicing).
- Some modules used `geom.user_type`, others inferred groups implicitly.

Now:
- Unified source of truth is `profile.groups.urgent_idx` / `profile.groups.normal_idx`.
- All urgent/normal index reads are normalized through `normalize_index_vector(x, K)` before use.
- `build_profile_urgent_normal` is the canonical place that converts `geom.user_type` into `profile.groups.*`.
- `evaluate_system_rsma` and sweep helpers assert:
  - urgent/normal sets are disjoint
  - their union covers all users exactly once.
- `debug_eval` now prints urgent index count and head indices for quick diagnostics.
