# Patched SEEM files

`train_rba.py` runs on SEEM v1.0 with two of its files changed. Copy **your** working versions of these
files into this folder before you upload the repository. They are the files the paper was run with.

| File in this folder | Where it comes from (inside your SEEM folder) | What was changed |
|---|---|---|
| `seem_v0.py` | `modeling/interface/seem_v0.py` | learned positional click prompt (`spatial_pos_prompt`, `pos_prompt_scale`, `content_gate`); returns the spatial candidate masks `pred_smasks` |
| `attention_data_struct_seemv0.py` | `modeling/interface/prototype/attention_data_struct_seemv0.py` | your changes to the spatial results, if any (optional: without it SEEM's original is used; `train_rba.py` taps the query vectors at run time) |

`setup_seem.py` copies the files into `SEEM/` and checks that `seem_v0.py` contains the positional prompt. It keeps the original files as `*.orig`.

The two files are derived from SEEM and keep its Apache License 2.0 header.
Do not remove that header when you copy them here.
