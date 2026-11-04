# PopQA cleaned dataset schema

`data/processed/popqa_clean.csv` is the cleaned PopQA test split (`akariasai/PopQA`), produced by `src/load_popqa.py`. It has 34 columns and 13,059 rows (14,267 raw rows; 1,208 duplicate-question rows removed, 0 rows dropped for missing values). One row is one (subject, relation, object) fact with its question.

Dtypes are as read back by `pandas.read_csv`. Example values come from the row with `id` = 6 (long lists are abbreviated with `…`).

## Columns

| Column | dtype | Description | Example |
|---|---|---|---|
| `id` | int64 | Unique PopQA question id. Rows are sorted by this. | `6` |
| `subj` | object | Subject entity name. 3 nulls. | `Boston` |
| `prop` | object | Relation type (16 values). | `capital of` |
| `obj` | object | Gold object (the answer). | `Suffolk County` |
| `subj_id` | int64 | Numeric id of the subject entity. | `1` |
| `prop_id` | int64 | Numeric id of the relation. | `97` |
| `obj_id` | int64 | Numeric id of the object entity. | `1778561` |
| `s_aliases` | object | JSON list (as string) of alternative names for the subject. | `["Beantown","The Hub", …]` |
| `o_aliases` | object | JSON list (as string) of alternative names for the object. | `["Suffolk County, Massachusetts"]` |
| `s_uri` | object | Wikidata URI of the subject. | `http://www.wikidata.org/entity/Q100` |
| `o_uri` | object | Wikidata URI of the object. | `http://www.wikidata.org/entity/Q54072` |
| `s_wiki_title` | object | Wikipedia title of the subject. | `Boston` |
| `o_wiki_title` | object | Wikipedia title of the object. 14 nulls. | `Suffolk County, Massachusetts` |
| `s_pop` | int64 | Subject popularity (Wikipedia page views). Our main popularity signal. | `170080` |
| `o_pop` | int64 | Object popularity (same measure). | `9390` |
| `question` | object | Natural-language question. | `What is Boston the capital of?` |
| `possible_answers` | object | JSON list (as string) of acceptable answers. Parse with `json.loads`. Never empty. | `["Suffolk County", "Suffolk County, Massachusetts", "Massachusetts", …]` |
| `s_pop_log` | float64 | `log10(s_pop + 1)`. Added by the cleaning script. | `5.2306558` |
| `rel_<relation>` | int64 | One-hot encoding of `prop`: exactly one of the 16 columns is 1 per row. Names are lowercased with non-alphanumerics replaced by `_`. | `rel_capital_of` = `1`, all others `0` |

The 16 one-hot columns are: `rel_author`, `rel_capital`, `rel_capital_of`, `rel_color`, `rel_composer`, `rel_country`, `rel_director`, `rel_father`, `rel_genre`, `rel_mother`, `rel_occupation`, `rel_place_of_birth`, `rel_producer`, `rel_religion`, `rel_screenwriter`, `rel_sport`.

## Cleaning steps

1. Drop rows with a missing or empty `question`, `possible_answers`, or `s_pop`.
2. Normalize each question (lowercase, strip punctuation, collapse whitespace). For rows that share a normalized question, keep the one with the highest `s_pop` (lowest `id` breaks ties).
3. Add `s_pop_log` and the `rel_*` one-hot columns.

## How to regenerate

`data/processed/` is gitignored, so the CSV is **not** in git. Each teammate should generate it locally from the repository root:

```bash
python src/load_popqa.py            # writes data/processed/popqa_clean.csv
python src/load_popqa.py --out path/to/file.csv
```

The script prints the raw schema, row counts, and a short summary. Output is deterministic, so every run produces an identical file. The first run downloads the dataset from Hugging Face.

# Features and fact graph

`src/build_features_graph.py` reads `popqa_clean.csv` and writes two more files. Like the cleaned CSV, they are in the gitignored `data/processed/`, so regenerate them locally (run `load_popqa.py` first if `popqa_clean.csv` is missing):

```bash
python src/build_features_graph.py                   # writes both files and results/graph_feature_stats.md
python src/build_features_graph.py --max-group 30 --seed 1
```

Output is deterministic for a given input, `--max-group`, and `--seed`. Row counts, bucket sizes, and graph statistics are written to `results/graph_feature_stats.md`.

## `data/processed/popqa_features.csv`

One row per fact (13,059 rows, 31 columns, same `id`s and order as `popqa_clean.csv`). Example values come from `id` = 6.

| Column | dtype | Description | Example |
|---|---|---|---|
| `id` | int64 | Fact id. Join key for forgetting labels and the graph edge list. | `6` |
| `prop` | object | Relation type, copied from the cleaned file. | `capital of` |
| `s_pop` | int64 | Subject popularity (raw). | `170080` |
| `o_pop` | int64 | Object popularity (raw). 15 rows have the value `-2`; left unchanged. | `9390` |
| `s_pop_log` | float64 | `log10(s_pop + 1)`. | `5.2306558` |
| `o_pop_log` | float64 | `log10(o_pop + 1)`, with negative `o_pop` clipped to 0 first. | `3.9727118` |
| `pop_bucket` | int64 | Subject-popularity quartile from `pd.qcut` on `s_pop`: 0 = least popular, 3 = most popular. About 3,265 facts each. | `3` |
| `pop_bucket_0` ... `pop_bucket_3` | int64 | One-hot encoding of `pop_bucket`. | `pop_bucket_3` = `1`, others `0` |
| `s_pop_z`, `o_pop_z`, `s_pop_log_z`, `o_pop_log_z` | float64 | Z-score (mean 0, std 1) of the matching column above. | `s_pop_log_z` = `2.2344621` |
| `rel_<relation>` (16 columns) | int64 | One-hot relation type, same as in `popqa_clean.csv`. | `rel_capital_of` = `1` |

## `data/processed/popqa_graph_edges.csv`

Undirected edge list of the fact graph. Each fact (`id`) is a node, and each edge is stored once with `src_id` < `dst_id`. There are no self-loops or duplicate edges. Facts are linked if they share a head entity (`s_uri`) or a tail entity (`o_uri`). Entity groups larger than `--max-group` (default 50) are linked in a seeded random ring instead of a full clique. Facts with no shared entity are isolated nodes and do not appear in this file, so use `id` from the features file as the full node list.

| Column | dtype | Description | Example |
|---|---|---|---|
| `src_id` | int64 | `id` of the lower-id fact. | `1130` |
| `dst_id` | int64 | `id` of the higher-id fact. | `285168` |
| `edge_type` | object | `head` (same subject entity) or `tail` (same object entity). A pair that shares both is stored once as `head`. | `tail` |
