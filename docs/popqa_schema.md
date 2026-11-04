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
