# Graph and feature stats

## Settings
- input: `popqa_clean.csv`
- max group size (K): 50
- seed: 0
- popularity buckets: 4 (quartiles of `s_pop`)

## Features
- rows: 13059
- bucket edges (`s_pop`): 2.0, 231.5, 967.0, 5,660.0, 15,101,521.0
- bucket 0: 3265 facts
- bucket 1: 3265 facts
- bucket 2: 3264 facts
- bucket 3: 3265 facts

## Graph
- nodes: 13059
- edges: 28676 (head: 1651, tail: 27025)
- isolated nodes: 4528
- degree min / median / max: 0 / 1 / 46
- capped entity groups: 10 (head: 0, tail: 10)
- pairs sharing both a head and a tail entity (stored once as head): 571
