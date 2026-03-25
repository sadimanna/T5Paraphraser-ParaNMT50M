# save as scripts/check_cache_specials.py or run inline
import os, glob
from transformers import T5Tokenizer
from data_classes import _safe_torch_load

cache_dir = "./dataset/train_cache_multi"
files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
print("files:", len(files))

data = None
item = None
for path in files:
    shard = _safe_torch_load(path)
    if shard:
        data = shard
        item = shard[5]
        print("using shard:", path, "items:", len(shard))
        break

if item is None:
    raise SystemExit("No non-empty shards found in cache_dir")

tok = T5Tokenizer.from_pretrained("google-t5/t5-small")
print("specials:", tok.special_tokens_map)

print("input w/ specials:", tok.decode(item["input_ids"], skip_special_tokens=False))
print("labels w/ specials:", tok.decode(item["labels"], skip_special_tokens=False))
