import os
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import Optional, List
import glob
from collections import defaultdict
import random
# from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
# =========================
# Utility: Parsing
# =========================
def parse_line(line, min_len=5, max_len=50):
    parts = line.strip().split("\t")
    if len(parts) < 2:
        return None

    s1, s2 = parts[0], parts[1]

    if s1.strip() == s2.strip():
        return None

    l1, l2 = len(s1.split()), len(s2.split())
    if not (min_len <= l1 <= max_len and min_len <= l2 <= max_len):
        return None

    return s1, s2


# =========================
# Cache Builder
# =========================
def build_cache(
    input_path: str,
    output_dir: str,
    tokenizer=None,
    chunk_size: int = 100000,
    tokenize: bool = True,
    max_length: int = 128,
):
    os.makedirs(output_dir, exist_ok=True)

    buffer = []
    shard_id = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue

            s1, s2 = parsed

            if tokenize and tokenizer is not None:
                model_input = tokenizer(
                    "paraphrase: " + s1,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )

                labels = tokenizer(
                    s2,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )["input_ids"]

                model_input["labels"] = labels
                buffer.append(model_input)

            else:
                buffer.append({"s1": s1, "s2": s2})

            if len(buffer) >= chunk_size:
                torch.save(buffer, os.path.join(output_dir, f"shard_{shard_id}.pt"))
                buffer = []
                shard_id += 1

    if buffer:
        torch.save(buffer, os.path.join(output_dir, f"shard_{shard_id}.pt"))

    print(f"Cache built at {output_dir}")



def build_paranmt_multi_cache(
    input_path,
    output_dir,
    tokenizer=None,
    tokenize=True,
    model_name="all-MiniLM-L6-v2",
    chunk_size=5000,
    group_size=3,
    min_len=5,
    max_len=50,
    min_sim=0.5,
    max_sim=0.95,
    max_length=128
):
    os.makedirs(output_dir, exist_ok=True)

    # SBERT model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    buffer = []
    shard_id = 0

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(sentences):
        inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)  # important

        return embeddings
    
    def similarity(s1, s2):
        emb = encode([s1, s2])
        return (emb[0] @ emb[1]).item()

    def batch_similarity(sentences1, sentences2):
        emb1 = encode(sentences1)
        emb2 = encode(sentences2)

        return torch.sum(emb1 * emb2, dim=1)  # cosine since normalized

    def process_buffer(buffer, shard_id):
        groups = defaultdict(list)
        results = []

        # Step 1: filtering + bi-directional grouping
        for s1, s2 in buffer:
            if s1.strip() == s2.strip():
                continue

            l1, l2 = len(s1.split()), len(s2.split())
            if not (min_len <= l1 <= max_len and min_len <= l2 <= max_len):
                continue

            groups[s1].append(s2)
            groups[s2].append(s1)

        # Step 2: SBERT embeddings (batch)
        all_sentences = list(groups.keys())
        embeddings = encode(all_sentences)

        emb_map = {s: embeddings[i] for i, s in enumerate(all_sentences)}

        # Step 3: scoring + grouping
        for anchor, paraphrases in groups.items():
            if len(paraphrases) < group_size:
                continue

            anchor_emb = emb_map[anchor]

            scored = []
            for p in paraphrases:
                sim = (anchor_emb @ emb_map[p]).item() #util.cos_sim(anchor_emb, emb_map.get(p, anchor_emb)).item()

                if min_sim <= sim <= max_sim:
                    scored.append((p, sim))

            if len(scored) < group_size:
                continue

            # sort by similarity
            scored.sort(key=lambda x: -x[1])

            # take top-k
            selected = scored[:group_size]

            for p, score in selected:
                if tokenize and tokenizer is not None:
                    inputs = tokenizer(
                        "paraphrase: " + anchor,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length
                    )

                    labels = tokenizer(
                        p,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length
                    )["input_ids"]

                    inputs["labels"] = labels
                    inputs["score"] = score
                    results.append(inputs)

                else:
                    results.append({
                        "s1": anchor,
                        "s2": p,
                        "score": score
                    })

        # save shard
        torch.save(results, os.path.join(output_dir, f"shard_{shard_id}.pt"))
        return shard_id + 1

    # =========================
    # Main loop
    # =========================
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building cache"):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            buffer.append((parts[0], parts[1]))

            if len(buffer) >= chunk_size:
                shard_id = process_buffer(buffer, shard_id)
                buffer = []

    if buffer:
        process_buffer(buffer, shard_id)

    print(f"Cache built at {output_dir}")

# =========================
# Similarity (lightweight)
# =========================
def jaccard_sim(s1, s2):
    w1, w2 = set(s1.split()), set(s2.split())
    if len(w1 | w2) == 0:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# =========================
# Dataset
# =========================
class ParaNMTMultiDataset(IterableDataset):
    def __init__(
        self,
        path,
        tokenizer=None,
        max_length=128,
        tokenize_on_the_fly=True,
        min_len=5,
        max_len=50,
        min_sim=0.3,
        max_sim=0.9,
        group_size=3,   # number of paraphrases per anchor
        buffer_size=10000  # for grouping
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenize_on_the_fly = tokenize_on_the_fly

        self.min_len = min_len
        self.max_len = max_len
        self.min_sim = min_sim
        self.max_sim = max_sim

        self.group_size = group_size
        self.buffer_size = buffer_size

    # =========================
    # Filtering
    # =========================
    def _filter(self, s1, s2):
        if s1.strip() == s2.strip():
            return False

        l1, l2 = len(s1.split()), len(s2.split())
        if not (self.min_len <= l1 <= self.max_len and
                self.min_len <= l2 <= self.max_len):
            return False

        sim = jaccard_sim(s1, s2)
        if not (self.min_sim <= sim <= self.max_sim):
            return False

        return True

    # =========================
    # Tokenization
    # =========================
    def _tokenize_pair(self, s1, s2):
        inputs = self.tokenizer(
            "paraphrase: " + s1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            s2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels["input_ids"].squeeze(0)
        return inputs

    # =========================
    # Iterator
    # =========================
    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        buffer = []
        groups = defaultdict(list)

        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):

                # worker sharding
                if i % num_workers != worker_id:
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                s1, s2 = parts[0], parts[1]

                # filtering
                if not self._filter(s1, s2):
                    continue

                buffer.append((s1, s2))

                # build groups periodically
                if len(buffer) >= self.buffer_size:

                    # group by anchor sentence
                    for a, b in buffer:
                        groups[a].append(b)
                        groups[b].append(a)  # 🔥 bi-directional grouping

                    # yield multi-paraphrase samples
                    for anchor, paraphrases in groups.items():
                        if len(paraphrases) < self.group_size:
                            continue

                        # sample group
                        sampled = random.sample(
                            paraphrases,
                            min(len(paraphrases), self.group_size)
                        )

                        # sort by similarity (score-wise pairing)
                        scored = [
                            (p, jaccard_sim(anchor, p))
                            for p in sampled
                        ]
                        print(scored)
                        scored.sort(key=lambda x: -x[1])

                        for p, score in scored:

                            if self.tokenize_on_the_fly and self.tokenizer:
                                yield self._tokenize_pair(anchor, p)
                            else:
                                yield {
                                    "s1": anchor,
                                    "s2": p,
                                    "score": score
                                }

                    # reset
                    buffer = []
                    groups = defaultdict(list)

        # flush remaining buffer
        for a, b in buffer:
            if self.tokenize_on_the_fly and self.tokenizer:
                yield self._tokenize_pair(a, b)
            else:
                yield {"s1": a, "s2": b}

# =========================
# Streaming Dataset
# =========================
class ParaNMTIterableDataset(IterableDataset):
    def __init__(
        self,
        path: str,
        tokenizer=None,
        max_length: int = 128,
        tokenize_on_the_fly: bool = True,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenize_on_the_fly = tokenize_on_the_fly

    def _tokenize(self, s1, s2):
        inputs = self.tokenizer(
            "paraphrase: " + s1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            s2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels["input_ids"].squeeze(0)
        return inputs

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # 🔥 worker sharding
                if i % num_workers != worker_id:
                    continue

                parsed = parse_line(line)
                if parsed is None:
                    continue

                s1, s2 = parsed

                if self.tokenize_on_the_fly and self.tokenizer is not None:
                    yield self._tokenize(s1, s2)
                else:
                    yield {"s1": s1, "s2": s2}


# =========================
# Cached Dataset (Iterable)
# =========================
class CachedIterableDataset(IterableDataset):
    def __init__(
        self,
        cache_dir: str,
        tokenizer=None,
        max_length: int = 128,
        tokenized: bool = True,
    ):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized = tokenized

    def _tokenize(self, s1, s2):
        inputs = self.tokenizer(
            "paraphrase: " + s1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            s2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels["input_ids"].squeeze(0)
        return inputs

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # shard at file level (efficient)
        for file_idx, file in enumerate(self.files):
            if file_idx % num_workers != worker_id:
                continue

            data = torch.load(file)

            for item in data:
                if self.tokenized:
                    yield item
                else:
                    yield self._tokenize(item["s1"], item["s2"])


class CachedParaNMTMultiDataset(IterableDataset):
    def __init__(
        self,
        cache_dir,
        tokenizer=None,
        tokenized=True,
        max_length=128
    ):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        self.tokenizer = tokenizer
        self.tokenized = tokenized
        self.max_length = max_length

    def _tokenize(self, s1, s2):
        inputs = self.tokenizer(
            "paraphrase: " + s1,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            s2,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = labels["input_ids"].squeeze(0)
        return inputs

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # file-level sharding (efficient)
        for file_idx, file in enumerate(self.files):
            if file_idx % num_workers != worker_id:
                continue

            data = torch.load(file)

            for item in data:
                if self.tokenized:
                    yield item
                else:
                    yield self._tokenize(item["s1"], item["s2"])

if __name__ == "__main__":
    datapath = "./dataset/para-nmt-5m-processed/para-nmt-5m-processed.txt"
    output_dir = "dataset/cache_paranmt"
    dataset = ParaNMTMultiDataset(
        datapath,
        tokenize_on_the_fly=True,
        group_size=5
    )
    print(next(iter(dataset)))