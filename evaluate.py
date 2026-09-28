"""Configs A (BM25), B (semantic), C (hybrid+rerank) — Precision@5, Recall@5, MRR, nDCG@5, latency (review sec. 6)."""
import time, numpy as np
from recommender import Recommender
Q = [("TMT reinforcement bars Fe 500 for RCC building", {"IS 1786": 2, "IS 456": 1}),
     ("PVC insulated copper wiring cable for house wiring", {"IS 694": 2, "IS 1554 (Part 1)": 1}),
     ("laptops printers and servers for office", {"IS 13252 (Part 1)": 2}),
     ("HDPE pipe for drinking water supply", {"IS 4984": 2}),
     ("fly ash based Portland pozzolana cement", {"IS 1489 (Part 1)": 2}),
     ("53 grade cement for high strength precast concrete", {"IS 12269": 2, "IS 456": 1}),
     ("structural steel angles and channels for fabrication", {"IS 2062": 2}),
     ("hollow concrete blocks for masonry wall", {"IS 2185 (Part 1)": 2}),
     ("common burnt clay bricks for wall", {"IS 1077": 2}),
     ("plug and socket outlet 16 A household", {"IS 1293": 2}),
     ("sand and coarse aggregate for concrete", {"IS 383": 2}),
     ("slag cement for marine works", {"IS 455": 2})]
CFG = {"A: BM25": dict(mode="bm25", rerank=False), "B: Semantic": dict(mode="semantic", rerank=False),
       "C: Hybrid+Rerank": dict(mode="hybrid", rerank=True)}
K = 5
def ndcg(ids, rel):
    dcg = sum(rel.get(x, 0) / np.log2(i + 2) for i, x in enumerate(ids))
    ideal = sum(g / np.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:K]))
    return dcg / ideal if ideal else 0
if __name__ == "__main__":
    r = Recommender()
    print(f"{'Config':18}{'P@5':>7}{'R@5':>7}{'MRR':>7}{'nDCG@5':>8}{'ms/q':>8}")
    for name, cfg in CFG.items():
        P, R, M, N, T = [], [], [], [], []
        for q, rel in Q:
            t = time.perf_counter(); res, _ = r.search(q, K, **cfg); T.append((time.perf_counter() - t) * 1000)
            ids = [x["is_number"] for x in res]; hits = [x for x in ids if rel.get(x, 0) > 0]
            P.append(len(hits) / K); R.append(len(hits) / len(rel))
            M.append(next((1 / (i + 1) for i, x in enumerate(ids) if rel.get(x, 0) > 0), 0)); N.append(ndcg(ids, rel))
        print(f"{name:18}{np.mean(P):7.3f}{np.mean(R):7.3f}{np.mean(M):7.3f}{np.mean(N):8.3f}{np.mean(T):8.1f}")
