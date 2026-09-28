"""Pipeline per literature review: extract -> BM25 + dense -> RRF -> rerank -> KG + version check -> RAG explanation."""
import os, re, pandas as pd, numpy as np, networkx as nx
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from extract import extract_requirements

tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
norm = lambda s: re.sub(r"\s", "", s.lower())

class Recommender:
    def __init__(self, csv="standards.csv", use_sbert=False):
        self.df = pd.read_csv(csv).fillna("")
        self.docs = (self.df.title + " " + self.df.scope + " " + self.df.keywords).tolist()
        self.vocab = {w for k in self.df.keywords for w in tok(k)}
        self.bm25 = BM25Okapi([tok(d) for d in self.docs])
        self.sbert = None
        if use_sbert:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
            self.emb = self.sbert.encode(self.docs, normalize_embeddings=True)
        else:
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.mat = self.tfidf.fit_transform(self.docs)
        self.idx = {r.is_number: i for i, r in self.df.iterrows()}
        self._build_kg()

    def _build_kg(self):  # relations: references / supersedes / amendment-of / applicable-to
        G = nx.MultiDiGraph()
        for _, r in self.df.iterrows():
            G.add_node(r.is_number, kind="standard", status=r.status)
            for x in filter(None, r.related.split(";")): G.add_edge(r.is_number, x, rel="references")
            for x in filter(None, r.supersedes.split(";")): G.add_edge(r.is_number, x, rel="supersedes")
            for x in filter(None, r.amendments.split(";")): G.add_edge(x, r.is_number, rel="amendment-of")
            for w in set(tok(r.keywords)): G.add_edge(w, r.is_number, rel="applicable-to")
        self.G = G

    def neighbours(self, no):
        return [(v, d["rel"]) for _, v, d in self.G.out_edges(no, data=True)] if no in self.G else []

    def version_note(self, no):  # amendment / supersession checking
        for u, _, d in self.G.in_edges(no, data=True):
            if d["rel"] == "supersedes": return f"Superseded by {u}"
        r = self.df.iloc[self.idx[no]]
        if r.status != "Active": return f"Status: {r.status}"
        return "Current" + (f"; amendments: {r.amendments}" if r.amendments else "")

    def _dense(self, q):
        if self.sbert: return self.emb @ self.sbert.encode([q], normalize_embeddings=True)[0]
        return cosine_similarity(self.tfidf.transform([q]), self.mat)[0]

    def search(self, query, k=5, mode="hybrid", rerank=True):
        req = extract_requirements(query, self.vocab)
        lex, sem = self.bm25.get_scores(tok(query)), self._dense(query)
        rk = lambda s: {i: r for r, i in enumerate(np.argsort(-s))}
        rl, rs = rk(lex), rk(sem)
        n = len(self.df)
        if mode == "bm25": s = lex.copy()
        elif mode == "semantic": s = sem.copy()
        else: s = np.array([1/(60+rl[i]) + 1/(60+rs[i]) for i in range(n)])
        s = s / (s.max() or 1)
        if rerank:  # domain signals: product type, technical parameters, version status
            for i in np.argsort(-s)[:15]:
                r = self.df.iloc[i]; txt = norm(self.docs[i])
                prod = set(req["products"]); dk = set(tok(r.keywords))
                s[i] += 0.3 * (len(prod & dk) / len(prod) if prod else 0)
                ps = [p for v in req["parameters"].values() for p in v]
                s[i] += 0.3 * (sum(p in txt for p in ps) / len(ps) if ps else 0)
                if self.version_note(r.is_number) != "Current" and not self.version_note(r.is_number).startswith("Current"): s[i] *= 0.5
        out = []
        for i in np.argsort(-s)[:k]:
            r = self.df.iloc[i]
            hit = sorted(set(tok(query)) & set(tok(self.docs[i])))
            out.append({"is_number": r.is_number, "title": r.title, "year": r.year, "classification": r.classification,
                        "score": float(s[i]), "version": self.version_note(r.is_number),
                        "graph": self.neighbours(r.is_number), "evidence": r.scope, "matched_terms": hit})
        for o in out: o["explanation"] = self.explain(query, req, o)
        return out, req

    def explain(self, query, req, hit):  # RAG: retrieved evidence is the only source of truth
        base = (f"{hit['is_number']}:{hit['year']} ({hit['title']}) is recommended because the requirement matches "
                f"its scope — \"{hit['evidence']}\" — on: {', '.join(hit['matched_terms'][:6]) or 'semantic similarity'}. "
                f"Version: {hit['version']}.")
        if not os.getenv("ANTHROPIC_API_KEY"): return base
        try:
            import anthropic
            m = anthropic.Anthropic().messages.create(model="claude-sonnet-5", max_tokens=200, messages=[{"role": "user",
                "content": f"Requirement: {query}\nEvidence (only source allowed): {hit['title']} — {hit['evidence']}\n"
                           f"Explain in 2 sentences why this Indian Standard applies. Do not add facts beyond the evidence."}])
            return m.content[0].text
        except Exception: return base
