"""Tests for term co-occurrence collection (Stage 6)."""


from sciscape.keyword_extraction.cooccurrence import collect_cooccurrence


def _batch_iter(docs):
    """Yield all docs as a single batch."""
    yield docs


class TestCollectCooccurrence:
    def test_basic_cooccurrence(self):
        docs = [
            "neural network deep learning model",
            "neural network training optimization",
            "deep learning architecture design",
        ]
        terms = ["neural", "network", "deep", "learning", "model"]
        cooc = collect_cooccurrence(_batch_iter(docs), terms)

        assert cooc.shape == (5, 5)
        # "neural" and "network" co-occur in docs 0 and 1
        assert cooc[0, 1] == 2  # neural-network
        assert cooc[1, 0] == 2  # symmetric
        # "deep" and "learning" co-occur in docs 0 and 2
        assert cooc[2, 3] == 2
        # "model" only in doc 0
        assert cooc[4, 0] == 1  # model-neural
        assert cooc[4, 2] == 1  # model-deep

    def test_symmetric(self):
        docs = ["alpha beta gamma", "alpha gamma delta"]
        terms = ["alpha", "beta", "gamma", "delta"]
        cooc = collect_cooccurrence(_batch_iter(docs), terms)

        # Check symmetry
        diff = cooc - cooc.T
        assert diff.nnz == 0

    def test_no_self_cooccurrence(self):
        docs = ["word word word repeated"]
        terms = ["word", "repeated"]
        cooc = collect_cooccurrence(_batch_iter(docs), terms)
        assert cooc[0, 0] == 0  # no self co-occurrence
        assert cooc[1, 1] == 0

    def test_empty_terms(self):
        docs = ["hello world"]
        cooc = collect_cooccurrence(_batch_iter(docs), [])
        assert cooc.shape == (0, 0)

    def test_min_cooc_filter(self):
        docs = [
            "alpha beta gamma",
            "alpha beta delta",
            "alpha gamma epsilon",
        ]
        terms = ["alpha", "beta", "gamma", "delta", "epsilon"]
        cooc = collect_cooccurrence(_batch_iter(docs), terms, min_cooc_count=2)

        # alpha-beta co-occur 2 times -> kept
        assert cooc[0, 1] >= 2
        # alpha-delta co-occur 1 time -> filtered
        assert cooc[0, 3] == 0

    def test_phrase_terms(self):
        docs = [
            "machine learning is a deep learning technique",
            "machine learning and neural networks",
        ]
        terms = ["machine learning", "deep learning", "neural"]
        cooc = collect_cooccurrence(_batch_iter(docs), terms)

        assert cooc.shape == (3, 3)
        # "machine learning" and "deep learning" co-occur in doc 0
        assert cooc[0, 1] >= 1

    def test_multiple_batches(self):
        def two_batches():
            yield ["alpha beta", "alpha gamma"]
            yield ["beta gamma", "alpha beta"]

        terms = ["alpha", "beta", "gamma"]
        cooc = collect_cooccurrence(two_batches(), terms)

        # alpha-beta: docs 0, 3 -> 2 co-occurrences
        # alpha-gamma: doc 1 -> 1
        # beta-gamma: doc 2 -> 1
        assert cooc[0, 1] >= 2
        assert cooc[0, 2] >= 1
