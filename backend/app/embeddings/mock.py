"""Mock embedding provider for development and testing."""

import re
from dataclasses import dataclass

from Levenshtein import distance as levenshtein_distance

from .base import EmbeddingProvider, SearchResult, SenseInfo, WordResult

# Cluster color palette
CLUSTER_COLORS = [
    "#6366f1",  # indigo
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
]


@dataclass
class MockNeighbor:
    """Mock neighbor data."""

    word: str
    similarity: float
    frequency: str
    cluster: int
    offset_x: float
    offset_y: float


@dataclass
class MockCluster:
    """Mock cluster definition."""

    label: str
    centroid_x: float
    centroid_y: float


@dataclass
class MockWordData:
    """Mock data for a word including senses and neighbors."""

    senses: list[SenseInfo]
    neighbors_by_sense: dict[str, list[MockNeighbor]]
    clusters_by_sense: dict[str, list[MockCluster]]


# Mock vocabulary with pre-computed relationships
MOCK_VOCABULARY: dict[str, MockWordData] = {
    "happy": MockWordData(
        senses=[SenseInfo("happy|ADJ", "happy (feeling joy)", 100)],
        neighbors_by_sense={
            "happy|ADJ": [
                MockNeighbor("joyful", 0.92, "common", 0, 0.05, 0.03),
                MockNeighbor("cheerful", 0.89, "common", 0, -0.03, 0.06),
                MockNeighbor("delighted", 0.87, "common", 0, 0.08, -0.02),
                MockNeighbor("elated", 0.84, "uncommon", 0, -0.06, 0.04),
                MockNeighbor("jubilant", 0.81, "rare", 0, 0.02, 0.08),
                MockNeighbor("blissful", 0.79, "uncommon", 0, -0.04, -0.05),
                MockNeighbor("ecstatic", 0.77, "uncommon", 0, 0.07, 0.01),
                MockNeighbor("content", 0.85, "common", 1, 0.04, -0.03),
                MockNeighbor("satisfied", 0.82, "common", 1, -0.02, 0.05),
                MockNeighbor("pleased", 0.80, "common", 1, 0.06, 0.02),
                MockNeighbor("gratified", 0.75, "uncommon", 1, -0.05, -0.04),
                MockNeighbor("fulfilled", 0.73, "uncommon", 1, 0.03, 0.06),
                MockNeighbor("lucky", 0.71, "common", 2, -0.02, 0.04),
                MockNeighbor("fortunate", 0.69, "common", 2, 0.05, -0.03),
                MockNeighbor("blessed", 0.67, "common", 2, -0.04, 0.02),
                MockNeighbor("serendipitous", 0.58, "rare", 2, 0.07, 0.05),
            ]
        },
        clusters_by_sense={
            "happy|ADJ": [
                MockCluster("joy & delight", 0.3, 0.7),
                MockCluster("satisfaction", 0.7, 0.6),
                MockCluster("fortunate", 0.5, 0.2),
            ]
        },
    ),
    "bank": MockWordData(
        senses=[
            SenseInfo("bank|NOUN:financial", "bank (financial institution)", 80),
            SenseInfo("bank|NOUN:river", "bank (edge of river)", 20),
        ],
        neighbors_by_sense={
            "bank|NOUN:financial": [
                MockNeighbor("institution", 0.88, "common", 0, 0.04, 0.02),
                MockNeighbor("lender", 0.85, "common", 0, -0.03, 0.05),
                MockNeighbor("treasury", 0.82, "uncommon", 0, 0.06, -0.02),
                MockNeighbor("vault", 0.78, "common", 0, -0.05, 0.03),
                MockNeighbor("money", 0.91, "common", 1, 0.02, 0.04),
                MockNeighbor("savings", 0.87, "common", 1, -0.04, 0.01),
                MockNeighbor("loan", 0.84, "common", 1, 0.05, -0.03),
                MockNeighbor("mortgage", 0.80, "common", 1, -0.02, 0.06),
                MockNeighbor("credit", 0.79, "common", 1, 0.03, 0.02),
                MockNeighbor("deposit", 0.86, "common", 2, -0.03, 0.04),
                MockNeighbor("withdrawal", 0.83, "common", 2, 0.04, -0.02),
                MockNeighbor("transfer", 0.81, "common", 2, -0.05, 0.01),
                MockNeighbor("transaction", 0.78, "common", 2, 0.02, 0.05),
            ],
            "bank|NOUN:river": [
                MockNeighbor("shore", 0.93, "common", 0, 0.03, 0.04),
                MockNeighbor("riverbank", 0.91, "common", 0, -0.02, 0.02),
                MockNeighbor("waterside", 0.85, "uncommon", 0, 0.05, -0.03),
                MockNeighbor("embankment", 0.82, "uncommon", 0, -0.04, 0.05),
                MockNeighbor("levee", 0.78, "uncommon", 0, 0.02, 0.01),
                MockNeighbor("slope", 0.75, "common", 1, -0.03, 0.03),
                MockNeighbor("hillside", 0.72, "common", 1, 0.04, -0.02),
                MockNeighbor("edge", 0.70, "common", 1, -0.05, 0.04),
                MockNeighbor("verge", 0.68, "uncommon", 1, 0.01, 0.06),
            ],
        },
        clusters_by_sense={
            "bank|NOUN:financial": [
                MockCluster("institutions", 0.2, 0.8),
                MockCluster("money & finance", 0.6, 0.7),
                MockCluster("transactions", 0.8, 0.3),
            ],
            "bank|NOUN:river": [
                MockCluster("water features", 0.3, 0.6),
                MockCluster("landscape", 0.7, 0.4),
            ],
        },
    ),
    "run": MockWordData(
        senses=[
            SenseInfo("run|VERB:move", "run (move quickly)", 70),
            SenseInfo("run|VERB:operate", "run (operate/manage)", 30),
        ],
        neighbors_by_sense={
            "run|VERB:move": [
                MockNeighbor("sprint", 0.91, "common", 0, 0.04, 0.02),
                MockNeighbor("jog", 0.88, "common", 0, -0.02, 0.05),
                MockNeighbor("dash", 0.86, "common", 0, 0.05, -0.03),
                MockNeighbor("race", 0.84, "common", 0, -0.04, 0.01),
                MockNeighbor("gallop", 0.79, "uncommon", 0, 0.02, 0.06),
                MockNeighbor("fast", 0.85, "common", 1, -0.03, 0.03),
                MockNeighbor("swift", 0.82, "common", 1, 0.04, -0.02),
                MockNeighbor("quick", 0.80, "common", 1, -0.05, 0.04),
                MockNeighbor("hasty", 0.75, "uncommon", 1, 0.01, 0.05),
                MockNeighbor("flee", 0.78, "common", 2, -0.02, 0.02),
                MockNeighbor("escape", 0.76, "common", 2, 0.05, -0.04),
                MockNeighbor("bolt", 0.73, "common", 2, -0.04, 0.03),
                MockNeighbor("abscond", 0.65, "rare", 2, 0.03, 0.06),
            ],
            "run|VERB:operate": [
                MockNeighbor("manage", 0.89, "common", 0, 0.03, 0.04),
                MockNeighbor("operate", 0.87, "common", 0, -0.02, 0.02),
                MockNeighbor("direct", 0.84, "common", 0, 0.05, -0.03),
                MockNeighbor("oversee", 0.81, "uncommon", 0, -0.04, 0.05),
                MockNeighbor("administer", 0.78, "uncommon", 0, 0.02, 0.01),
                MockNeighbor("function", 0.85, "common", 1, -0.03, 0.03),
                MockNeighbor("work", 0.83, "common", 1, 0.04, -0.02),
                MockNeighbor("execute", 0.80, "common", 1, -0.05, 0.04),
                MockNeighbor("perform", 0.77, "common", 1, 0.01, 0.06),
            ],
        },
        clusters_by_sense={
            "run|VERB:move": [
                MockCluster("locomotion", 0.2, 0.7),
                MockCluster("speed", 0.6, 0.8),
                MockCluster("escape", 0.8, 0.4),
            ],
            "run|VERB:operate": [
                MockCluster("manage", 0.3, 0.6),
                MockCluster("function", 0.7, 0.5),
            ],
        },
    ),
    "ocean": MockWordData(
        senses=[SenseInfo("ocean|NOUN", "ocean (large body of water)", 100)],
        neighbors_by_sense={
            "ocean|NOUN": [
                MockNeighbor("sea", 0.95, "common", 0, 0.03, 0.02),
                MockNeighbor("atlantic", 0.88, "common", 0, -0.02, 0.05),
                MockNeighbor("pacific", 0.87, "common", 0, 0.05, -0.01),
                MockNeighbor("waters", 0.84, "common", 0, -0.04, 0.03),
                MockNeighbor("deep", 0.79, "common", 0, 0.02, 0.06),
                MockNeighbor("tide", 0.73, "common", 0, -0.05, -0.02),
                MockNeighbor("wave", 0.78, "common", 0, 0.01, 0.04),
                MockNeighbor("whale", 0.75, "common", 1, -0.03, 0.02),
                MockNeighbor("dolphin", 0.72, "common", 1, 0.04, -0.03),
                MockNeighbor("coral", 0.69, "common", 1, -0.05, 0.04),
                MockNeighbor("fish", 0.68, "common", 1, 0.02, 0.01),
                MockNeighbor("marine", 0.76, "common", 1, -0.01, 0.05),
                MockNeighbor("salt", 0.52, "common", 1, 0.06, -0.04),
                MockNeighbor("vast", 0.71, "common", 2, 0.03, -0.02),
                MockNeighbor("endless", 0.67, "common", 2, -0.04, 0.03),
                MockNeighbor("immense", 0.65, "uncommon", 2, 0.05, 0.04),
                MockNeighbor("fathomless", 0.58, "rare", 2, -0.02, 0.06),
                MockNeighbor("voyage", 0.62, "uncommon", 3, 0.04, -0.01),
                MockNeighbor("sail", 0.60, "common", 3, -0.03, 0.02),
                MockNeighbor("horizon", 0.55, "common", 3, 0.02, 0.05),
            ]
        },
        clusters_by_sense={
            "ocean|NOUN": [
                MockCluster("water bodies", 0.2, 0.7),
                MockCluster("marine life", 0.5, 0.5),
                MockCluster("vastness", 0.8, 0.6),
                MockCluster("journey", 0.6, 0.2),
            ]
        },
    ),
    "bright": MockWordData(
        senses=[SenseInfo("bright|ADJ", "bright (emitting light / intelligent)", 100)],
        neighbors_by_sense={
            "bright|ADJ": [
                MockNeighbor("luminous", 0.90, "uncommon", 0, 0.03, 0.04),
                MockNeighbor("radiant", 0.88, "uncommon", 0, -0.02, 0.02),
                MockNeighbor("brilliant", 0.86, "common", 0, 0.05, -0.03),
                MockNeighbor("shining", 0.84, "common", 0, -0.04, 0.05),
                MockNeighbor("gleaming", 0.81, "uncommon", 0, 0.02, 0.01),
                MockNeighbor("incandescent", 0.75, "rare", 0, -0.03, 0.06),
                MockNeighbor("clever", 0.82, "common", 1, 0.04, -0.02),
                MockNeighbor("intelligent", 0.80, "common", 1, -0.05, 0.03),
                MockNeighbor("smart", 0.79, "common", 1, 0.01, 0.05),
                MockNeighbor("sharp", 0.76, "common", 1, -0.02, -0.04),
                MockNeighbor("astute", 0.72, "uncommon", 1, 0.05, 0.02),
                MockNeighbor("vivid", 0.85, "common", 2, -0.03, 0.04),
                MockNeighbor("vibrant", 0.83, "common", 2, 0.04, -0.01),
                MockNeighbor("colorful", 0.78, "common", 2, -0.05, 0.02),
                MockNeighbor("striking", 0.74, "common", 2, 0.02, 0.06),
            ]
        },
        clusters_by_sense={
            "bright|ADJ": [
                MockCluster("luminous", 0.3, 0.7),
                MockCluster("intelligent", 0.7, 0.6),
                MockCluster("vivid", 0.5, 0.3),
            ]
        },
    ),
    "light": MockWordData(
        senses=[
            SenseInfo("light|NOUN", "light (illumination)", 60),
            SenseInfo("light|ADJ", "light (not heavy / bright)", 40),
        ],
        neighbors_by_sense={
            "light|NOUN": [
                MockNeighbor("illumination", 0.91, "uncommon", 0, 0.03, 0.02),
                MockNeighbor("brightness", 0.89, "common", 0, -0.02, 0.05),
                MockNeighbor("glow", 0.86, "common", 0, 0.05, -0.03),
                MockNeighbor("radiance", 0.83, "uncommon", 0, -0.04, 0.01),
                MockNeighbor("beam", 0.80, "common", 0, 0.02, 0.06),
                MockNeighbor("photon", 0.75, "uncommon", 1, -0.03, 0.03),
                MockNeighbor("wavelength", 0.72, "uncommon", 1, 0.04, -0.02),
                MockNeighbor("spectrum", 0.70, "uncommon", 1, -0.05, 0.04),
                MockNeighbor("ray", 0.78, "common", 1, 0.01, 0.05),
                MockNeighbor("vision", 0.68, "common", 2, -0.02, 0.02),
                MockNeighbor("sight", 0.66, "common", 2, 0.05, -0.04),
                MockNeighbor("clarity", 0.64, "common", 2, -0.04, 0.03),
            ],
            "light|ADJ": [
                MockNeighbor("lightweight", 0.92, "common", 0, 0.03, 0.04),
                MockNeighbor("featherweight", 0.85, "uncommon", 0, -0.02, 0.02),
                MockNeighbor("airy", 0.82, "common", 0, 0.05, -0.03),
                MockNeighbor("weightless", 0.80, "uncommon", 0, -0.04, 0.05),
                MockNeighbor("bright", 0.88, "common", 1, 0.02, 0.01),
                MockNeighbor("pale", 0.79, "common", 1, -0.03, 0.03),
                MockNeighbor("fair", 0.75, "common", 1, 0.04, -0.02),
                MockNeighbor("easy", 0.73, "common", 2, -0.05, 0.04),
                MockNeighbor("simple", 0.70, "common", 2, 0.01, 0.06),
                MockNeighbor("gentle", 0.68, "common", 2, -0.02, -0.03),
            ],
        },
        clusters_by_sense={
            "light|NOUN": [
                MockCluster("illumination", 0.3, 0.7),
                MockCluster("physics", 0.7, 0.6),
                MockCluster("perception", 0.5, 0.3),
            ],
            "light|ADJ": [
                MockCluster("weight", 0.3, 0.7),
                MockCluster("brightness", 0.7, 0.5),
                MockCluster("ease", 0.5, 0.2),
            ],
        },
    ),
}

# Simple typo correction dictionary
TYPO_CORRECTIONS = {
    "happyness": "happiness",
    "hapiness": "happiness",
    "happines": "happiness",
    "hapy": "happy",
    "ocaen": "ocean",
    "oceam": "ocean",
    "bankk": "bank",
    "runn": "run",
    "runing": "running",
    "brigth": "bright",
    "brite": "bright",
    "lite": "light",
    "ligth": "light",
}


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider using pre-defined vocabulary."""

    def __init__(self):
        self._vocabulary = MOCK_VOCABULARY
        self._typo_corrections = TYPO_CORRECTIONS

    def _normalize(self, word: str) -> str:
        """Normalize a word: lowercase, strip whitespace, remove non-alpha."""
        return re.sub(r"[^a-z]", "", word.lower().strip())

    def search(
        self,
        word: str,
        sense: str | None = None,
        limit: int = 100,
    ) -> SearchResult | None:
        normalized = self._normalize(word)

        if not normalized or normalized not in self._vocabulary:
            return None

        word_data = self._vocabulary[normalized]

        # Determine which sense to use
        if sense and sense in word_data.neighbors_by_sense:
            selected_sense = sense
        else:
            # Default to most frequent sense
            selected_sense = max(word_data.senses, key=lambda s: s.frequency).sense

        neighbors_data = word_data.neighbors_by_sense.get(selected_sense, [])
        clusters_data = word_data.clusters_by_sense.get(selected_sense, [])

        # Build clusters
        clusters = [
            {
                "id": i,
                "label": c.label,
                "color": CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                "centroid": {"x": c.centroid_x, "y": c.centroid_y},
            }
            for i, c in enumerate(clusters_data)
        ]

        # Build neighbors with coordinates
        neighbors = []

        # Add the query word itself at cluster 0 centroid
        if clusters_data:
            neighbors.append(
                WordResult(
                    word=normalized,
                    similarity=1.0,
                    coordinates=(clusters_data[0].centroid_x, clusters_data[0].centroid_y),
                    frequency="common",
                    cluster=0,
                )
            )

        # Add neighbors
        for n in neighbors_data[:limit]:
            cluster_def = clusters_data[n.cluster] if n.cluster < len(clusters_data) else clusters_data[0]
            neighbors.append(
                WordResult(
                    word=n.word,
                    similarity=n.similarity,
                    coordinates=(
                        cluster_def.centroid_x + n.offset_x,
                        cluster_def.centroid_y + n.offset_y,
                    ),
                    frequency=n.frequency,
                    cluster=n.cluster,
                )
            )

        return SearchResult(
            word=word,
            normalized_word=normalized,
            sense=selected_sense,
            available_senses=word_data.senses,
            neighbors=neighbors,
            clusters=clusters,
        )

    def has_word(self, word: str) -> bool:
        normalized = self._normalize(word)
        return normalized in self._vocabulary

    def find_similar_word(self, word: str) -> str | None:
        normalized = self._normalize(word)

        # Check explicit typo corrections first
        if normalized in self._typo_corrections:
            return self._typo_corrections[normalized]

        # Check vocabulary with Levenshtein distance
        closest = None
        min_distance = float("inf")

        for vocab_word in self._vocabulary.keys():
            dist = levenshtein_distance(normalized, vocab_word)
            if dist <= 2 and dist < min_distance:
                min_distance = dist
                closest = vocab_word

        return closest

    def get_suggestions(self, limit: int = 3) -> list[str]:
        return list(self._vocabulary.keys())[:limit]

    def get_available_words(self) -> list[str]:
        return list(self._vocabulary.keys())
