"""FAISS-based similarity search for historical analogues and similar regions."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import pickle
from loguru import logger


@dataclass
class SearchResult:
    """Result from a similarity search."""
    id: str
    score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class Analogue:
    """Historical analogue with similarity information."""
    region: str
    start_date: datetime
    end_date: datetime
    similarity_score: float
    outcome: str  # description of what happened next
    key_features: List[str] = field(default_factory=list)


class SimilaritySearch:
    """FAISS-based similarity search for historical analogues."""

    def __init__(self, embedding_dim: int = 320, index_type: str = 'flat'):
        """Initialize similarity search with FAISS index.

        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat' for exact search)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # Initialize FAISS index for inner product (cosine similarity after normalization)
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Store metadata alongside FAISS index
        self.metadata_df = pd.DataFrame()

        logger.info(f"Initialized SimilaritySearch with dim={embedding_dim}, type={index_type}")

    def add_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata: pd.DataFrame
    ) -> None:
        """Add embeddings to the index with metadata.

        Args:
            ids: List of IDs formatted as "{region}_{date}"
            embeddings: Array of shape (n_samples, embedding_dim)
            metadata: DataFrame with columns including 'region' and 'date'
        """
        if embeddings.shape[0] != len(ids):
            raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) must match number of ids ({len(ids)})")

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) must match index dimension ({self.embedding_dim})")

        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings.astype('float32')
        faiss.normalize_L2(normalized_embeddings)

        # Get current index size for tracking
        start_idx = self.index.ntotal

        # Add to FAISS index
        self.index.add(normalized_embeddings)

        # Prepare metadata DataFrame
        metadata_copy = metadata.copy()
        metadata_copy['id'] = ids
        metadata_copy['embedding_idx'] = range(start_idx, start_idx + len(ids))

        # Ensure required columns exist
        if 'region' not in metadata_copy.columns or 'date' not in metadata_copy.columns:
            raise ValueError("Metadata must contain 'region' and 'date' columns")

        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(metadata_copy['date']):
            metadata_copy['date'] = pd.to_datetime(metadata_copy['date'])

        # Append to existing metadata
        self.metadata_df = pd.concat([self.metadata_df, metadata_copy], ignore_index=True)

        logger.info(f"Added {len(ids)} embeddings to index (total: {self.index.ntotal})")

    def search(self, query: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        """Search for similar embeddings.

        Args:
            query: Query embedding of shape (embedding_dim,) or (1, embedding_dim)
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects sorted by similarity score (descending)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return []

        # Ensure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query
        query_normalized = query.astype('float32')
        faiss.normalize_L2(query_normalized)

        # Search
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_normalized, top_k)

        # Convert to SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            # Get metadata for this result
            meta_row = self.metadata_df[self.metadata_df['embedding_idx'] == idx].iloc[0]
            metadata = meta_row.to_dict()

            results.append(SearchResult(
                id=metadata['id'],
                score=float(score),
                metadata=metadata
            ))

        logger.debug(f"Found {len(results)} results for query")
        return results

    def find_analogues(
        self,
        region: str,
        current_embedding: np.ndarray,
        exclude_recent_days: int = 90
    ) -> List[Analogue]:
        """Find historical analogues for a region.

        Args:
            region: Region code to find analogues for
            current_embedding: Current state embedding
            exclude_recent_days: Exclude data from same region within this many days

        Returns:
            List of Analogue objects sorted by similarity score (descending)
        """
        # Search for similar embeddings
        all_results = self.search(current_embedding, top_k=100)

        # Filter and convert to analogues
        analogues = []
        cutoff_date = datetime.now() - timedelta(days=exclude_recent_days)

        for result in all_results:
            result_region = result.metadata['region']
            result_date = result.metadata['date']

            # Skip same region's recent data
            if result_region == region and result_date > cutoff_date:
                continue

            # Create analogue
            # For start_date and end_date, we use the result date as both
            # In a real scenario, these would represent a time window
            analogue = Analogue(
                region=result_region,
                start_date=result_date,
                end_date=result_date,  # Could be extended to multi-day windows
                similarity_score=result.score,
                outcome=self._generate_outcome(result.metadata),
                key_features=self._extract_key_features(result.metadata)
            )
            analogues.append(analogue)

        logger.info(f"Found {len(analogues)} analogues for region {region}")
        return analogues

    def find_similar_regions(
        self,
        target_region: str,
        target_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Find regions with similar current conditions.

        Args:
            target_region: Region to compare against
            target_embedding: Embedding of target region's current state
            top_k: Number of similar regions to return

        Returns:
            List of SearchResult objects for different regions
        """
        # Search for similar embeddings
        all_results = self.search(target_embedding, top_k=top_k * 3)

        # Filter to get unique regions, excluding target
        seen_regions = {target_region}
        similar_regions = []

        for result in all_results:
            result_region = result.metadata['region']

            # Skip target region and already seen regions
            if result_region in seen_regions:
                continue

            seen_regions.add(result_region)
            similar_regions.append(result)

            if len(similar_regions) >= top_k:
                break

        logger.info(f"Found {len(similar_regions)} similar regions to {target_region}")
        return similar_regions

    def save_index(self, path: Path) -> None:
        """Save FAISS index and metadata to disk.

        Args:
            path: Directory path to save index files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata_df': self.metadata_df,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)

        logger.info(f"Saved index to {path}")

    @classmethod
    def load_index(cls, path: Path) -> 'SimilaritySearch':
        """Load FAISS index and metadata from disk.

        Args:
            path: Directory path containing index files

        Returns:
            SimilaritySearch instance with loaded index
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)

        # Create instance
        instance = cls(
            embedding_dim=data['embedding_dim'],
            index_type=data['index_type']
        )

        # Load FAISS index
        index_path = path / "faiss_index.bin"
        instance.index = faiss.read_index(str(index_path))

        # Restore metadata
        instance.metadata_df = data['metadata_df']

        logger.info(f"Loaded index from {path} (total: {instance.index.ntotal})")
        return instance

    def _generate_outcome(self, metadata: Dict) -> str:
        """Generate outcome description from metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            Outcome description string
        """
        # Extract relevant fields for outcome
        region = metadata.get('region', 'Unknown')
        date = metadata.get('date', 'Unknown')

        # In a real scenario, this would look ahead to see what happened next
        outcome = f"Historical pattern from {region} on {date}"

        return outcome

    def _extract_key_features(self, metadata: Dict) -> List[str]:
        """Extract key features from metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            List of key feature descriptions
        """
        features = []

        # Extract notable metadata fields
        for key, value in metadata.items():
            if key in ['id', 'embedding_idx', 'region', 'date']:
                continue

            # Add as key feature if it's a notable value
            if isinstance(value, (int, float)) and not pd.isna(value):
                features.append(f"{key}: {value}")

        return features[:5]  # Limit to top 5 features
