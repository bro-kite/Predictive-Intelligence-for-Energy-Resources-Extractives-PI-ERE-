"""Text embedding module using sentence transformers.

Provides text embeddings for news headlines, event descriptions, and other
textual data sources using pre-trained sentence transformer models.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from pi_ere.utils.config import config


class TextEmbedder:
    """Text embedder using sentence transformers.

    Wraps Sentence-BERT models for encoding text data into dense embeddings.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """Initialize text embedder.

        Args:
            model_name: Name of sentence-transformer model (None = use config)
            device: Device to use ('cpu', 'cuda', etc.)
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name or config.get(
            'embedding.text_embedding.model_name',
            'sentence-transformers/all-MiniLM-L6-v2'
        )

        self.device = device or config.device
        self.normalize = normalize

        # Load model
        logger.info(f"Loading sentence transformer: {self.model_name}")

        self.model = SentenceTransformer(self.model_name, device=self.device)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Initialized TextEmbedder "
            f"(model={self.model_name}, dim={self.embedding_dim})"
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        # Encode using sentence transformer
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Encode texts from a DataFrame column.

        Args:
            df: DataFrame containing text
            text_column: Name of text column to encode
            batch_size: Batch size for encoding

        Returns:
            DataFrame with added embedding column
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        logger.info(f"Encoding {len(df)} texts from column '{text_column}'")

        # Get texts (handle NaN)
        texts = df[text_column].fillna('').tolist()

        # Encode
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress=len(texts) > 1000,
        )

        # Add to DataFrame
        df['text_embedding'] = list(embeddings)

        logger.info(f"Encoded {len(embeddings)} texts")

        return df

    def aggregate_embeddings_by_date(
        self,
        df: pd.DataFrame,
        text_column: str,
        date_column: str = 'date',
        region_column: Optional[str] = None,
        aggregation: str = 'mean',
    ) -> pd.DataFrame:
        """Aggregate text embeddings by date (and optionally region).

        Useful for creating daily/weekly embeddings from multiple news articles.

        Args:
            df: DataFrame with text and date columns
            text_column: Name of text column
            date_column: Name of date column
            region_column: Optional region column for grouping
            aggregation: Aggregation method ('mean', 'sum', 'max')

        Returns:
            DataFrame with aggregated embeddings
        """
        logger.info(
            f"Aggregating text embeddings by {date_column}"
            f"{' and ' + region_column if region_column else ''}"
        )

        # Encode texts if not already encoded
        if 'text_embedding' not in df.columns:
            df = self.encode_dataframe(df, text_column)

        # Group by date (and region if specified)
        group_cols = [date_column]
        if region_column:
            group_cols.append(region_column)

        # Convert embeddings to array for aggregation
        embeddings_array = np.stack(df['text_embedding'].values)

        # Group and aggregate
        df['embedding_idx'] = range(len(df))

        grouped = df.groupby(group_cols)['embedding_idx'].apply(list).reset_index()

        def aggregate_embeddings(indices):
            group_embeddings = embeddings_array[indices]

            if aggregation == 'mean':
                return np.mean(group_embeddings, axis=0)
            elif aggregation == 'sum':
                return np.sum(group_embeddings, axis=0)
            elif aggregation == 'max':
                return np.max(group_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

        grouped['aggregated_embedding'] = grouped['embedding_idx'].apply(
            aggregate_embeddings
        )

        # Add metadata
        grouped['num_texts'] = grouped['embedding_idx'].apply(len)

        # Drop temporary column
        grouped = grouped.drop('embedding_idx', axis=1)

        logger.info(
            f"Aggregated to {len(grouped)} time periods "
            f"(avg {grouped['num_texts'].mean():.1f} texts per period)"
        )

        return grouped

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = 'cosine',
    ) -> np.ndarray:
        """Compute similarity between embeddings.

        Args:
            embeddings1: First set of embeddings (n1, dim)
            embeddings2: Second set of embeddings (n2, dim)
            metric: Similarity metric ('cosine', 'dot', 'euclidean')

        Returns:
            Similarity matrix (n1, n2)
        """
        if metric == 'cosine':
            # Normalize if not already
            if not self.normalize:
                embeddings1 = embeddings1 / (
                    np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8
                )
                embeddings2 = embeddings2 / (
                    np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8
                )

            # Cosine similarity = dot product of normalized vectors
            similarity = np.dot(embeddings1, embeddings2.T)

        elif metric == 'dot':
            similarity = np.dot(embeddings1, embeddings2.T)

        elif metric == 'euclidean':
            # Negative euclidean distance (so higher = more similar)
            # Using broadcasting to compute pairwise distances
            distances = np.linalg.norm(
                embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :],
                axis=2
            )
            similarity = -distances

        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

        return similarity

    def find_similar(
        self,
        query: Union[str, np.ndarray],
        candidates: Union[List[str], np.ndarray],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Find most similar texts/embeddings to a query.

        Args:
            query: Query text or embedding
            candidates: List of candidate texts or embeddings
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        # Encode query if it's text
        if isinstance(query, str):
            query_emb = self.encode([query])[0]
        else:
            query_emb = query

        # Encode candidates if they're text
        if isinstance(candidates[0], str):
            candidate_embs = self.encode(candidates)
        else:
            candidate_embs = np.array(candidates)

        # Compute similarities
        similarities = self.similarity(
            query_emb[np.newaxis, :],
            candidate_embs,
        )[0]

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        results = list(zip(top_indices.tolist(), top_scores.tolist()))

        return results

    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Perform semantic search on a text corpus.

        Args:
            query: Search query
            corpus: List of texts to search
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            DataFrame with search results
        """
        logger.info(f"Semantic search: '{query}' in {len(corpus)} documents")

        # Find similar texts
        results = self.find_similar(query, corpus, top_k=top_k)

        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                'rank': i + 1,
                'index': idx,
                'text': corpus[idx],
                'similarity': score,
            }
            for i, (idx, score) in enumerate(results)
        ])

        # Filter by threshold if specified
        if score_threshold is not None:
            results_df = results_df[results_df['similarity'] >= score_threshold]

        logger.info(f"Found {len(results_df)} results")

        return results_df

    def batch_encode_and_save(
        self,
        texts: List[str],
        output_path: Union[str, Path],
        batch_size: int = 32,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """Encode large batch of texts and save to disk.

        Args:
            texts: List of texts to encode
            output_path: Path to save embeddings
            batch_size: Batch size for encoding
            metadata: Optional metadata DataFrame to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Encoding and saving {len(texts)} texts to {output_path}")

        # Encode
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            show_progress=True,
        )

        # Save
        data_to_save = {
            'embeddings': embeddings,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
        }

        if metadata is not None:
            data_to_save['metadata'] = metadata

        np.savez_compressed(output_path, **data_to_save)

        logger.info(f"Saved embeddings to {output_path}")

    @staticmethod
    def load_embeddings(path: Union[str, Path]) -> dict:
        """Load saved embeddings.

        Args:
            path: Path to embeddings file

        Returns:
            Dictionary with embeddings and metadata
        """
        data = np.load(path, allow_pickle=True)

        result = {
            'embeddings': data['embeddings'],
            'model_name': str(data['model_name']),
            'embedding_dim': int(data['embedding_dim']),
        }

        if 'metadata' in data:
            result['metadata'] = data['metadata'].item()

        logger.info(
            f"Loaded {len(result['embeddings'])} embeddings "
            f"(dim={result['embedding_dim']})"
        )

        return result
