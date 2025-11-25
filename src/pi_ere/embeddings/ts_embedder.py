"""Time-series embedding module using contrastive learning.

Implements a TS2Vec-inspired architecture for learning robust time-series representations
through self-supervised contrastive learning.

Reference: TS2Vec (Yue et al., 2022) - A Universal Time Series Representation Learning Framework
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from pi_ere.utils.config import config


class DilatedConvEncoder(nn.Module):
    """Dilated convolutional encoder for time-series.

    Uses dilated convolutions to capture patterns at multiple temporal scales.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        depth: int = 10,
        kernel_size: int = 3,
    ):
        """Initialize dilated convolution encoder.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            depth: Number of dilated conv layers
            kernel_size: Convolution kernel size
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Input projection
        self.input_projection = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=1
        )

        # Dilated convolution layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                padding=(kernel_size - 1) * (2 ** i) // 2
            )
            for i in range(depth)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(depth)
        ])

        # Activation
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, time, features)

        Returns:
            Encoded tensor of shape (batch, time, hidden_dim)
        """
        # Transpose to (batch, features, time) for Conv1d
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_projection(x)

        # Apply dilated convolutions with residual connections
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x

            # Conv -> transpose -> norm -> transpose -> activation
            x = conv(x)
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            x = self.activation(x)

            # Residual connection
            x = x + residual

        # Transpose back to (batch, time, hidden_dim)
        x = x.transpose(1, 2)

        return x


class TimeSeriesDataset(Dataset):
    """Dataset for time-series data."""

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 30,
        stride: int = 1,
    ):
        """Initialize dataset.

        Args:
            data: Time-series data of shape (n_series, time_steps, features)
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride

        # Generate window indices
        self.windows = self._create_windows()

    def _create_windows(self) -> List[Tuple[int, int]]:
        """Create sliding windows.

        Returns:
            List of (series_idx, start_idx) tuples
        """
        windows = []

        for series_idx in range(len(self.data)):
            n_steps = self.data[series_idx].shape[0]

            for start_idx in range(0, n_steps - self.window_size + 1, self.stride):
                windows.append((series_idx, start_idx))

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a time-series window.

        Args:
            idx: Index

        Returns:
            Time-series window tensor
        """
        series_idx, start_idx = self.windows[idx]
        window = self.data[series_idx][start_idx:start_idx + self.window_size]

        return torch.FloatTensor(window)


class TimeSeriesEmbedder(nn.Module):
    """Time-series embedder using contrastive learning.

    Learns representations through temporal and instance-level contrastive learning.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 320,
        hidden_dim: int = 64,
        depth: int = 10,
        temperature: float = 0.2,
    ):
        """Initialize time-series embedder.

        Args:
            input_dim: Number of input features
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden dimension for encoder
            depth: Depth of dilated conv encoder
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # Encoder
        self.encoder = DilatedConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        logger.info(
            f"Initialized TimeSeriesEmbedder "
            f"(input_dim={input_dim}, embedding_dim={embedding_dim})"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input time-series (batch, time, features)
            mask: Optional mask for missing values (batch, time)

        Returns:
            Embeddings (batch, embedding_dim)
        """
        # Encode
        encoded = self.encoder(x)  # (batch, time, hidden_dim)

        # Global pooling (temporal aggregation)
        if mask is not None:
            # Masked average pooling
            mask = mask.unsqueeze(-1)  # (batch, time, 1)
            pooled = (encoded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            # Simple average pooling
            pooled = encoded.mean(dim=1)  # (batch, hidden_dim)

        # Project to embedding space
        embedding = self.projection(pooled)  # (batch, embedding_dim)

        return embedding

    def encode_batch(
        self,
        data: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
        device: str = 'cpu',
    ) -> np.ndarray:
        """Encode a batch of time-series.

        Args:
            data: Time-series data (n_samples, time, features)
            batch_size: Batch size for encoding
            device: Device to use

        Returns:
            Embeddings (n_samples, embedding_dim)
        """
        self.eval()
        self.to(device)

        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)

        embeddings = []

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(device)
                emb = self.forward(batch)
                embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def hierarchical_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hierarchical contrastive loss.

        Args:
            z1: Embeddings from augmented view 1 (batch, embedding_dim)
            z2: Embeddings from augmented view 2 (batch, embedding_dim)

        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        batch_size = z1.shape[0]

        # Positive pairs: (z1[i], z2[i])
        pos_sim = (z1 * z2).sum(dim=1) / self.temperature  # (batch,)

        # Negative pairs: all other combinations
        # Similarity matrix: (batch, batch)
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Mask out diagonal (positive pairs)
        mask = torch.eye(batch_size, device=z1.device).bool()
        neg_sim = sim_matrix.masked_fill(mask, -float('inf'))

        # Compute loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def train_contrastive(
        self,
        train_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        save_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Train embedder using contrastive learning.

        Args:
            train_data: Training data (n_series, time_steps, features)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            device: Device to train on
            save_dir: Directory to save checkpoints

        Returns:
            Dictionary with training history
        """
        logger.info(
            f"Training TimeSeriesEmbedder for {epochs} epochs "
            f"(batch_size={batch_size}, lr={learning_rate})"
        )

        self.to(device)
        self.train()

        # Create dataset and dataloader
        window_size = config.get('embedding.ts_embedding.window_size', 30)

        dataset = TimeSeriesDataset(
            data=train_data,
            window_size=window_size,
            stride=1,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

        # Training history
        history = {'loss': []}

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []

            for batch in dataloader:
                batch = batch.to(device)

                # Create augmented views (simple time jittering/masking)
                aug1 = self._augment(batch)
                aug2 = self._augment(batch)

                # Get embeddings
                z1 = self.forward(aug1)
                z2 = self.forward(aug2)

                # Compute contrastive loss
                loss = self.hierarchical_contrastive_loss(z1, z2)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # Update scheduler
            scheduler.step()

            # Log progress
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )

            # Save checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                self.save(save_dir / f"checkpoint_epoch_{epoch + 1}.pt")

        logger.info("Training complete")

        return history

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation.

        Args:
            x: Input tensor (batch, time, features)

        Returns:
            Augmented tensor
        """
        # Simple augmentations: jittering and scaling
        jitter = torch.randn_like(x) * 0.01
        scale = torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * 0.1 + 1.0

        augmented = x * scale + jitter

        return augmented

    def save(self, path: Union[str, Path]):
        """Save model state.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'embedding_dim': self.embedding_dim,
                'temperature': self.temperature,
            }
        }, path)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = 'cpu') -> 'TimeSeriesEmbedder':
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)

        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        logger.info(f"Loaded model from {path}")

        return model
