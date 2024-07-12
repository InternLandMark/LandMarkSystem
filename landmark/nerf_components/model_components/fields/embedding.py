import torch
from torch import nn


class AppearanceEmbedding(nn.Module):
    """
    A class for the appearance embedding.
    """

    def __init__(self, n_imgs: int, n_component: int, device: torch.device) -> None:
        """
        Initialize the AppearanceEmbedding module.

        Args:
            n_imgs (int): Number of images.
            n_component (int): Dimensionality of the embedding.
            device (torch.device): Device to use for computation.
        """
        super().__init__()

        self.device = device

        self.embedding = torch.nn.Embedding(num_embeddings=n_imgs, embedding_dim=n_component, device=device)

    def forward(self, idxs: torch.Tensor = None, xyz_sampled: torch.Tensor = None, app_code: int = None):
        """
        Compute the appearance latent vector.

        Args:
            idxs (torch.Tensor, optional): Indices of the sampled points.
            xyz_sampled (torch.Tensor, optional): Sampled points.
            app_code (int, optional): Given appearance code.

        Returns:
            torch.Tensor: The appearance latent vector.
        """
        if idxs is not None:
            app_latent = self.embedding(idxs)
        elif app_code is not None:
            fake_idxs = torch.ones(xyz_sampled.shape[:-1], dtype=torch.long, device=self.device)
            fake_idxs *= app_code.long()
            app_latent = self.embedding(fake_idxs)
        else:
            app_latent = None  # TODO: raise a warning (frank)

        return app_latent
