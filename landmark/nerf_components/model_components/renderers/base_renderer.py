from abc import abstractmethod

from landmark.nerf_components.model.base_module import BaseModule


class BaseRenderer(BaseModule):
    """Base renderer class"""

    @abstractmethod
    def forward(self, *args, **kwargs) -> dict:
        """
        Performs the forward pass of the renderer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: A dictionary containing RGB and depth information.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
