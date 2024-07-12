from .base_renderer import BaseRenderer


class GassianSplattingRasterizer(BaseRenderer):
    def forward(self, model, **kwargs) -> dict:
        """
        Args:
            model (list): List of models to be rendered.

        Returns:
            dict: A dictionary containing the rendered output.
        """
        pass
