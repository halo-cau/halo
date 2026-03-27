"""Engine-level exceptions — pure Python, no web framework coupling."""


class EngineError(Exception):
    """Base class for all engine errors."""


class MeshProcessingError(EngineError):
    """The mesh could not be processed (empty, too noisy, no floor, etc.)."""


class RoomTooLargeError(EngineError):
    """The mesh bounding box exceeds the allowed room dimensions."""

    def __init__(
        self,
        dimensions: tuple[float, float, float],
        max_dims: tuple[float, float, float],
    ) -> None:
        self.dimensions = dimensions
        self.max_dims = max_dims
        super().__init__(
            f"Room bounding box {dimensions} m exceeds maximum "
            f"allowed dimensions {max_dims} m."
        )
