"""HTTP exception wrappers for engine-level errors."""

from fastapi import HTTPException, status


class MeshTooLargeError(HTTPException):
    def __init__(self, size_bytes: int) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Uploaded file is {size_bytes / 1024 / 1024:.1f} MB. "
                f"Maximum allowed is 50 MB."
            ),
        )


ALLOWED_SCAN_EXTENSIONS = (".obj", ".ply")


class InvalidFileTypeError(HTTPException):
    def __init__(self, filename: str) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: '{filename}'. Only .obj and .ply files are accepted.",
        )


class ModelNotAvailableError(HTTPException):
    """Raised when the RL policy checkpoint is missing on the host."""

    def __init__(self, detail: str = "RL model checkpoint not deployed") -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


class ScanNotFoundError(HTTPException):
    """Raised when a scan_id has expired from the cache or never existed."""

    def __init__(self, scan_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"scan_id '{scan_id}' not found. Re-upload the scan via /visualize.",
        )
