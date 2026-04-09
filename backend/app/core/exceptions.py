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
