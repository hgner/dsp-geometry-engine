"""Input allowlisting, image normalization, and output confinement."""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from bodymesh_server import config

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_DIMENSION = 16_384
MAX_PIXELS = 64_000_000
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_input_image(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser().resolve(strict=True)
    roots = [root.expanduser().resolve(strict=False) for root in config.input_roots()]
    if not any(_inside(path, root) for root in roots):
        raise ValueError(
            f"image is outside BODYMESH_INPUT_ROOTS: {path}; allowed roots: "
            + ", ".join(str(root) for root in roots)
        )
    if not path.is_file():
        raise ValueError(f"image is not a file: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image suffix {path.suffix!r}; allowed: {sorted(IMAGE_SUFFIXES)}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"image exceeds {MAX_INPUT_BYTES // (1024 * 1024)} MiB limit: {path}")
    return path


def normalize_image(source: Path, destination: Path, *, mask: bool = False) -> tuple[int, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as opened:
                opened.verify()
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened)
                if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
                    raise ValueError(f"image dimensions exceed {MAX_DIMENSION}px: {image.size}")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError(f"image exceeds {MAX_PIXELS} decoded-pixel limit: {image.size}")
                image = image.convert("L" if mask else "RGB")
                if mask:
                    image = image.point(lambda value: 255 if value >= 128 else 0, mode="1").convert("L")
                    if image.getbbox() is None:
                        raise ValueError(f"mask contains no foreground pixels: {source}")
                image.save(destination, format="PNG", optimize=True)
                width, height = image.size
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise ValueError(f"invalid image {source}: {exc}") from exc
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return width, height, digest


def normalize_reference_mask(source: Path, destination: Path, size: int) -> Path:
    """Center a binary reference silhouette on the same square canvas as Blender renders."""

    with Image.open(source) as opened:
        mask = opened.convert("L").point(lambda value: 255 if value >= 128 else 0)
        bounds = mask.getbbox()
        if bounds is None:
            raise ValueError(f"reference mask contains no foreground pixels: {source}")
        cropped = mask.crop(bounds)
        usable = max(1, int(size * 0.90))
        scale = min(usable / cropped.width, usable / cropped.height)
        width = max(1, round(cropped.width * scale))
        height = max(1, round(cropped.height * scale))
        resized = cropped.resize((width, height), resample=Image.Resampling.NEAREST)
        canvas = Image.new("L", (size, size), 0)
        canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="PNG", optimize=True)
    return destination


def validate_id(value: str, *, kind: str) -> str:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid {kind} id {value!r}")
    return value


def confined_child(root: Path, child: Path) -> Path:
    resolved_root = root.resolve(strict=False)
    resolved = child.resolve(strict=False)
    if not _inside(resolved, resolved_root):
        raise ValueError(f"path escapes output root: {resolved}")
    return resolved
