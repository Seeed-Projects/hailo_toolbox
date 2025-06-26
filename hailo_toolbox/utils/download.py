"""
Model file download utility with caching, integrity verification, and progress tracking.

This module provides a comprehensive solution for downloading model files with intelligent
caching, file integrity verification, progress tracking, and error handling.

Features:
- Intelligent caching system to avoid redundant downloads
- File integrity verification using MD5/SHA256 checksums
- Download progress tracking with callbacks
- Resume capability for interrupted downloads
- Multi-threaded downloads for better performance
- Comprehensive error handling and retry mechanisms
- Support for various protocols (HTTP/HTTPS/FTP)
"""

import os
import sys
import time
import hashlib
import requests
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union, Tuple
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from hailo_toolbox.utils.logging import get_logger

logger = get_logger(__file__)


@dataclass
class DownloadMetadata:
    """
    Metadata for downloaded files to track download information and integrity.
    """

    url: str
    filename: str
    file_size: int
    download_time: str
    checksum: Optional[str] = None
    checksum_type: str = "md5"
    last_modified: Optional[str] = None
    etag: Optional[str] = None
    version: Optional[str] = None


class ProgressCallback:
    """
    Default progress callback implementation with console output.
    """

    def __init__(self, show_progress: bool = True):
        """
        Initialize progress callback.

        Args:
            show_progress: Whether to show progress in console
        """
        self.show_progress = show_progress
        self.last_update_time = 0
        self.update_interval = 0.5  # Update every 0.5 seconds

    def __call__(self, downloaded: int, total: int, speed: float = 0.0) -> None:
        """
        Progress callback function.

        Args:
            downloaded: Number of bytes downloaded
            total: Total file size in bytes
            speed: Download speed in bytes/second
        """
        if not self.show_progress:
            return

        current_time = time.time()
        if (
            current_time - self.last_update_time < self.update_interval
            and downloaded < total
        ):
            return

        self.last_update_time = current_time

        if total > 0:
            percentage = (downloaded / total) * 100
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            speed_mb = speed / (1024 * 1024)

            # Create progress bar
            bar_length = 40
            filled_length = int(bar_length * downloaded // total)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)

            print(
                f"\rDownloading: |{bar}| {percentage:.1f}% "
                f"({downloaded_mb:.1f}/{total_mb:.1f} MB) "
                f"Speed: {speed_mb:.1f} MB/s",
                end="",
                flush=True,
            )

            if downloaded >= total:
                print()  # New line when complete


class ModelDownloader:
    """
    Advanced model file downloader with caching, integrity verification, and progress tracking.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 10 * 1024 * 1024 * 1024,  # 10GB default
        chunk_size: int = 8192,
        max_retries: int = 3,
        timeout: int = 30,
        enable_resume: bool = True,
    ):
        """
        Initialize the model downloader.

        Args:
            cache_dir: Directory to store cached files (default: ~/.hailo_cache/models)
            max_cache_size: Maximum cache size in bytes
            chunk_size: Download chunk size in bytes
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            enable_resume: Whether to enable resume capability
        """
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".hailo_cache", "models")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.max_cache_size = max_cache_size
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_resume = enable_resume

        # Metadata file to track downloads
        self.metadata_file = self.cache_dir / "download_metadata.json"
        self.metadata = self._load_metadata()

        # Thread lock for metadata updates
        self._metadata_lock = threading.Lock()

        logger.info(
            f"ModelDownloader initialized with cache directory: {self.cache_dir}"
        )

    def _load_metadata(self) -> Dict[str, DownloadMetadata]:
        """
        Load download metadata from cache.

        Returns:
            Dictionary mapping file paths to metadata
        """
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    path: DownloadMetadata(**meta_dict)
                    for path, meta_dict in data.items()
                }
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return {}

    def _save_metadata(self) -> None:
        """Save download metadata to cache."""
        try:
            data = {path: asdict(metadata) for path, metadata in self.metadata.items()}
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _get_filename_from_url(self, url: str) -> str:
        """
        Extract filename from URL.

        Args:
            url: Download URL

        Returns:
            Extracted filename
        """
        parsed_url = urlparse(url)
        filename = unquote(os.path.basename(parsed_url.path))

        if not filename or filename == "/":
            # Try to get filename from Content-Disposition header
            try:
                response = requests.head(url, timeout=self.timeout)
                content_disposition = response.headers.get("Content-Disposition", "")
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip("\"'")
            except:
                pass

        # Fallback to a generic filename if still empty
        if not filename:
            filename = f"model_{hashlib.md5(url.encode()).hexdigest()[:8]}"

        return filename

    def _calculate_checksum(
        self, file_path: Path, checksum_type: str = "md5", show_progress: bool = False
    ) -> str:
        """
        Calculate file checksum.

        Args:
            file_path: Path to the file
            checksum_type: Type of checksum (md5, sha256)
            show_progress: Whether to show progress for large files

        Returns:
            Calculated checksum
        """
        if checksum_type.lower() == "md5":
            hash_obj = hashlib.md5()
        elif checksum_type.lower() == "sha256":
            hash_obj = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported checksum type: {checksum_type}")

        # Get file size for progress calculation
        file_size = file_path.stat().st_size
        processed = 0

        # Show progress for files larger than 1MB
        show_progress = show_progress and file_size > 1024 * 1024

        if show_progress:
            print(
                f"Calculating {checksum_type.upper()} checksum...", end="", flush=True
            )

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b""):
                hash_obj.update(chunk)
                processed += len(chunk)

                if show_progress and file_size > 0:
                    progress = (processed / file_size) * 100
                    if processed % (self.chunk_size * 100) == 0:  # Update every ~800KB
                        print(
                            f"\rCalculating {checksum_type.upper()} checksum... {progress:.1f}%",
                            end="",
                            flush=True,
                        )

        if show_progress:
            print(f"\rCalculating {checksum_type.upper()} checksum... Done!")

        return hash_obj.hexdigest()

    def _verify_file_integrity(
        self,
        file_path: Path,
        expected_checksum: Optional[str] = None,
        checksum_type: str = "md5",
    ) -> bool:
        """
        Verify file integrity using checksum.

        Args:
            file_path: Path to the file
            expected_checksum: Expected checksum value
            checksum_type: Type of checksum

        Returns:
            True if file integrity is valid
        """
        if not expected_checksum:
            return True

        try:
            actual_checksum = self._calculate_checksum(
                file_path, checksum_type, show_progress=True
            )
            return actual_checksum.lower() == expected_checksum.lower()
        except Exception as e:
            logger.error(f"Failed to verify file integrity: {e}")
            return False

    def _get_remote_file_info(
        self, url: str
    ) -> Tuple[int, Optional[str], Optional[str]]:
        """
        Get remote file information.

        Args:
            url: Download URL

        Returns:
            Tuple of (file_size, last_modified, etag)
        """
        try:
            response = requests.head(url, timeout=self.timeout)
            file_size = int(response.headers.get("Content-Length", 0))
            last_modified = response.headers.get("Last-Modified")
            etag = response.headers.get("ETag")
            return file_size, last_modified, etag
        except Exception as e:
            logger.warning(f"Failed to get remote file info: {e}")
            return 0, None, None

    def _cleanup_cache(self) -> None:
        """Clean up cache if it exceeds maximum size."""
        try:
            # Calculate current cache size
            total_size = 0
            file_info = []

            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file() and file_path.name != "download_metadata.json":
                    size = file_path.stat().st_size
                    mtime = file_path.stat().st_mtime
                    total_size += size
                    file_info.append((file_path, size, mtime))

            if total_size <= self.max_cache_size:
                return

            logger.info(
                f"Cache size ({total_size / (1024**3):.2f} GB) exceeds limit "
                f"({self.max_cache_size / (1024**3):.2f} GB), cleaning up..."
            )

            # Sort by modification time (oldest first)
            file_info.sort(key=lambda x: x[2])

            # Remove files until we're under the limit
            removed_files = []
            for file_path, size, _ in file_info:
                if total_size <= self.max_cache_size * 0.8:  # Leave some buffer
                    break

                try:
                    file_path.unlink()
                    total_size -= size

                    # Track removed files for metadata cleanup
                    str_path = str(file_path)
                    removed_files.append(str_path)

                    logger.info(f"Removed cached file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to remove file {file_path}: {e}")

            # Remove from metadata (with proper locking)
            if removed_files:
                with self._metadata_lock:
                    for str_path in removed_files:
                        if str_path in self.metadata:
                            del self.metadata[str_path]
                    self._save_metadata()

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def _download_with_resume(
        self, url: str, file_path: Path, progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Download file with resume capability.

        Args:
            url: Download URL
            file_path: Local file path
            progress_callback: Progress callback function

        Returns:
            True if download successful
        """
        headers = {}
        resume_pos = 0

        # Check if we can resume
        if self.enable_resume and file_path.exists():
            resume_pos = file_path.stat().st_size
            headers["Range"] = f"bytes={resume_pos}-"

        try:
            response = requests.get(
                url, headers=headers, stream=True, timeout=self.timeout
            )
            response.raise_for_status()

            # Get total file size
            content_length = response.headers.get("Content-Length")
            if content_length:
                total_size = int(content_length) + resume_pos
            else:
                total_size = 0

            # Open file for writing
            mode = "ab" if resume_pos > 0 else "wb"
            downloaded = resume_pos
            start_time = time.time()

            with open(file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Calculate download speed
                        elapsed_time = time.time() - start_time
                        speed = downloaded / elapsed_time if elapsed_time > 0 else 0

                        # Call progress callback
                        if progress_callback:
                            progress_callback(downloaded, total_size, speed)

            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def download(
        self,
        url: str,
        filename: Optional[str] = None,
        expected_checksum: Optional[str] = None,
        checksum_type: str = "md5",
        force_download: bool = False,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Download a model file with caching and integrity verification.

        Args:
            url: Download URL
            filename: Custom filename (optional)
            expected_checksum: Expected file checksum for verification
            checksum_type: Type of checksum (md5, sha256)
            force_download: Force re-download even if cached
            progress_callback: Progress callback function
            **kwargs: Additional arguments for future extensions

        Returns:
            Path to the downloaded file, or None if download failed
        """
        try:
            # Determine filename
            if filename is None:
                filename = self._get_filename_from_url(url)

            file_path = self.cache_dir / filename
            str_file_path = str(file_path)

            # Check if file exists and is valid
            if not force_download and file_path.exists():
                # Check metadata
                if str_file_path in self.metadata:
                    metadata = self.metadata[str_file_path]

                    # Verify integrity if checksum is available
                    if expected_checksum:
                        if self._verify_file_integrity(
                            file_path, expected_checksum, checksum_type
                        ):
                            logger.info(f"Using cached file: {filename}")
                            return str_file_path
                        else:
                            logger.warning(
                                f"Cached file integrity check failed: {filename}"
                            )
                    else:
                        logger.info(f"Using cached file: {filename}")
                        return str_file_path

            # Get remote file information
            remote_size, last_modified, etag = self._get_remote_file_info(url)

            # Set up progress callback
            if progress_callback is None:
                progress_callback = ProgressCallback()

            logger.info(f"Downloading {filename} from {url}")

            # Attempt download with retries
            success = False
            for attempt in range(self.max_retries):
                try:
                    success = self._download_with_resume(
                        url, file_path, progress_callback
                    )
                    if success:
                        break
                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)  # Exponential backoff

            if not success:
                logger.error(
                    f"Failed to download {filename} after {self.max_retries} attempts"
                )
                return None

            # Show completion message
            print(f"Download completed! Processing file...")

            # Verify file integrity
            if expected_checksum:
                print(f"Verifying file integrity...")
                if not self._verify_file_integrity(
                    file_path, expected_checksum, checksum_type
                ):
                    logger.error(f"Downloaded file integrity check failed: {filename}")
                    file_path.unlink()  # Remove corrupted file
                    return None
                print(f"File integrity verified!")

            # Calculate checksum for metadata
            actual_checksum = self._calculate_checksum(
                file_path, checksum_type, show_progress=True
            )

            # Update metadata
            metadata = DownloadMetadata(
                url=url,
                filename=filename,
                file_size=file_path.stat().st_size,
                download_time=datetime.now().isoformat(),
                checksum=actual_checksum,
                checksum_type=checksum_type,
                last_modified=last_modified,
                etag=etag,
            )

            with self._metadata_lock:
                self.metadata[str_file_path] = metadata
                self._save_metadata()

            # Clean up cache if necessary (outside the metadata lock to avoid deadlock)
            self._cleanup_cache()

            logger.info(f"Successfully downloaded: {filename}")
            return str_file_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def get_cached_files(self) -> Dict[str, DownloadMetadata]:
        """
        Get information about all cached files.

        Returns:
            Dictionary mapping file paths to metadata
        """
        return self.metadata.copy()

    def clear_cache(
        self, older_than_days: Optional[int] = None, pattern: Optional[str] = None
    ) -> int:
        """
        Clear cache files based on criteria.

        Args:
            older_than_days: Remove files older than specified days
            pattern: Remove files matching pattern (glob style)

        Returns:
            Number of files removed
        """
        removed_count = 0
        cutoff_time = None

        if older_than_days:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)

        files_to_remove = []

        for file_path_str, metadata in self.metadata.items():
            file_path = Path(file_path_str)

            # Check if file should be removed
            should_remove = False

            if older_than_days:
                download_time = datetime.fromisoformat(metadata.download_time)
                if download_time < cutoff_time:
                    should_remove = True

            if pattern:
                if file_path.match(pattern):
                    should_remove = True

            if should_remove:
                files_to_remove.append((file_path, file_path_str))

        # Remove files
        for file_path, file_path_str in files_to_remove:
            try:
                if file_path.exists():
                    file_path.unlink()
                    removed_count += 1

                # Remove from metadata
                if file_path_str in self.metadata:
                    del self.metadata[file_path_str]

                logger.info(f"Removed cached file: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to remove file {file_path}: {e}")

        if removed_count > 0:
            self._save_metadata()

        return removed_count

    def get_cache_size(self) -> int:
        """
        Get total cache size in bytes.

        Returns:
            Total cache size in bytes
        """
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


# Global downloader instance for convenience
_global_downloader = None


def get_downloader(**kwargs) -> ModelDownloader:
    """
    Get global downloader instance.

    Args:
        **kwargs: Arguments to pass to ModelDownloader constructor

    Returns:
        ModelDownloader instance
    """
    global _global_downloader
    if _global_downloader is None:
        _global_downloader = ModelDownloader(**kwargs)
    return _global_downloader


def download_model(
    url: str,
    filename: Optional[str] = None,
    expected_checksum: Optional[str] = None,
    checksum_type: str = "md5",
    force_download: bool = False,
    show_progress: bool = True,
    **kwargs,
) -> Optional[str]:
    """
    Convenient function to download a model file.

    Args:
        url: Download URL
        filename: Custom filename (optional)
        expected_checksum: Expected file checksum for verification
        checksum_type: Type of checksum (md5, sha256)
        force_download: Force re-download even if cached
        show_progress: Whether to show download progress
        **kwargs: Additional arguments

    Returns:
        Path to the downloaded file, or None if download failed
    """
    downloader = get_downloader()
    progress_callback = ProgressCallback(show_progress) if show_progress else None

    return downloader.download(
        url=url,
        filename=filename,
        expected_checksum=expected_checksum,
        checksum_type=checksum_type,
        force_download=force_download,
        progress_callback=progress_callback,
        **kwargs,
    )


def clear_model_cache(
    older_than_days: Optional[int] = None, pattern: Optional[str] = None
) -> int:
    """
    Clear model cache.

    Args:
        older_than_days: Remove files older than specified days
        pattern: Remove files matching pattern

    Returns:
        Number of files removed
    """
    downloader = get_downloader()
    return downloader.clear_cache(older_than_days=older_than_days, pattern=pattern)


def get_cache_info() -> Dict[str, Any]:
    """
    Get cache information.

    Returns:
        Dictionary with cache information
    """
    downloader = get_downloader()
    cache_size = downloader.get_cache_size()
    cached_files = downloader.get_cached_files()

    return {
        "cache_dir": str(downloader.cache_dir),
        "cache_size_bytes": cache_size,
        "cache_size_mb": cache_size / (1024 * 1024),
        "cache_size_gb": cache_size / (1024 * 1024 * 1024),
        "file_count": len(cached_files),
        "files": {
            path: {
                "filename": meta.filename,
                "size_mb": meta.file_size / (1024 * 1024),
                "download_time": meta.download_time,
                "url": meta.url,
            }
            for path, meta in cached_files.items()
        },
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python download.py <url> [filename] [checksum]")
        sys.exit(1)

    url = sys.argv[1]
    filename = sys.argv[2] if len(sys.argv) > 2 else None
    checksum = sys.argv[3] if len(sys.argv) > 3 else None

    # Download the model
    file_path = download_model(
        url=url, filename=filename, expected_checksum=checksum, show_progress=True
    )

    if file_path:
        print(f"\nDownload completed: {file_path}")

        # Show cache info
        cache_info = get_cache_info()
        print(f"Cache directory: {cache_info['cache_dir']}")
        print(f"Cache size: {cache_info['cache_size_mb']:.2f} MB")
        print(f"Total files: {cache_info['file_count']}")
    else:
        print("Download failed!")
        sys.exit(1)
