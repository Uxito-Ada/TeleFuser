"""Video processing utilities for loading, saving, and manipulating video data."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from telefuser.utils.logging import logger

# Maximum pixel area for common video resolutions
MAX_AREA_FOR_RESOLUTION: dict[str, int] = {
    "720p": 720 * 1280,
    "480p": 480 * 854,
    "1080p": 1080 * 1920,
    "2k": 1440 * 2560,
    "4k": 2160 * 3840,
}

# Standard aspect ratios supported by the system
SUPPORTED_ASPECT_RATIO_LIST: set[str] = {
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "1:1",
    "2:3",
    "3:2",
}


def get_target_video_size_from_ratio(
    aspect_ratio: str,
    resolution: str = "720p",
    height_division_factor: int = 1,
    width_division_factor: int = 1,
) -> tuple[int, int]:
    """Calculate target video dimensions from aspect ratio and resolution."""
    if aspect_ratio not in SUPPORTED_ASPECT_RATIO_LIST:
        logger.error(f"aspect ratio {aspect_ratio} is not supported")
    w, h = aspect_ratio.split(":")
    return get_target_image_size(int(w), int(h), resolution, height_division_factor, width_division_factor)


def get_target_image_size(
    raw_width: int,
    raw_height: int,
    resolution: str = "720p",
    height_division_factor: int = 1,
    width_division_factor: int = 1,
) -> tuple[int | None, int | None]:
    """Calculate target image/video dimensions maintaining aspect ratio.

    Args:
        raw_width: Original width
        raw_height: Original height
        resolution: Target resolution preset (e.g., "720p", "1080p")
        height_division_factor: Divisor for height alignment
        width_division_factor: Divisor for width alignment

    Returns:
        Tuple of (width, height) aligned to division factors
    """
    if resolution not in MAX_AREA_FOR_RESOLUTION:
        logger.error(f"{resolution} is not supported")
        return None, None
    max_area = MAX_AREA_FOR_RESOLUTION[resolution]
    aspect_ratio = raw_height / raw_width
    h = np.sqrt(max_area * aspect_ratio)
    w = np.sqrt(max_area / aspect_ratio)
    h = (h + height_division_factor - 1) // height_division_factor * height_division_factor
    w = (w + width_division_factor - 1) // width_division_factor * width_division_factor

    return int(w), int(h)


def images_to_base_64(images: list[Image.Image]) -> list[str]:
    """Convert PIL Images to base64 encoded strings."""
    base64_images: list[str] = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
        base64_images.append(base64_encoded)
    return base64_images


class LowMemoryVideo:
    """Memory-efficient video reader using lazy frame loading."""

    def __init__(self, file_name: str) -> None:
        self.reader = imageio.get_reader(file_name)

    def fps(self) -> float:
        return self.reader.get_meta_data().get("fps", 30)

    def __len__(self) -> int:
        return self.reader.count_frames()

    def __getitem__(self, item: int) -> Image.Image:
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self) -> None:
        self.reader.close()


def split_file_name(file_name: str) -> tuple[Any, ...]:
    """Split filename into components for natural sorting.

    Splits "frame_001.png" into ['frame_', 1, '.png'] for proper numeric sorting.
    """
    result: list[Any] = []
    number = -1
    for i in file_name:
        if ord(i) >= ord("0") and ord(i) <= ord("9"):
            if number == -1:
                number = 0
            number = number * 10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    result_tuple: tuple[Any, ...] = tuple(result)
    return result_tuple


def search_for_images(folder: str) -> list[str]:
    """Find and sort image files in a folder using natural sorting."""
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    file_list = [os.path.join(folder, i) for i in file_list]
    return file_list


class LowMemoryImageFolder:
    """Memory-efficient image folder reader with lazy loading."""

    def __init__(self, folder: str, file_list: list[str] | None = None) -> None:
        if file_list is None:
            self.file_list = search_for_images(folder)
        else:
            self.file_list = [os.path.join(folder, file_name) for file_name in file_list]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, item: int) -> Image.Image:
        return Image.open(self.file_list[item]).convert("RGB")

    def __del__(self) -> None:
        pass


def crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    """Crop image to target aspect ratio and resize.

    Maintains aspect ratio by center cropping to match target ratio, then resizes.
    """
    image_array = np.array(image)
    image_height, image_width, _ = image_array.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image_array = image_array[:, left : left + croped_width]
        result = Image.fromarray(image_array).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image_array = image_array[left : left + croped_height, :]
        result = Image.fromarray(image_array).resize((width, height))
    return result


class VideoData:
    """Unified interface for video data from file, folder, or frame list."""

    def __init__(
        self,
        video_file: str | None = None,
        image_folder: str | None = None,
        frame_list: list[Image.Image] | None = None,
        height: int | None = None,
        width: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.data_type: str
        self.data: LowMemoryVideo | LowMemoryImageFolder | list[Image.Image]
        self.video_file_path: str | None = None
        self._fps: float
        self.length: int | None = None
        self.height: int | None = None
        self.width: int | None = None

        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
            # Save video file path for audio extraction
            self.video_file_path = video_file
            self._fps = self.data.fps()
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
            self.video_file_path = None
            self._fps = kwargs.get("fps", 16)
        elif frame_list is not None:
            self.data_type = "images"
            self.data = frame_list
            self._fps = kwargs.get("fps", 16)
        else:
            raise ValueError("Cannot open video or image folder")

        if height is not None and width is not None:
            self.set_shape(height, width)
        else:
            self.width, self.height = self.__getitem__(0).size

    def fps(self) -> float:
        return self._fps

    def raw_data(self) -> list[Image.Image]:
        """Load all frames into memory."""
        frames: list[Image.Image] = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length: int) -> None:
        self.length = length

    def set_shape(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __len__(self) -> int:
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self) -> tuple[int | None, int | None]:
        return self.width, self.height

    def __getitem__(self, item: int) -> Image.Image:
        frame: Image.Image = self.data.__getitem__(item)
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = crop_and_resize(frame, self.height, self.width)
        return frame

    def __del__(self) -> None:
        pass

    def save_images(self, folder: str) -> None:
        """Save all frames as individual images."""
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(self.__len__()), desc="Saving images"):
            frame = self.__getitem__(i)
            frame.save(os.path.join(folder, f"{i}.png"))

    def extract_audio(self, audio_output_path: str | None = None) -> str | None:
        """Extract audio from video using ffmpeg.

        Args:
            audio_output_path: Audio output path, if None returns temporary file path

        Returns:
            Audio file path or None if no audio
        """
        if self.data_type != "video" or not hasattr(self, "video_file_path") or self.video_file_path is None:
            logger.warning("Audio can only be extracted from video files")
            return None

        import subprocess
        import tempfile

        # If no output path specified, create temporary file
        if audio_output_path is None:
            _, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        else:
            temp_audio_path = audio_output_path

        # Use ffmpeg to extract audio
        try:
            # Check if video has audio stream
            check_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                self.video_file_path,
            ]

            result = subprocess.run(check_cmd, capture_output=True, text=True)
            has_audio = bool(result.stdout.strip())

            if not has_audio:
                logger.warning("No audio track found in the video file")
                return None

            # Use ffmpeg to extract audio
            extract_cmd = [
                "ffmpeg",
                "-y",  # -y overwrite output file
                "-i",
                self.video_file_path,
                "-vn",  # Don't process video stream
                "-acodec",
                "pcm_s16le",  # PCM encoding
                "-ar",
                "44100",  # Sample rate
                "-ac",
                "2",  # Stereo
                temp_audio_path,
            ]

            subprocess.run(extract_cmd, check=True, capture_output=True)
            logger.info(f"Audio extracted successfully to {temp_audio_path}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio using ffmpeg: {str(e)}")
            return None
        except FileNotFoundError:
            logger.error("ffmpeg or ffprobe not found. Please install ffmpeg.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {str(e)}")
            return None

        return temp_audio_path


def save_video(
    frames: list[Image.Image] | list[np.ndarray],
    save_path: str,
    fps: float,
    quality: int = 9,
    ffmpeg_params: list[str] | None = None,
    audio_path: str | None = None,
) -> None:
    """Save frames as video file with optional audio.

    Args:
        frames: List of PIL Images or numpy arrays
        save_path: Output video file path
        fps: Frames per second
        quality: Video quality (1-10, higher is better)
        ffmpeg_params: Additional ffmpeg parameters
        audio_path: Optional audio file to merge
    """
    import subprocess

    # First save video without audio to temporary file
    temp_video_path = save_path + ".temp.mp4"
    writer = imageio.get_writer(
        temp_video_path,
        fps=fps,
        quality=quality,
        ffmpeg_params=ffmpeg_params,
        macro_block_size=2,
    )
    for frame in tqdm(frames, desc="Saving video"):
        frame_array = np.array(frame)
        writer.append_data(frame_array)
    writer.close()

    # If there's an audio file, use ffmpeg to merge
    if audio_path and os.path.exists(audio_path):
        try:
            # Use ffmpeg to merge audio and video
            cmd = [
                "ffmpeg",
                "-y",  # -y overwrite output file
                "-i",
                temp_video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",  # Copy video stream directly
                "-c:a",
                "aac",  # Audio encoding as aac
                "-shortest",  # Use the shortest stream
                save_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            # Delete temporary file
            os.remove(temp_video_path)
        except subprocess.CalledProcessError as e:
            # If merge fails, use video without audio
            os.rename(temp_video_path, save_path)
            logger.warning(f"Audio merge failed: {e}, saved video without audio")
        except FileNotFoundError:
            # If ffmpeg is not installed, use video without audio
            os.rename(temp_video_path, save_path)
            logger.warning("ffmpeg not found, saved video without audio")
    else:
        # No audio, just rename
        os.rename(temp_video_path, save_path)


def save_frames(frames: list[Image.Image], save_path: str) -> None:
    """Save frames as individual PNG images."""
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(save_path, f"{i}.png"))


def color_video_by_mask(
    video: list[Image.Image],
    mask: list[Image.Image],
    color: list[int] = [128, 128, 128],
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Apply color tint to video regions based on binary mask."""
    masked_video: list[Image.Image] = []
    mask_video: list[Image.Image] = []

    for frame, bin_frame in zip(video, mask):
        source_array = np.array(frame)
        binary_array = np.array(bin_frame)
        tmp_binary_array = binary_array.copy()
        mask_bool = binary_array.sum(axis=-1) > 0.1

        tmp_binary_array[mask_bool] = [255, 255, 255]
        tmp_binary_array_img = Image.fromarray(tmp_binary_array.astype(np.uint8))
        mask_video.append(tmp_binary_array_img)

        source_array[mask_bool] = color
        merged_frame = Image.fromarray(source_array.astype(np.uint8))
        masked_video.append(merged_frame)

    return masked_video, mask_video


def resize_video(
    video: list[Image.Image],
    width: int,
    height: int,
    frames: int,
) -> list[Image.Image]:
    """Resize video to target dimensions, keeping first N frames."""
    resized_video: list[Image.Image] = []
    assert len(video) >= frames, (
        f"The number of frames in the video {len(video)} is less than the specified number {frames}."
    )
    for frame in video[:frames]:
        resized_frame = frame.resize((width, height))
        resized_video.append(resized_frame)
    return resized_video
