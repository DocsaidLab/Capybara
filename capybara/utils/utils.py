import os
from pprint import pprint
from typing import Any, Generator, Iterable, List, Union

import requests

from ..enums import COLORSTR, FORMATSTR

__all__ = [
    'make_batch', 'colorstr', 'pprint',
    'download_from_google',
]


def make_batch(
    data: Union[Iterable, Generator],
    batch_size: int
) -> Generator[List, None, None]:
    """
    This function is used to make data to batched data.

    Args:
        generator (Generator): A data generator.
        batch_size (int): batch size of batched data.

    Yields:
        batched data (list): batched data
    """
    batch = []
    for i, d in enumerate(data):
        batch.append(d)
        if (i + 1) % batch_size == 0:
            yield batch
            batch = []
    if batch:
        yield batch


def colorstr(
    obj: Any,
    color: Union[COLORSTR, int, str] = COLORSTR.BLUE,
    fmt: Union[FORMATSTR, int, str] = FORMATSTR.BOLD
) -> str:
    """
    This function is make colorful string for python.

    Args:
        obj (Any): The object you want to make it print colorful.
        color (Union[COLORSTR, int, str], optional):
            The print color of obj. Defaults to COLORSTR.BLUE.
        fmt (Union[FORMATSTR, int, str], optional):
            The print format of obj. Defaults to FORMATSTR.BOLD.
            Options = {
                'bold', 'underline'
            }

    Returns:
        string: color string.
    """
    if isinstance(color, str):
        color = color.upper()
    if isinstance(fmt, str):
        fmt = fmt.upper()
    color_code = COLORSTR.obj_to_enum(color).value
    format_code = FORMATSTR.obj_to_enum(fmt).value
    color_string = f'\033[{format_code};{color_code}m{obj}\033[0m'
    return color_string


def download_from_google(file_id: str, file_name: str, target: str = "."):
    """
    Downloads a file from Google Drive, handling potential confirmation tokens for large files.

    Args:
        file_id (str):
            The ID of the file to download from Google Drive.
        file_name (str):
            The name to save the downloaded file as.
        target (str, optional):
            The directory to save the file in. Defaults to the current directory (".").

    Raises:
        Exception: If the download fails or the file cannot be created.

    Notes:
        This function handles both small and large files. For large files, it automatically processes
        Google's confirmation token to bypass warnings about virus scans or file size limits.

    Example:
        Download a file to the current directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt"
            )

        Download a file to a specific directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt",
                target="./downloads"
            )
    """
    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    # Step 1: Attempt to download the file
    response = session.get(base_url, params={"id": file_id}, stream=True)

    # Step 2: Check if confirmation is required (e.g., for large files)
    if "content-disposition" not in response.headers:
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                # Retry with confirmation token
                response = session.get(
                    base_url, params={"id": file_id, "confirm": value}, stream=True)
                break

    # Ensure the target directory exists
    os.makedirs(target, exist_ok=True)
    file_path = os.path.join(target, file_name)

    # Step 3: Save the file to the specified directory
    try:
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:  # Avoid writing empty chunks
                    file.write(chunk)
        print(f"File successfully downloaded to: {file_path}")
    except Exception as e:
        raise Exception(f"File download failed: {e}")
