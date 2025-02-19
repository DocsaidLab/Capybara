import os
import re
from pprint import pprint
from typing import Any, Generator, Iterable, List, Union

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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
    # 第一次嘗試：docs.google.com/uc?export=download&id=檔案ID
    base_url = "https://docs.google.com/uc"
    session = requests.Session()
    params = {
        "export": "download",
        "id": file_id
    }
    response = session.get(base_url, params=params, stream=True)

    # 如果已經出現 Content-Disposition，代表直接拿到檔案
    if "content-disposition" not in response.headers:
        # 先嘗試從 cookies 拿 token
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        # 如果 cookies 沒有，就從 HTML 解析
        if not token:
            soup = BeautifulSoup(response.text, "html.parser")
            # 常見情況：HTML 裡面有一個 form#download-form
            download_form = soup.find("form", {"id": "download-form"})
            if download_form and download_form.get("action"):
                # 將 action 裡的網址抓出來，可能是 drive.usercontent.google.com/download
                download_url = download_form["action"]
                # 收集所有 hidden 欄位
                hidden_inputs = download_form.find_all(
                    "input", {"type": "hidden"})
                form_params = {}
                for inp in hidden_inputs:
                    if inp.get("name") and inp.get("value") is not None:
                        form_params[inp["name"]] = inp["value"]

                # 用這些參數去重新 GET
                # 注意：原本 action 可能只是相對路徑，這裡直接用完整網址
                response = session.get(
                    download_url, params=form_params, stream=True)
            else:
                # 或者有些情況是直接在 HTML 裡 search confirm=xxx
                match = re.search(r'confirm=([0-9A-Za-z-_]+)', response.text)
                if match:
                    token = match.group(1)
                    # 帶上 confirm token 再重新請求 docs.google.com
                    params["confirm"] = token
                    response = session.get(
                        base_url, params=params, stream=True)
                else:
                    raise Exception("無法在回應中找到下載連結或確認參數，下載失敗。")

        else:
            # 直接帶上 cookies 抓到的 token 再打一次
            params["confirm"] = token
            response = session.get(base_url, params=params, stream=True)

    # 確保下載目錄存在
    os.makedirs(target, exist_ok=True)
    file_path = os.path.join(target, file_name)

    # 開始把檔案 chunk 寫到本地，附帶進度條
    try:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, "wb") as f, tqdm(
            desc=file_name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"File successfully downloaded to: {file_path}")

    except Exception as e:
        raise Exception(f"File download failed: {e}")
