import shutil
from pathlib import Path

__all__ = ["Path", "get_curdir", "rm_path"]


def get_curdir(path: str | Path, absolute: bool = True) -> Path:
    """
    Function to get the path of current workspace.

    Args:
        path (Union[str, Path]): file path.
        absolute (bool, optional): Whether to return abs path. Defaults to True.

    Returns:
        folder (Union[str, Path]): folder path.
    """
    path = Path(path).absolute() if absolute else Path(path)
    return path.parent.resolve() if absolute else path.parent


def rm_path(path: str | Path):
    pth = Path(path)
    if pth.is_dir() and not pth.is_symlink():
        shutil.rmtree(pth)
        return
    pth.unlink()


def copy_path(path_src: str | Path, path_dst: str | Path):
    if not Path(path_src).is_file():
        raise ValueError(f'Input path: "{path_src}" is invaild.')
    shutil.copy(path_src, path_dst)
