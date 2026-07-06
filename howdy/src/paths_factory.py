from pathlib import PurePath
import paths


def dlib_data_dir_path() -> str:
    """Base directory for the recognition model data (name kept for compatibility)"""
    return str(paths.dlib_data_dir)


def model_pack_dir_path(pack: str) -> str:
    """Directory holding the downloaded ONNX files of an InsightFace model pack"""
    return str(paths.dlib_data_dir / "models" / pack)


def compiled_models_dir_path(pack: str) -> str:
    """Directory holding the GPU-compiled model cache of a pack"""
    return str(paths.dlib_data_dir / "compiled" / pack)


def user_model_path(user: str) -> str:
    return str(paths.user_models_dir / f"{user}.dat")


def config_file_path() -> str:
    return str(paths.config_dir / "config.ini")


def snapshots_dir_path() -> PurePath:
    return paths.log_path / "snapshots"


def snapshot_path(snapshot: str) -> str:
    return str(snapshots_dir_path() / snapshot)


def user_models_dir_path() -> PurePath:
    return paths.user_models_dir


def logo_path() -> str:
    return str(paths.data_dir / "logo.png")
