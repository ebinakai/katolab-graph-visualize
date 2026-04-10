import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import pandas as pd


@dataclass(frozen=True)
class PlotStyle:
    figure_size: Tuple[float, float] = (8, 5)
    dpi: int = 300
    x_label: str = "Raman shift (cm^-1)"
    y_label: str = "Intensity (a.u.)"


class SpectrumDataStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def list_input_files(
        self, pattern: str = "*.txt", recursive: bool = True
    ) -> List[Path]:
        paths = (
            self.data_dir.rglob(pattern) if recursive else self.data_dir.glob(pattern)
        )
        return sorted(path for path in paths if path.is_file())

    @staticmethod
    def load_txt(file_path: Path, sort_x: bool = True) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["X", "Y"])
            df = df.dropna()
            if sort_x:
                df = df.sort_values("X")
            return df.reset_index(drop=True)
        except Exception as e:
            logging.error("読み込みエラー %s: %s", file_path, e)
            return pd.DataFrame(columns=["X", "Y"])

    @staticmethod
    def save_dataframe(df: pd.DataFrame, save_path: Path, index: bool = False) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=index)

    @staticmethod
    def save_figure(fig, save_path: Path, dpi: int, bbox_inches: str = "tight") -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)


def prepare_input_files(
    data_store: SpectrumDataStore, data_dir: Path, pattern: str = "*.txt"
) -> List[Path]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    input_files = data_store.list_input_files(pattern=pattern, recursive=True)

    if not input_files:
        logging.warning("%s に .txt ファイルが見つかりませんでした", data_dir)
        return []

    logging.info(
        "%d 個のファイルが見つかりました。処理を開始します...", len(input_files)
    )
    return input_files
