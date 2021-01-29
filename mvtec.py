import os
from pathlib import Path
from random import Random
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from typing_extensions import Literal


def create_info_csv() -> DataFrame:

    train_df = create_mode_df(mode="train")
    test_df = create_mode_df(mode="test")

    for category in train_df["category"].unique():
        category_df = train_df.loc[train_df["category"] == category]
        category_index = category_df.index.tolist()
        Random(5).shuffle(category_index)
        for i, val_index in enumerate(np.array_split(category_index, 5)):
            train_df.loc[val_index, f"cv{i}"] = "val"

    df = pd.concat([train_df, test_df])
    df = df.reset_index()
    df.to_csv("/data/info.csv", index=False)

    return df


def create_mode_df(mode: Literal["train", "test"]) -> DataFrame:

    di: Dict[str, List[str]] = {
        "old_img_path": [],
        "old_stem": [],
        "defect": [],
        "mode": [],
        "category": [],
    }
    for p in Path("/data/MVTec").glob(f"*/{mode}/*/*.png"):
        di["old_img_path"].append(str(p))
        di["old_stem"].append(p.stem)
        di["defect"].append(p.parents[0].name)
        di["mode"].append(p.parents[1].name)
        di["category"].append(p.parents[2].name)

    df = pd.DataFrame(di)
    df["cv0"] = mode
    df["cv1"] = mode
    df["cv2"] = mode
    df["cv3"] = mode
    df["cv4"] = mode
    df["stem"] = ""
    df["old_mask_path"] = ""
    for i in df.index:
        old_stem, defect, mode, category = df.loc[i, ["old_stem", "defect", "mode", "category"]]
        stem = f"{category}_{mode}_{defect}_{old_stem}"
        old_mask_path = f"/data/MVTec/{category}/ground_truth/{defect}/{old_stem}_mask.png"
        df.loc[i, "stem"] = stem
        df.loc[i, "old_mask_path"] = old_mask_path

    return df


def move_images_and_masks(df: pd.DataFrame) -> None:

    os.mkdir("/data/images")
    os.mkdir("/data/masks")
    for i in tqdm(df.index):
        old_img_path, old_mask_path, stem = df.loc[i, ["old_img_path", "old_mask_path", "stem"]]

        if os.path.exists(old_mask_path):
            os.rename(old_mask_path, f"/data/masks/{stem}.png")
        else:
            img = cv2.imread(old_img_path)
            mask = np.zeros(img.shape)
            cv2.imwrite(f"/data/masks/{stem}.png", mask)

        os.rename(old_img_path, f"/data/images/{stem}.png")


if __name__ == "__main__":

    df = create_info_csv()
    move_images_and_masks(df)
