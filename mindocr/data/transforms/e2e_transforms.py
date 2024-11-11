import json
from typing import List

import cv2
import numpy as np
import mindspore as ms
import mindspore.ops as ops

from .rec_transforms import RecCTCLabelEncode

__all__ = ["E2ELabelEncodeTest", "E2EResizeForTest"]


class E2ELabelEncodeTest(RecCTCLabelEncode):
    """支持不定长奇数点polygon
    need key:
        label
        (img_path)

    add key:
        polys
        texts
        ignore_tags
    """
    def __init__(self, keep_invalid: bool = True, special_id: int = -1, **kwargs):
        self.keep_invalid = keep_invalid  # 是否保留模型无法处理的特殊字符
        self.special_id = special_id  # 代表特殊字符
        super().__init__(**kwargs)

    def expand_points(self, boxes: list) -> list:
        if len(boxes) == 0:
            return boxes

        max_points_num = max(len(b) for b in boxes)
        ex_boxes = [b + [b[-1]] * (max_points_num - len(b)) for b in boxes]  # NOTE 允许奇数
        return ex_boxes

    def encode(self, text: str) -> List[int]:
        """convert text-label into text-index."""
        if len(text) == 0 or len(text) > self.max_text_len:
            return None

        if self.lower:
            text = text.lower()

        code = []
        for char in text:
            if char in self.dict:
                code.append(self.dict[char])
            elif self.keep_invalid:  # NOTE 这很坑，totaltext存在特殊字符，metric用mode_b则不排除它们，但pgnet默认词表没有它们
                code.append(self.special_id)

        if len(code) == 0:
            return None
        return code

    def __call__(self, data: dict) -> dict:
        label = json.loads(data["label"])

        polys, texts, text_tags = [], [], []
        for info in label:
            polys.append(info["points"])

            text = info["transcription"]
            ignore = text in ["*", "#", "###"]
            text_tags.append(ignore)

            if not ignore:
                code = self.encode(text)
                if code is None:
                    raise ValueError(data["img_path"].split("/")[-1])
                    return None
            else:
                code = []
            texts.append(code + [self.num_valid_chars] * (self.max_text_len - len(code)))  # use 36 to pad

        polys = self.expand_points(polys)
        data["polys"] = np.array(polys, dtype=np.float32)
        data["texts"] = np.array(texts, dtype=np.int32)
        data["ignore_tags"] = np.array(text_tags, dtype=bool)
        return data


class E2EResizeForTest:
    """
    need key:
        image

    add key:
        shape_list
    """
    def __init__(self, max_side_len: int, dataset: str, **kwargs):
        self.max_side_len = max_side_len
        self.dataset = dataset

    def __call__(self, data: dict) -> dict:
        image = data["image"]
        h, w, _ = image.shape

        if self.dataset == "totaltext":
            ratio = 1.25
            if h * ratio > self.max_side_len:
                ratio = self.max_side_len / h
        else:
            ratio = self.max_side_len / max(h, w)  # Fix the longer side

        max_stride = 128
        resize_h = int(h * ratio + max_stride - 1) // max_stride * max_stride
        resize_w = int(w * ratio + max_stride - 1) // max_stride * max_stride
        image = cv2.resize(image, dsize=(resize_w, resize_h))

        ratio_h = resize_h / h
        ratio_w = resize_w / w

        data["image"] = image
        data["shape_list"] = np.array([h, w, ratio_h, ratio_w], dtype=np.float32)
        return data
