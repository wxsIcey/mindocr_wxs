import glob
import logging
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

_logger = logging.getLogger("mindocr")


def get_image_paths(img_dir: str) -> List[str]:
    """
    Args:
        img_dir: path to an image or a directory containing multiple images.

    Returns:
        List: list of image paths in the directory and its subdirectories.
    """
    img_dir = Path(img_dir)
    assert img_dir.exists(), f"{img_dir} does NOT exist. Please check the directory / file path."

    extensions = [".jpg", ".png", ".jpeg"]
    if img_dir.is_file():
        img_paths = [str(img_dir)]
    else:
        img_paths = [str(file) for file in img_dir.rglob("*.*") if file.suffix.lower() in extensions]

    assert (
        len(img_paths) > 0
    ), f"{img_dir} does NOT exist, or no image files exist in {img_dir}. Please check the `image_dir` arg value."
    return sorted(img_paths)


def get_ckpt_file(ckpt_dir):
    if os.path.isfile(ckpt_dir):
        ckpt_load_path = ckpt_dir
    else:
        # ckpt_load_path = os.path.join(ckpt_dir, 'best.ckpt')
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        assert len(ckpt_paths) != 0, f"No .ckpt files found in {ckpt_dir}"
        ckpt_load_path = ckpt_paths[0]
        if len(ckpt_paths) > 1:
            _logger.warning(f"More than one .ckpt files found in {ckpt_dir}. Pick {ckpt_load_path}")

    return ckpt_load_path


def crop_text_region(img, points, box_type="quad", rotate_if_vertical=True):  # polygon_type='poly'):
    # box_type: quad or poly
    def crop_img_box(img, points, rotate_if_vertical=True):
        assert len(points) == 4, "shape of points must be [4, 2]"
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        dst_pts = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])

        trans_matrix = cv2.getPerspectiveTransform(points, dst_pts)
        dst_img = cv2.warpPerspective(
            img, trans_matrix, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )

        if rotate_if_vertical:
            h, w = dst_img.shape[0:2]
            if h / float(w) >= 1.5:
                dst_img = np.rot90(dst_img)

        return dst_img

    if box_type[:4] != "poly":
        return crop_img_box(img, points, rotate_if_vertical=rotate_if_vertical)
    else:  # polygons
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = crop_img_box(img, np.array(box))
        return crop_img


def sorted_boxes(boxes):
    """
    Sort boxes in order from top to bottom, left to right
    boxes: list of box
    """

    sorted_boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))

    return sorted_boxes


def eval_rec_res(rec_res_fp, gt_fp, lower=True, ignore_space=True, filter_ood=True, char_dict_path=None):
    # gt_list = os.listdir(gt)
    # res = Parallel(n_jobs=parallel_num, backend="multiprocessing")(delayed(eval_each_rec)
    # (gt_file, gt, pred, eval_func) for gt_file in tqdm(gt_list))

    pred = []
    with open(rec_res_fp, "r") as fp:
        for line in fp:
            fn, text = line.split("\t")
            pred.append((fn, text))
    gt = []
    with open(gt_fp, "r") as fp:
        for line in fp:
            fn, text = line.split("\t")
            gt.append((fn, text))

    pred = sorted(pred, key=lambda x: x[0])
    gt = sorted(gt, key=lambda x: x[0])

    # read_dict
    if char_dict_path is None:
        ch_dict = [c for c in "0123456789abcdefghijklmnopqrstuvwxyz"]
    else:
        ch_dict = []
        with open(char_dict_path, "r") as f:
            for line in f:
                c = line.rstrip("\n\r")
                ch_dict.append(c)

    tot = 0
    correct = 0
    for i, (fn, pred_text) in enumerate(pred):
        if fn in gt[i][0]:
            gt_text = gt[i][1]
            if ignore_space:
                gt_text = gt_text.replace(" ", "")
                pred_text = pred_text.replace(" ", "")
            if lower:
                gt_text = gt_text.lower()
                pred_text = pred_text.lower()
            if filter_ood:
                gt_text = [c for c in gt_text if c in ch_dict]
                pred_text = [c for c in pred_text if c in ch_dict]

            if pred_text == gt_text:
                correct += 1

            tot += 1
        else:
            _logger.warning("Mismatched file name in pred result and gt: {fn}, {gt[i][0]}. skip this sample")

    acc = correct / tot

    return {"acc:": acc, "correct_num:": correct, "total_num:": tot}


def get_ocr_result_paths(ocr_result_dir: str) -> List[str]:
    """
    Args:
        ocr_result_dir: path to an ocr result or a directory containing multiple ocr result.

    Returns:
        List: list of ocr result in the directory and its subdirectories.
    """
    ocr_result_dir = Path(ocr_result_dir)
    if ocr_result_dir.exists() is False:
        raise ValueError(f"{ocr_result_dir} does NOT exist. Please check the directory / file path.")

    extensions = [".txt", ".json"]
    if ocr_result_dir.is_file():
        ocr_result_path = [str(ocr_result_dir)]
    else:
        ocr_result_path = [str(file) for file in ocr_result_dir.rglob("*.*") if file.suffix.lower() in extensions]

    if len(ocr_result_path) == 0:
        raise ValueError(
            f"{ocr_result_dir} does NOT exist, or no image files exist in {ocr_result_dir}."
            "Please check the `image_dir` arg value."
        )
    return sorted(ocr_result_path)


def add_padding(image, padding_size, padding_color=(0, 0, 0)):
    """
    Add padding to the image with color
    """
    if isinstance(padding_size, int):
        top, bottom, left, right = padding_size, padding_size, padding_size, padding_size
    else:
        top, bottom, left, right = padding_size

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return padded_image


def sort_words_by_poly(words, polys):
    """
    Sort detected word-boxes by polygon position in order to create a sentence
    """
    from functools import cmp_to_key

    def compare(x, y):
        dist1 = y[1][3][1] - x[1][0][1]
        dist2 = x[1][3][1] - y[1][0][1]
        if abs(dist1 - dist2) < x[1][3][1] - x[1][0][1] or abs(dist1 - dist2) < y[1][3][1] - y[1][0][1]:
            if x[1][0][0] < y[1][0][0]:
                return -1
            elif x[1][0][0] == y[1][0][0]:
                return 0
            else:
                return 1
        else:
            if x[1][0][1] < y[1][0][1]:
                return -1
            elif x[1][0][1] == y[1][0][1]:
                return 0
            else:
                return 1

    tmp = sorted(zip(words, polys), key=cmp_to_key(compare))
    return [item[0][0] for item in tmp]


def get_dict_from_file(file_path: str) -> dict:
    """
    Read a file and return a dictionary
    Args:
        file_path: path to a file
    """
    with open(file_path, "rb") as f:
        lines = f.readlines()
    return {i + 1: line.decode("utf-8").strip("\n").strip("\r\n") for i, line in enumerate(lines)}


def img_rotate(image, angle):
    """
    Rotate the incoming image at a specified angle.

    Args:
        image: an encoded image that needs to be rotated.
        angle: the target Angle at which the image is rotated

    Returns:
        rotated: the output image after rotation.
    """

    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def draw_e2e_res(dt_boxes, strs, img_path):
    src_im = cv2.imread(img_path)
    for box, str in zip(dt_boxes, strs):
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        cv2.putText(
            src_im,
            str,
            org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=1,
        )
    return src_im