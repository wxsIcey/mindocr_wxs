from .det_dataset import DetDataset

__all__ = ["PGDataset"]

class PGDataset(DetDataset):
    """
    General dataset for e2e recognition
    The annotation format should follow:

    .. code-block: none

        # image file name\tannotation info containing text and polygon points encoded by json.dumps
        img61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]

    Args:
        is_train (bool): whether it is in training stage
        data_dir (str):  directory to the image data
        label_file (Union[str, List[str]]): (list of) path to the label file(s),
            where each line in the label fle contains the image file name and its ocr annotation.
        sample_ratio (Union[float, List[float]]): sample ratios for the data items in label files
        shuffle(bool): Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
                    if None, default transform pipeline for text detection will be taken.
        output_columns (list): required, indicates the keys in data dict that are expected to output for dataloader.
                            if None, all data keys will be used for return.
        global_config: additional info, used in data transformation, possible keys:
            - character_dict_path

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a tuple for the specified data item.
        You can specify the `output_columns` arg to order the output data for dataloader.

    Notes:
        1. The data file structure should be like
            ├── data_dir
            │     ├── img1.jpg
            │     ├── img2.jpg
            │     ├── {image_file_name}
            ├── label_file.txt
    """
