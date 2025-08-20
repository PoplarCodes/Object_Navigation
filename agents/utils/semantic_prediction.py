
import argparse
import time
import os

import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T


# 引入常量：COCO类别映射、小物体索引、房间类别数量以及物体类别数量
from constants import (
    coco_categories_mapping,
    small_object_indices,
    NUM_ROOM_CATEGORIES,
    NUM_OBJECT_CATEGORIES,
)

class SemanticPredMaskRCNN():

    def __init__(self, args):
        self.segmentation_model = ImageSegmentation(args)
        self.args = args
        # 如果提供房间标注目录，则记录下来以便后续读取真实房间掩码
        self.room_mask_dir = getattr(args, "room_mask_dir", None)

    def get_prediction(self, img):
        args = self.args
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        seg_predictions, vis_output = self.segmentation_model.get_predictions(
            image_list, visualize=args.visualize == 2)

        if args.visualize == 2:
            img = vis_output.get_image()

        # semantic_input = np.zeros((img.shape[0], img.shape[1], 15 + 1))
        # 构建语义输入：15个物体类别 + 背景 + 房间N类
        semantic_input = np.zeros(
         (img.shape[0], img.shape[1],
         NUM_OBJECT_CATEGORIES + 1 + NUM_ROOM_CATEGORIES))
        # 针对每个类别仅保留得分最高的实例，避免多个相同目标造成摇摆
        best_masks = {}

        for j, class_idx in enumerate(
                seg_predictions[0]['instances'].pred_classes.cpu().numpy()):
            # 获取当前检测的置信度、掩码以及包围盒
            score = seg_predictions[0]['instances'].scores[j].item()
            obj_mask = seg_predictions[0]['instances'].pred_masks[j]
            mask_area = obj_mask.sum().item()
            bbox = seg_predictions[0]['instances'].pred_boxes.tensor[j]
            box_w = (bbox[2] - bbox[0]).item()
            box_h = (bbox[3] - bbox[1]).item()
            if class_idx in list(coco_categories_mapping.keys()) \
                    and score >= self.args.sem_pred_prob_thr \
                    and mask_area >= self.args.min_mask_area:
                # 通过面积和置信度过滤误检
                idx = coco_categories_mapping[class_idx]
                # 对于小物体额外限制掩码面积和长宽比
                # 过滤床等大物体被误判为chair，以及狭长障碍物被误判的问题
                if idx in small_object_indices:
                    img_area = img.shape[0] * img.shape[1]
                    if mask_area > self.args.max_mask_ratio * img_area:
                        continue
                    # 避免零宽高导致除零错误
                    if box_w <= 0 or box_h <= 0:
                        continue
                    bbox_ratio = max(box_w / box_h, box_h / box_w)
                    if bbox_ratio > self.args.max_bbox_ratio:
                        continue

                # 仅保留当前类别得分最高的掩码
                if idx not in best_masks or score > best_masks[idx][0]:
                    best_masks[idx] = (score, obj_mask)

        for idx, (_, mask) in best_masks.items():
            semantic_input[:, :, idx] = mask.cpu().numpy()

        # 读取或生成房间分割结果，seg_predictions中未提供时尝试从目录加载或调用模型生成
        room_masks = seg_predictions[0].get('room_masks', None)
        if room_masks is None:
            # 若未提供预测结果，则先尝试从指定目录读取
            room_masks = self._load_room_masks(img)
        if room_masks is None:
            # 目录中也没有时，可调用内部方法生成（此处为占位实现）
            room_masks = self._generate_room_masks(img)
        if room_masks is not None:
            # room_masks 维度应为 [房间数, H, W]
            room_masks = self._ensure_room_mask_shape(room_masks, img.shape[:2])
            # 将房间掩码写回预测结果，便于后续模块复用
            seg_predictions[0]['room_masks'] = room_masks
            # 如提供目录，则将掩码保存成npy文件，方便离线调试
            if self.room_mask_dir is not None:
                self._save_room_masks(room_masks)
            object_offset = NUM_OBJECT_CATEGORIES + 1  # 跳过物体通道和背景通道
            for r_idx in range(NUM_ROOM_CATEGORIES):
                semantic_input[:, :, object_offset + r_idx] = room_masks[r_idx]
        return semantic_input, img

    def _load_room_masks(self, img):
        """从外部标注目录读取房间掩码，便于在没有房间模型时使用真值掩码"""
        if self.room_mask_dir is None:
            return None
        # 此处默认读取固定文件 room_masks.npy，实际应用中可根据图像索引区分
        mask_path = os.path.join(self.room_mask_dir, 'room_masks.npy')
        if os.path.exists(mask_path):
            return np.load(mask_path)
        return None

    def _generate_room_masks(self, img):
        """调用房间分割模型生成掩码，当前为占位实现"""
        if img is None:
            return None
        # 这里简单返回全零掩码，实际应用中应替换为真实的房间分割模型预测
        h, w, _ = img.shape
        return np.zeros((NUM_ROOM_CATEGORIES, h, w), dtype=np.float32)

    def _ensure_room_mask_shape(self, room_masks, img_shape):
        """确保房间掩码维度与 NUM_ROOM_CATEGORIES 和图像尺寸一致"""
        h, w = img_shape
        if isinstance(room_masks, torch.Tensor):
            room_masks = room_masks.cpu().numpy()
        room_masks = np.asarray(room_masks)
        # 若通道数不足则补零，过多则截断
        if room_masks.shape[0] < NUM_ROOM_CATEGORIES:
            pad = np.zeros((NUM_ROOM_CATEGORIES - room_masks.shape[0], h, w), dtype=room_masks.dtype)
            room_masks = np.concatenate([room_masks, pad], axis=0)
        elif room_masks.shape[0] > NUM_ROOM_CATEGORIES:
            room_masks = room_masks[:NUM_ROOM_CATEGORIES]
        # 若尺寸不匹配则调整为图像大小
        if room_masks.shape[1] != h or room_masks.shape[2] != w:
            room_masks = np.array([
                cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                for mask in room_masks
            ])
        return room_masks

    def _save_room_masks(self, room_masks):
        """将房间掩码保存到指定目录，便于后续复现或调试"""
        os.makedirs(self.room_mask_dir, exist_ok=True)
        mask_path = os.path.join(self.room_mask_dir, 'room_masks.npy')
        np.save(mask_path, room_masks)



def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map


class ImageSegmentation():
    def __init__(self, args):
        # string_args = """
        #     --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        #     --input input1.jpeg
        #     --confidence-threshold {}
        #     --opts MODEL.WEIGHTS
        #     detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
        #     """.format(args.sem_pred_prob_thr)
        string_args = """
                    --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
                    --input input1.jpeg
                    --confidence-threshold {}
                    --opts MODEL.WEIGHTS
                    detectron2/model_final_f10217.pkl
                    """.format(args.sem_pred_prob_thr)

        if args.sem_gpu_id == -2:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += """ MODEL.DEVICE cuda:{}""".format(args.sem_gpu_id)

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, img, visualize=0):
        return self.demo.run_on_image(img, visualize=visualize)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    return cfg


def get_seg_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = BatchPredictor(cfg)

    def run_on_image(self, image_list, visualize=0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        all_predictions = self.predictor(image_list)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        if visualize:
            predictions = all_predictions[0]
            image = image_list[0]
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        return all_predictions, vis_output


class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.

    Compared to using the model directly, this class does the following
    additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
