import copy
import datetime
import io
from collections import defaultdict, deque
from contextlib import redirect_stdout
import math
import os
import sys
import time

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchvision import tv_tensors
from hydra import compose, initialize
from torchvision.transforms.v2 import functional as F


with initialize(config_path=os.path.join("..", "..", "conf"), version_base="1.3"):
    cfg = compose(config_name="config")
    cfg_setup = cfg.setup
    cfg_data = cfg.data


def finetune_faster_rcnn(
    model: torch.nn.Module,
    dataset_name: str,
    dataset_train: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    modality: str,
    seed: int,
    run_name: str,
    learning_rate: int = 0.02,
    n_epochs: int = 10,
    ) -> None:
    """
    Fine-tunes a Faster R-CNN model on the given dataset.

    Args:
        model (torch.nn.Module): The Faster R-CNN model to be fine-tuned.
        dataset_name (str): Name of the dataset used for training.
        dataset_train (torch.utils.data.Dataset): Training dataset.
        dataset_test (torch.utils.data.Dataset): Test dataset.
        modality (str): Modality type (e.g., "rgb", "ir", "rgbir_ef").
        seed (int): Random seed for reproducibility.
        run_name (str): Experiment name for logging and saving models.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.02.
        n_epochs (int, optional): Number of training epochs. Defaults to 10.
    """
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn_with_modality(modality=modality),
        num_workers=4 if not cfg_setup.device == "cpu" else 0
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_with_modality(modality=modality),
        num_workers=4 if not cfg_setup.device == "cpu" else 0
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.to(cfg_setup.device)

    finetune_path = os.path.join(cfg_data.results_path, "finetuning", "fasterrcnn")
    training_log_path = os.path.join(finetune_path, dataset_train.__class__.__name__)

    writer = SummaryWriter(log_dir=os.path.join(training_log_path, f"run_{run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))

    for epoch in range(n_epochs):
        train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            cfg_setup.device,
            epoch,
            print_freq=10,
            writer=writer,
        )
        lr_scheduler.step()
        coco_evaluator = evaluate(model, data_loader_test, device=cfg_setup.device, modality=modality)

        for iou_type, coco_eval in coco_evaluator.coco_eval.items():
            writer.add_scalar(f"{iou_type}_AP", coco_eval.stats[0], epoch)
            writer.add_scalar(f"{iou_type}_AP50", coco_eval.stats[1], epoch)
            writer.add_scalar(f"{iou_type}_AP75", coco_eval.stats[2], epoch)
            writer.add_scalar(f"{iou_type}_APsmall", coco_eval.stats[3], epoch)
            writer.add_scalar(f"{iou_type}_APmedium", coco_eval.stats[4], epoch)
            writer.add_scalar(f"{iou_type}_APlarge", coco_eval.stats[5], epoch)
            writer.add_scalar(f"{iou_type}_AR", coco_eval.stats[6], epoch)
            writer.add_scalar(f"{iou_type}_AR50", coco_eval.stats[7], epoch)
            writer.add_scalar(f"{iou_type}_AR75", coco_eval.stats[8], epoch)
            writer.add_scalar(f"{iou_type}_ARsmall", coco_eval.stats[9], epoch)
            writer.add_scalar(f"{iou_type}_ARmedium", coco_eval.stats[10], epoch)
            writer.add_scalar(f"{iou_type}_ARlarge", coco_eval.stats[11], epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        for pname, param in model.named_parameters():
            writer.add_histogram(f"{pname}/weights", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"{pname}/gradients", param.grad, epoch)

    writer.close()

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(
        finetune_path,
        f"state_dict-{dataset_name.lower()}-{modality}-{seed}-{start_time}-{run_name}.pth",
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model weights saved at {final_model_path}")


### The following code is copied from https://github.com/pytorch/vision/blob/main/references/detection and just slightly adjusted and shortened ###



def train_one_epoch(
    model, optimizer, data_loader, device, epoch, print_freq, writer=None, scaler=None
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log training metrics to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", loss_value, epoch * len(data_loader) + i)
            writer.add_scalar(
                "Learning_Rate_precise",
                optimizer.param_groups[0]["lr"],
                epoch * len(data_loader) + i,
            )
            for loss_name, loss_val in loss_dict_reduced.items():
                writer.add_scalar(
                    f"Loss/train/{loss_name}",
                    loss_val.item(),
                    epoch * len(data_loader) + i,
                )

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device, modality):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = convert_to_coco_api(data_loader.dataset, modality)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(
                f"This constructor expects iou_types of type list or tuple, instead got {type(iou_types)}"
            )
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate2(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate2(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(
        -1, len(imgs.params.areaRng), len(imgs.params.imgIds)
    )


def convert_to_coco_api(ds, modality):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img_rgb, img_ir, targets = ds[img_idx]

        if modality == "rgb":
            img = img_rgb
        elif modality == "ir":
            img = img_ir
        elif modality == "rgbir_ef":
            img = 0.8 * img_rgb + 0.2 * img_ir.repeat(3, 1, 1)
        img = tv_tensors.Image(img)

        targets["boxes"] = tv_tensors.BoundingBoxes(
            targets["imageLabels"].bboxes, format="XYXY", canvas_size=F.get_size(img)
        )
        targets["labels"] = targets["imageLabels"].class_ids
        targets["area"] = (
            targets["imageLabels"].bboxes[:, 3] - targets["imageLabels"].bboxes[:, 1]
        ) * (targets["imageLabels"].bboxes[:, 2] - targets["imageLabels"].bboxes[:, 0])
        targets["iscrowd"] = torch.zeros((len(targets["labels"]),), dtype=torch.int64)

        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)"
        )


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def collate_fn_with_modality(modality):
    def collate_fn(batch):
        images = []
        labels = []
        for img_rgb, img_ir, target in batch:
            if modality == "rgb":
                img = img_rgb
            elif modality == "ir":
                img = img_ir
            elif modality == "rgbir_ef":
                img = 0.8 * img_rgb + 0.2 * img_ir.repeat(3, 1, 1)
            img = tv_tensors.Image(img)
            img = img.to(torch.float32)
            images.append(img)

            target["boxes"] = tv_tensors.BoundingBoxes(
                target["imageLabels"].bboxes, format="XYXY", canvas_size=F.get_size(img)
            )
            target["labels"] = target["imageLabels"].class_ids.to(torch.int64)
            target["area"] = (
                target["imageLabels"].bboxes[:, 3] - target["imageLabels"].bboxes[:, 1]
            ) * (
                target["imageLabels"].bboxes[:, 2] - target["imageLabels"].bboxes[:, 0]
            )
            target["iscrowd"] = torch.zeros((len(target["labels"]),), dtype=torch.int64)

            labels.append(target)

        return images, labels

    return collate_fn
