from __future__ import annotations

import math
import random
import tempfile
import time
import warnings
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import autocast, torch_distributed_zero_first, unwrap_model

from engine.distill import DistillationTrainer


class DetectionTrainer(BaseTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        self.hub_session = overrides.pop("session", None)

        teacher_raw = overrides.pop("teacher", None)
        self.kd_loss_weight = overrides.pop("kd_loss_weight", 0.1)

        if isinstance(teacher_raw, (str, Path)):
            self.teacher = None
            self.teacher_path = str(teacher_raw)
        elif isinstance(teacher_raw, torch.nn.Module):
            self.teacher = teacher_raw
            tmp = tempfile.NamedTemporaryFile(suffix="_teacher_distill.pt", delete=False)
            torch.save({"model": deepcopy(teacher_raw)}, tmp.name)
            self.teacher_path = tmp.name
            tmp.close()
        else:
            self.teacher = None
            self.teacher_path = None

        super().__init__(cfg, overrides, _callbacks)

    def build_teacher_model(self):
        """Override để load IRFormer teacher từ state_dict."""
        # from models.irformer import Model as IRFormer
        # return IRFormer(in_nc=3, out_nc=3, base_nf=16)
        # from models.mbllen import MBLLEN
        # return MBLLEN()
        from models.CIDNet.CIDNet import CIDNet
        return CIDNet()

    def _do_train(self):
        """Detection-specific training loop with distillation kept local to this trainer."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        distill_trainer = None
        if self.teacher_path is not None:
            if self.teacher is None:
                LOGGER.info(f"{colorstr('Distillation:')} loading teacher from '{self.teacher_path}'")
                ckpt = torch.load(self.teacher_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    if "model" in ckpt or "ema" in ckpt:
                        self.teacher = ckpt.get("model") or ckpt.get("ema")
                    elif "state_dict" in ckpt:
                        teacher_model = self.build_teacher_model()
                        teacher_model.load_state_dict(ckpt["state_dict"], strict=True)
                        self.teacher = teacher_model
                        LOGGER.info(f"{colorstr('Distillation:')} loaded teacher via state_dict")
                    else:
                        raise ValueError(
                            f"Distillation: checkpoint '{self.teacher_path}' has no recognised key. "
                            f"Available keys: {list(ckpt.keys())}"
                        )
                else:
                    self.teacher = ckpt

                for _ in range(5):
                    if hasattr(self.teacher, "model") and isinstance(self.teacher.model, torch.nn.Module):
                        has_cv2 = any(
                            "cv2" in n and n.split(".")[0] == "model"
                            for n, _ in self.teacher.named_modules()
                        )
                        if has_cv2:
                            break
                        self.teacher = self.teacher.model
                    else:
                        break

            self.teacher = self.teacher.to(self.device).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

            distill_trainer = DistillationTrainer(
                student=unwrap_model(self.model),
                teacher=self.teacher,
                device=self.device,
                num_classes=getattr(unwrap_model(self.model), "nc", 1),
                teacher_layer_names=["fem_blocks.8"],
                student_layer_names=["model.6.m.1.1.mlp.1.act"],
                teacher_channels=[32],
                student_channels=[128],
            )

            kd_params = list(distill_trainer.loss_fn.diffkd.parameters())
            if kd_params:
                self.optimizer.add_param_group(
                    {
                        "params": kd_params,
                        "lr": self.args.lr0,
                        "initial_lr": self.args.lr0,
                        "weight_decay": self.args.weight_decay,
                        "param_group": "kd_diffkd",
                    }
                )

        epoch = self.start_epoch
        self.optimizer.zero_grad()
        self._oom_retries = 0
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if distill_trainer is not None:
                distill_trainer.register_hooks()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                try:
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                        if distill_trainer is not None:
                            with torch.no_grad():
                                teacher_input = F.interpolate(
                                    batch["img"],
                                    size=(256, 256),
                                    mode="bilinear",
                                    align_corners=False,
                                ) if distill_trainer._teacher_layer_names is not None else batch["img"]
                                self.teacher(teacher_input)

                            d_loss = distill_trainer.get_loss() * self.kd_loss_weight
                            self.loss = self.loss + d_loss

                    self.scaler.scale(self.loss).backward()
                except torch.cuda.OutOfMemoryError:
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"CUDA out of memory with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    self._clear_memory()
                    self._build_train_pipeline()
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if self.stop:
                            break

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break
            else:
                self._oom_retries = 0

            if self._oom_retries and not self.stop:
                continue

            if distill_trainer is not None:
                distill_trainer.remove_hooks()

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()

            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            if self.stop:
                break
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        if distill_trainer is not None:
            distill_trainer.remove_hooks()
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")
    
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size",
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            labels=batch, paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg", on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        del train_dataset
        return super().auto_batch(max_num_obj)