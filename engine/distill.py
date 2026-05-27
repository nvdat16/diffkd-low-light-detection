from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from models.diffkd import DiffKD
from ultralytics.utils import LOGGER


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, device=None, kd_weight=1.0):
        super().__init__()
        self.kd_weight = kd_weight

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.diffkd = nn.ModuleList(
            [
                DiffKD(
                    student_channels=s,
                    teacher_channels=t,
                    use_ae=(t > s),
                ).to(self.device)
                for s, t in zip(channels_s, channels_t)
            ]
        )

    def forward(self, y_s, y_t):
        if len(y_s) != len(y_t):
            y_t = y_t[-len(y_s):]

        total_loss = 0.0

        for i, (s, t) in enumerate(zip(y_s, y_t)):
            if s.shape[2:] != t.shape[2:]:
                t = F.interpolate(t, size=s.shape[2:], mode="bilinear", align_corners=False)

            _, _, kd_loss, ae_loss = self.diffkd[i](s, t.detach())

            loss = self.kd_weight * kd_loss
            if ae_loss is not None:
                loss = loss + ae_loss

            if not torch.isfinite(loss):
                LOGGER.warning(f"layer {i} NaN → zero")
                loss = torch.zeros_like(loss)

            total_loss += loss

        return total_loss


class DistillationTrainer:
    DEFAULT_LAYERS = ["6"]

    def __init__(
        self,
        student,
        teacher,
        layers=None,
        device=None,
        num_classes=1,
        teacher_layer_names=None,
        student_layer_names=None,
        teacher_channels=None,
        student_channels=None,
    ):
        self.layers = layers or self.DEFAULT_LAYERS
        self.student = student
        self.teacher = teacher
        self._handles = []
        self.student_outputs, self.teacher_outputs = [], []
        self.student_logits, self.teacher_logits = [], []

        if device is None:
            device = next(student.parameters()).device
        self.device = torch.device(device)

        self._teacher_layer_names = teacher_layer_names
        self._student_layer_names = student_layer_names
        self._teacher_channels_override = teacher_channels
        self._student_channels_override = student_channels

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640, device=self.device)
            try:
                student(dummy)
            except Exception:
                pass
            try:
                teacher(dummy)
            except Exception:
                pass

        self.channels_s, self.channels_t = [], []
        self.teacher_modules, self.student_modules = [], []
        self._find_layers()
        self.loss_fn = FeatureLoss(
            channels_s=self.channels_s,
            channels_t=self.channels_t,
            device=self.device,
        ).to(self.device)

    def _find_layers(self):
        if self._teacher_layer_names or self._student_layer_names:
            self._find_layers_by_name()
        else:
            self._find_layers_by_cv2()

    def _find_layers_by_cv2(self):
        def _collect(model, target_list, channel_list, tag=""):
            found = []
            for name, module in model.named_modules():
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "model" and parts[1] in self.layers and "cv2" in name:
                    m = module.conv if hasattr(module, "conv") else module
                    found.append((name, m))

            seen = {}
            for name, m in found:
                idx = name.split(".")[1]
                seen[idx] = (name, m)

            for idx in sorted(seen.keys(), key=lambda x: int(x)):
                n, m = seen[idx]
                ch = self._infer_out_channels(m)
                channel_list.append(ch)
                target_list.append(m)
                LOGGER.info(f"  ✓ [{tag}] Hooked layer {n} | Channels: {ch}")

        _collect(self.teacher, self.teacher_modules, self.channels_t, "TEACHER")
        _collect(self.student, self.student_modules, self.channels_s, "STUDENT")

    def _infer_out_channels(self, module):
        if hasattr(module, "out_channels"):
            return module.out_channels
        if hasattr(module, "num_features"):
            return module.num_features
        if hasattr(module, "conv") and hasattr(module.conv, "out_channels"):
            return module.conv.out_channels

        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m.out_channels
            if isinstance(m, nn.BatchNorm2d):
                return m.num_features
        return None

    def get_loss(self):
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        feat_loss = self.loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs)
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        self.student_logits.clear()
        self.teacher_logits.clear()
        return feat_loss

    def _find_layers_by_name(self):
        def _collect(model, target_names, module_list, channel_list, channels_override):
            named = dict(model.named_modules())
            for i, name in enumerate(target_names):
                if name not in named:
                    raise RuntimeError(
                        f"DistillationTrainer: module '{name}' not found. "
                        f"First 30 modules: {list(named.keys())[:30]}"
                    )
                module = named[name]
                module_list.append(module)
                ch = (
                    channels_override[i]
                    if channels_override is not None and i < len(channels_override)
                    else self._infer_out_channels(module)
                )
                channel_list.append(ch)
                LOGGER.info(f"  ✓ hooked '{name}' → out_channels={ch}")

        LOGGER.info(
            f"DistillationTrainer: hooking by name | teacher={self._teacher_layer_names} | student={self._student_layer_names}"
        )
        _collect(
            self.teacher,
            self._teacher_layer_names,
            self.teacher_modules,
            self.channels_t,
            self._teacher_channels_override,
        )
        _collect(
            self.student,
            self._student_layer_names,
            self.student_modules,
            self.channels_s,
            self._student_channels_override,
        )
        assert len(self.channels_s) == len(self.channels_t), (
            f"teacher layers ({len(self.channels_t)}) != student layers ({len(self.channels_s)})"
        )
        LOGGER.info(
            f"DistillationTrainer: hooked {len(self.channels_s)} pair(s) | student_ch={self.channels_s} | teacher_ch={self.channels_t}"
        )

    def register_hooks(self):
        self.remove_hooks()
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        self.student_logits.clear()
        self.teacher_logits.clear()

        def _first_tensor(out):
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, (list, tuple)):
                for item in out:
                    feat = _first_tensor(item)
                    if feat is not None:
                        return feat
            if isinstance(out, dict):
                for item in out.values():
                    feat = _first_tensor(item)
                    if feat is not None:
                        return feat
            return None

        def _make_hook(storage, detach=False):
            def hook(m, inp, out):
                feat = _first_tensor(out)
                if feat is None:
                    return
                feat = feat.detach() if detach else feat
                storage.append(feat)

            return hook

        for t_mod, s_mod in zip(self.teacher_modules, self.student_modules):
            self._handles.append(t_mod.register_forward_hook(_make_hook(self.teacher_outputs, detach=True)))
            self._handles.append(s_mod.register_forward_hook(_make_hook(self.student_outputs, detach=False)))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class GLARETeacher(nn.Module):
    """Wrapper to use GLARE as a frozen teacher inside DiffKD training."""

    def __init__(self, conf_path, glare_root=None):
        super().__init__()
        import sys

        conf_path = Path(conf_path).expanduser().resolve()
        if glare_root is None:
            glare_root = conf_path.parents[2]
        glare_root = Path(glare_root).expanduser().resolve()
        glare_code = glare_root / "code"
        if str(glare_code) not in sys.path:
            sys.path.insert(0, str(glare_code))

        import options.options as option
        from models import create_model
        from utils.util import opt_get

        cwd = os.getcwd()
        os.chdir(glare_root)
        try:
            opt = option.parse(str(conf_path), is_train=False)
            opt["gpu_ids"] = None
            opt = option.dict_to_nonedict(opt)
            self.glare = create_model(opt)

            model_path = opt_get(opt, ["model_path"], None)
            if model_path is not None:
                self.glare.load_network(load_path=model_path, network=self.glare.netG)
        finally:
            os.chdir(cwd)

        self.opt = opt
        self.netG = self.glare.netG
        self.net_hq = self.glare.net_hq

    def forward(self, x):
        if next(self.netG.parameters()).device != x.device:
            self.glare.netG = self.glare.netG.to(x.device)
            self.glare.net_hq = self.glare.net_hq.to(x.device)
            self.netG = self.glare.netG
            self.net_hq = self.glare.net_hq

        with torch.cuda.amp.autocast(enabled=x.is_cuda):
            if self.opt["datasets"]["train"].get("log_low", False):
                x = torch.log(torch.clamp(x + 1e-3, min=1e-3))
            sr, _ = self.glare.get_sr_with_z(x, heat=0)
        return sr
