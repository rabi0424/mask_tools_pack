# ComfyUI/custom_nodes/mask_tools_pack/__init__.py

import math

# -------------------------------
# Node 1: Crop By Mask (Rect + Two Masks)
# -------------------------------
class CropByMaskExpanded:
    """
    - 入力: IMAGE, MASK
    - 手順:
      1) マスクの外接矩形を取得（threshold > bbox_threshold）
      2) 中心基準で縦横ともに scale 倍へ拡張
      3) はみ出す辺だけ画像端で止める
      4) その矩形で画像を切り抜く（マスク適用はしない）
      5) 出力マスク:
         ① 元画像サイズの「くり抜き範囲（拡張後矩形）」マスク
         ② くり抜き範囲内の「選択領域」マスク（元マスクのクロップ）
    - 出力: (IMAGE, MASK, MASK)
      IMAGE: くり抜き画像（矩形全体）
      MASK①: 元画像サイズ（H×W）の矩形マスク
      MASK②: くり抜き画像サイズ（h×w）の選択領域マスク
    - 制約: 現状 B=1 のみ対応
    - サイズと位置はすべて8の倍数
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "scale": ("FLOAT", {
                    "default": 1.15, "min": 1.0, "max": 8.0, "step": 0.01,
                    "label": "Expand scale (width & height)"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01,
                    "label": "Mask threshold for bbox"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("cropped_image", "crop_region_mask_full", "selection_mask_in_crop")
    FUNCTION = "run"
    CATEGORY = "Mask Tools/Crop"

    def run(self, image, mask, scale=1.15, bbox_threshold=0.01):
        import torch

        # 形状チェック
        if image.dim() != 4 or image.size(-1) not in (3, 4):
            raise ValueError("image must be a 4D tensor [B,H,W,C] with 3 or 4 channels.")
        if mask.dim() != 3:
            raise ValueError("mask must be a 3D tensor [B,H,W].")
        if image.shape[0] != mask.shape[0]:
            raise ValueError(f"Batch size mismatch: image B={image.shape[0]} vs mask B={mask.shape[0]}")
        if image.shape[0] != 1:
            raise ValueError("This node currently supports only batch size B=1.")
        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            raise ValueError(f"Image (H,W)=({image.shape[1]},{image.shape[2]}) and Mask (H,W)=({mask.shape[1]},{mask.shape[2]}) must match.")

        device = image.device
        B, H, W, C = image.shape
        img = image[0]   # [H,W,C]
        msk = mask[0]    # [H,W]

        # 外接矩形のための2値化（bbox_thresholdより大きい画素を選択）
        sel = msk > bbox_threshold

        if torch.any(sel):
            ys, xs = torch.where(sel)
            y_min = int(torch.min(ys).item())
            y_max = int(torch.max(ys).item())
            x_min = int(torch.min(xs).item())
            x_max = int(torch.max(xs).item())
        else:
            # 選択なし: 画像全体をくり抜き範囲とみなす
            y_min, x_min = 0, 0
            y_max, x_max = H - 1, W - 1

        # 元の幅/高さ
        orig_w = x_max - x_min + 1
        orig_h = y_max - y_min + 1

        # 中心を基準に拡張
        scale = max(1.0, float(scale))
        new_w = int(math.ceil(orig_w * scale))
        new_h = int(math.ceil(orig_h * scale))

        # 8の倍数に切り上げ
        new_w = ((new_w + 7) // 8) * 8
        new_h = ((new_h + 7) // 8) * 8

        # 中心座標
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        # 左上・右下（8の倍数を保つため、leftとtopを8の倍数に丸める）
        left0 = int(cx - new_w / 2.0)
        top0 = int(cy - new_h / 2.0)
        
        # 8の倍数に調整（floor to nearest multiple of 8）
        left0 = (left0 // 8) * 8
        top0 = (top0 // 8) * 8
        
        right0 = left0 + new_w - 1
        bottom0 = top0 + new_h - 1

        # 画像端で止める（はみ出す辺だけ）
        # はみ出す場合も8の倍数を維持
        if left0 < 0:
            left = 0
            right = min(W - 1, new_w - 1)
            # 幅を8の倍数に調整
            actual_w = right - left + 1
            actual_w = ((actual_w + 7) // 8) * 8
            right = min(W - 1, left + actual_w - 1)
        elif right0 >= W:
            right = W - 1
            left = max(0, right - new_w + 1)
            # leftを8の倍数に調整
            left = (left // 8) * 8
            # 幅を8の倍数に調整
            actual_w = right - left + 1
            actual_w = ((actual_w + 7) // 8) * 8
            left = max(0, right - actual_w + 1)
        else:
            left = left0
            right = right0

        if top0 < 0:
            top = 0
            bottom = min(H - 1, new_h - 1)
            # 高さを8の倍数に調整
            actual_h = bottom - top + 1
            actual_h = ((actual_h + 7) // 8) * 8
            bottom = min(H - 1, top + actual_h - 1)
        elif bottom0 >= H:
            bottom = H - 1
            top = max(0, bottom - new_h + 1)
            # topを8の倍数に調整
            top = (top // 8) * 8
            # 高さを8の倍数に調整
            actual_h = bottom - top + 1
            actual_h = ((actual_h + 7) // 8) * 8
            top = max(0, bottom - actual_h + 1)
        else:
            top = top0
            bottom = bottom0

        # 最終的な幅と高さが8の倍数であることを確認
        final_w = right - left + 1
        final_h = bottom - top + 1
        
        # 念のため再調整（画像境界内で8の倍数を保証）
        if final_w % 8 != 0:
            target_w = ((final_w + 7) // 8) * 8
            if left + target_w - 1 < W:
                right = left + target_w - 1
            else:
                right = W - 1
                left = max(0, ((right - target_w + 1) // 8) * 8)
        
        if final_h % 8 != 0:
            target_h = ((final_h + 7) // 8) * 8
            if top + target_h - 1 < H:
                bottom = top + target_h - 1
            else:
                bottom = H - 1
                top = max(0, ((bottom - target_h + 1) // 8) * 8)

        # 最低8ピクセルは確保
        if right < left:
            left = 0
            right = min(W - 1, 7)
        if bottom < top:
            top = 0
            bottom = min(H - 1, 7)

        # 画像の切り抜き（マスクは適用しない）
        crop_img = img[top:bottom + 1, left:right + 1, :]
        # くり抜き範囲内の選択領域マスク（元マスクのクロップ）
        crop_sel_msk = msk[top:bottom + 1, left:right + 1]

        # 元画像サイズの「くり抜き範囲（矩形）」マスク
        crop_region_full = torch.zeros((H, W), dtype=msk.dtype, device=device)
        crop_region_full[top:bottom + 1, left:right + 1] = 1.0

        # バッチ次元を復元
        out_img = crop_img.unsqueeze(0)               # [1,h,w,C]
        out_mask_full = crop_region_full.unsqueeze(0) # [1,H,W]
        out_mask_inside = crop_sel_msk.unsqueeze(0)   # [1,h,w]

        # 値域の安全確保（0..1）
        out_img = torch.clamp(out_img, 0.0, 1.0)
        out_mask_full = torch.clamp(out_mask_full, 0.0, 1.0)
        out_mask_inside = torch.clamp(out_mask_inside, 0.0, 1.0)

        return (out_img, out_mask_full, out_mask_inside)


# -------------------------------
# Node 2: Overlay By Mask (Largest Region)
# -------------------------------
class OverlayByMaskLargestRect:
    """
    入力:
      - base_image (IMAGE): 下地の画像 A [B,H,W,C]
      - overlay_image (IMAGE): 重ねる画像 B [B,H,W,C]
      - mask (MASK): 合成位置決定用マスク [B,H,W], 0..1
    手順:
      1) マスクを2値化し、最大の連結領域（デフォルト 8近傍）を特定
      2) その領域の外接矩形を算出
      3) 画像Bを矩形サイズにリサイズ（オプションでアスペクト維持）
      4) 矩形位置に画像Bをはめ込み合成
         - 画像BがRGBAかつ use_overlay_alpha=True の場合、アルファ合成
         - それ以外は不透明度=opacityでブレンド
      5) 出力は合成済みの画像（base と同じサイズ）
    制約:
      - 現状 B=1 のみ対応
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "mask": ("MASK",),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01,
                    "label": "Mask threshold (selection > threshold)"
                }),
                "connectivity": ("STRING", {
                    "default": "8",
                    "choices": ["8", "4"],
                    "label": "Connected components"
                }),
                "keep_aspect_ratio": ("BOOLEAN", {
                    "default": False,
                    "label": "Keep overlay aspect ratio"
                }),
                "use_overlay_alpha": ("BOOLEAN", {
                    "default": True,
                    "label": "Use overlay alpha if present"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "label": "Opacity (if no alpha or to scale alpha)"
                }),
                "interpolation": ("STRING", {
                    "default": "bilinear",
                    "choices": ["bilinear", "bicubic", "nearest"],
                    "label": "Resize interpolation"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited_image",)
    FUNCTION = "run"
    CATEGORY = "Mask Tools/Composite"

    def run(self, base_image, overlay_image, mask, bbox_threshold=0.01,
            connectivity="8", keep_aspect_ratio=False, use_overlay_alpha=True,
            opacity=1.0, interpolation="bilinear"):
        import torch
        import torch.nn.functional as F
        import numpy as np
        from collections import deque

        # --- shape checks ---
        if base_image.dim() != 4 or base_image.size(-1) not in (3, 4):
            raise ValueError("base_image must be [B,H,W,C] with C=3 or 4.")
        if overlay_image.dim() != 4 or overlay_image.size(-1) not in (3, 4):
            raise ValueError("overlay_image must be [B,H,W,C] with C=3 or 4.")
        if mask.dim() != 3:
            raise ValueError("mask must be [B,H,W].")
        if base_image.shape[0] != 1 or overlay_image.shape[0] != 1 or mask.shape[0] != 1:
            raise ValueError("This node currently supports only batch size B=1.")
        H, W = base_image.shape[1:3]
        if mask.shape[1] != H or mask.shape[2] != W:
            raise ValueError(f"Mask size must match base_image. Got mask ({mask.shape[1]},{mask.shape[2]}) vs base ({H},{W}).")

        device = base_image.device
        dtype = base_image.dtype

        out = base_image.clone()        # [1,H,W,C]
        base = out[0]                   # [H,W,C]
        over = overlay_image[0].to(device=device, dtype=dtype)  # [hB,wB,cB]
        msk = mask[0]                   # [H,W]

        # --- find largest connected component bbox from mask ---
        sel = (msk > bbox_threshold).detach().to("cpu", copy=True).numpy().astype(np.bool_)
        if not np.any(sel):
            # No selection: return base unchanged
            print("[OverlayByMaskLargestRect] Empty selection. Returning base image.")
            return (out,)

        def largest_component_bbox(binary, conn=8):
            Hh, Ww = binary.shape
            visited = np.zeros_like(binary, dtype=np.bool_)
            best_area = 0
            best_bbox = None
            if conn == 8:
                neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            else:  # 4-connectivity
                neighbors = [(-1,0),(0,-1),(0,1),(1,0)]

            for r in range(Hh):
                for c in range(Ww):
                    if not binary[r, c] or visited[r, c]:
                        continue
                    q = deque()
                    q.append((r, c))
                    visited[r, c] = True
                    min_r = max_r = r
                    min_c = max_c = c
                    area = 0
                    while q:
                        rr, cc = q.popleft()
                        area += 1
                        if rr < min_r: min_r = rr
                        if rr > max_r: max_r = rr
                        if cc < min_c: min_c = cc
                        if cc > max_c: max_c = cc
                        for dr, dc in neighbors:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < Hh and 0 <= nc < Ww and binary[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    if area > best_area:
                        best_area = area
                        best_bbox = (min_r, min_c, max_r, max_c)
            return best_bbox

        bbox = largest_component_bbox(sel, conn=8 if connectivity == "8" else 4)
        if bbox is None:
            ys, xs = np.where(sel)
            top, left = int(ys.min()), int(xs.min())
            bottom, right = int(ys.max()), int(xs.max())
        else:
            top, left, bottom, right = map(int, bbox)

        # clamp to image bounds (safety)
        top = max(0, min(top, H - 1))
        left = max(0, min(left, W - 1))
        bottom = max(0, min(bottom, H - 1))
        right = max(0, min(right, W - 1))
        rect_h = max(1, bottom - top + 1)
        rect_w = max(1, right - left + 1)

        # --- prepare overlay resize ---
        oh, ow, oc = over.shape
        mode = interpolation

        def resize_hw(img_hw_c, new_h, new_w, mode="bilinear"):
            # img_hw_c: [H,W,C], return [new_h,new_w,C] on same device/dtype
            import torch.nn.functional as F
            Cc = img_hw_c.shape[-1]
            t = img_hw_c.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
            if mode in ("bilinear", "bicubic"):
                t2 = F.interpolate(t, size=(new_h, new_w), mode=mode, align_corners=False)
            else:
                t2 = F.interpolate(t, size=(new_h, new_w), mode=mode)
            return t2.squeeze(0).permute(1, 2, 0)

        # compute target size and placement
        if keep_aspect_ratio:
            scale = min(rect_w / max(1, ow), rect_h / max(1, oh))
            new_w = max(1, int(round(ow * scale)))
            new_h = max(1, int(round(oh * scale)))
            off_x = (rect_w - new_w) // 2
            off_y = (rect_h - new_h) // 2
        else:
            new_w, new_h = rect_w, rect_h
            off_x = 0
            off_y = 0

        # resize overlay to new size
        over_resized = resize_hw(over, new_h, new_w, mode=mode)  # [new_h,new_w,oc]

        # --- composite into ROI ---
        # ROI slices on base
        y0 = top + off_y
        x0 = left + off_x
        y1 = y0 + new_h
        x1 = x0 + new_w

        # Safety clipping (should already fit)
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            oy0 = max(0, -y0)
            ox0 = max(0, -x0)
            y0 = max(0, y0); x0 = max(0, x0)
            y1 = min(H, y1); x1 = min(W, x1)
            over_resized = over_resized[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0), :]

        roi = base[y0:y1, x0:x1, :]  # [h,w,Cb]
        h_roi, w_roi, cb = roi.shape
        if h_roi == 0 or w_roi == 0:
            print("[OverlayByMaskLargestRect] Computed ROI is empty. Returning base.")
            return (out,)

        # ensure channel compatibility
        over_rgb = over_resized[..., :3]
        if oc == 4 and use_overlay_alpha:
            over_a = over_resized[..., 3].clamp(0.0, 1.0)
            alpha = (over_a * opacity).unsqueeze(-1)  # [h,w,1]
        else:
            import torch
            alpha = torch.full((h_roi, w_roi, 1), float(opacity), dtype=dtype, device=device)

        # blend: out = over*alpha + base*(1-alpha) on RGB
        base_rgb = roi[..., :3]
        blended_rgb = over_rgb * alpha + base_rgb * (1.0 - alpha)
        roi[..., :3] = blended_rgb

        # write back
        base[y0:y1, x0:x1, :] = roi

        # clamp to [0,1]
        out = torch.clamp(base.unsqueeze(0), 0.0, 1.0)
        return (out,)


# -------------------------------
# Registration
# -------------------------------
NODE_CLASS_MAPPINGS = {
    "CropByMaskExpanded": CropByMaskExpanded,
    "OverlayByMaskLargestRect": OverlayByMaskLargestRect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropByMaskExpanded": "Crop By Mask (Rect + Two Masks)",
    "OverlayByMaskLargestRect": "Overlay By Mask (Largest Region)",
}
