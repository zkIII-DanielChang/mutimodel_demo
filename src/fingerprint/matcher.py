"""指纹细节点提取 + SIFT 邻域描述比对。

性能优化：
1. 细节点提取向量化：cv2.filter2D 一次算出 8 邻域和，替代 Python 双层循环。
2. 分析缓存：按 (路径, mtime, size) 缓存 _analyze 结果，
   compare 和 visualize 共享同一次分析，同一张图只处理一次。
3. 细节点采样：超过 max_minutiae 时按端点和分叉点分组均匀采样，
   避免 SIFT compute 在噪声产生的大量伪细节点上耗时。
4. 细节点映射用坐标字典 O(1) 查找，替代原来的 O(N^2) 最近邻。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.common.logger import get_logger

logger = get_logger("fingerprint.matcher")


class FingerprintMatcher:
    def __init__(
        self,
        ratio_threshold: float = 0.75,
        min_good_matches: int = 5,
        min_coverage: float = 0.15,
        border_margin: int = 15,
        gabor_ksize: int = 15,
        max_minutiae: int = 300,
    ):
        self.ratio_threshold = ratio_threshold
        self.min_good_matches = min_good_matches
        self.min_coverage = min_coverage
        self.border_margin = border_margin
        self.gabor_ksize = gabor_ksize
        self.max_minutiae = max_minutiae
        self._sift = cv2.SIFT_create()
        self._bf = cv2.BFMatcher(cv2.NORM_L2)
        self._cache: Dict[tuple, dict] = {}
        self._neighbor_kernel = np.array(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
        )
        logger.info(
            "FingerprintMatcher 初始化: ratio=%.2f, min_good=%d, min_coverage=%.2f, "
            "max_minutiae=%d",
            ratio_threshold,
            min_good_matches,
            min_coverage,
            max_minutiae,
        )

    # ---------------- 预处理 ----------------
    def _load_gray(self, image_path: str) -> Optional[np.ndarray]:
        path = Path(image_path)
        if not path.exists():
            logger.warning("指纹图像不存在: %s", path)
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("无法读取指纹图像: %s", path)
            return None
        return img

    def _gabor_enhance(self, img: np.ndarray) -> np.ndarray:
        enhanced = np.zeros_like(img, dtype=np.float32)
        for theta in (0, 45, 90, 135):
            kernel = cv2.getGaborKernel(
                (self.gabor_ksize, self.gabor_ksize),
                sigma=4.0,
                theta=np.deg2rad(theta),
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            resp = cv2.filter2D(img, cv2.CV_32F, kernel)
            enhanced = np.maximum(enhanced, resp)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced.astype(np.uint8)

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        eq = cv2.equalizeHist(gray)
        enhanced = self._gabor_enhance(eq)
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def _thin(self, binary: np.ndarray) -> np.ndarray:
        try:
            skeleton = cv2.ximgproc.thinning(binary)
            if skeleton is not None and skeleton.size > 0:
                return skeleton
        except Exception as e:
            logger.debug("cv2.ximgproc.thinning 不可用: %s", e)
        try:
            from skimage.morphology import skeletonize
            sk = skeletonize(binary > 0)
            return sk.astype(np.uint8) * 255
        except Exception as e:
            logger.warning("细化失败，返回原二值图: %s", e)
            return binary

    # ---------------- 细节点（向量化） ----------------
    def _extract_minutiae(
        self, skeleton: np.ndarray
    ) -> List[Tuple[float, float, str]]:
        h, w = skeleton.shape
        skel = (skeleton > 0).astype(np.uint8)
        # 一次卷积算出每个像素的 8 邻域骨架像素数
        neighbors = cv2.filter2D(
            skel, cv2.CV_16S, self._neighbor_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        b = self.border_margin
        margin_mask = np.zeros((h, w), dtype=bool)
        if h > 2 * b and w > 2 * b:
            margin_mask[b : h - b, b : w - b] = True
        else:
            margin_mask[:] = True

        is_skel = skel == 1
        endpoints_mask = is_skel & (neighbors == 1) & margin_mask
        bifurcations_mask = is_skel & (neighbors >= 3) & margin_mask

        ys, xs = np.where(endpoints_mask)
        endpoints = [
            (float(x), float(y), "endpoint") for x, y in zip(xs, ys)
        ]
        ys, xs = np.where(bifurcations_mask)
        bifurcations = [
            (float(x), float(y), "bifurcation") for x, y in zip(xs, ys)
        ]
        return endpoints + bifurcations

    def _limit_minutiae(
        self, minutiae: List[Tuple[float, float, str]]
    ) -> List[Tuple[float, float, str]]:
        if len(minutiae) <= self.max_minutiae:
            return minutiae
        eps = [m for m in minutiae if m[2] == "endpoint"]
        bis = [m for m in minutiae if m[2] == "bifurcation"]
        rng = np.random.default_rng(42)
        n_each = self.max_minutiae // 2
        if len(eps) > n_each:
            idx = sorted(rng.choice(len(eps), n_each, replace=False))
            eps = [eps[i] for i in idx]
        if len(bis) > n_each:
            idx = sorted(rng.choice(len(bis), n_each, replace=False))
            bis = [bis[i] for i in idx]
        logger.debug(
            "细节点过多，采样后 endpoint=%d, bifurcation=%d", len(eps), len(bis)
        )
        return eps + bis

    def _describe_minutiae(
        self, gray: np.ndarray, minutiae: List[Tuple[float, float, str]]
    ) -> Tuple[List[Tuple[float, float, str]], Optional[np.ndarray]]:
        if not minutiae:
            return [], None
        kps = [cv2.KeyPoint(m[0], m[1], 20.0) for m in minutiae]
        try:
            kps_out, descs = self._sift.compute(gray, kps)
        except Exception as e:
            logger.warning("SIFT compute 失败: %s", e)
            return [], None
        if descs is None or len(descs) == 0:
            return [], None
        # 坐标字典 O(1) 映射回原始细节点
        pos_map = {(round(m[0]), round(m[1])): m for m in minutiae}
        valid: List[Tuple[float, float, str]] = []
        for kp in kps_out:
            px, py = kp.pt
            m = pos_map.get((round(px), round(py)))
            if m is not None:
                valid.append(m)
        return valid, descs

    # ---------------- 分析缓存 ----------------
    def _cache_key(self, image_path: str):
        p = Path(image_path)
        if not p.exists():
            return None
        try:
            st = p.stat()
            return (str(p.resolve()), st.st_mtime, st.st_size)
        except Exception:
            return None

    def _analyze(self, image_path: str):
        key = self._cache_key(image_path)
        if key is not None and key in self._cache:
            return self._cache[key]
        gray = self._load_gray(image_path)
        if gray is None:
            return None
        binary = self._preprocess(gray)
        skeleton = self._thin(binary)
        minutiae = self._extract_minutiae(skeleton)
        minutiae = self._limit_minutiae(minutiae)
        valid, descs = self._describe_minutiae(gray, minutiae)
        result = {
            "gray": gray,
            "skeleton": skeleton,
            "minutiae": valid,
            "descriptors": descs,
        }
        if key is not None:
            self._cache[key] = result
        return result

    # ---------------- 匹配 ----------------
    def _match(
        self, desc1: np.ndarray, desc2: np.ndarray
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        try:
            pairs = self._bf.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            logger.warning("BFMatcher 失败: %s", e)
            return []
        good: List[cv2.DMatch] = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)
        return good

    def compare(
        self, registered_path: str, probe_path: str
    ) -> Optional[Dict[str, object]]:
        reg = self._analyze(registered_path)
        prb = self._analyze(probe_path)
        if reg is None or prb is None:
            return None

        reg_min = reg["minutiae"]
        prb_min = prb["minutiae"]
        reg_desc = reg["descriptors"]
        prb_desc = prb["descriptors"]

        if reg_desc is None or prb_desc is None:
            return {
                "good_matches": 0,
                "total_matches": 0,
                "ratio": 0.0,
                "coverage": 0.0,
                "decision": "DIFFERENT",
                "registered_keypoints": len(reg_min),
                "probe_keypoints": len(prb_min),
            }

        good = self._match(reg_desc, prb_desc)
        good_count = len(good)
        denom = max(min(len(reg_desc), len(prb_desc)), 1)
        coverage = good_count / denom
        total = min(len(reg_desc), len(prb_desc))

        decision = (
            "SAME"
            if (good_count >= self.min_good_matches and coverage >= self.min_coverage)
            else "DIFFERENT"
        )

        return {
            "good_matches": good_count,
            "total_matches": total,
            "ratio": float(coverage),
            "coverage": float(coverage),
            "decision": decision,
            "registered_keypoints": len(reg_min),
            "probe_keypoints": len(prb_min),
        }

    # ---------------- 可视化 ----------------
    def visualize(
        self, registered_path: str, probe_path: str, max_width: int = 900
    ) -> Optional[np.ndarray]:
        reg = self._analyze(registered_path)
        prb = self._analyze(probe_path)
        if reg is None or prb is None:
            return None

        reg_gray = reg["gray"]
        prb_gray = prb["gray"]
        reg_min = reg["minutiae"]
        prb_min = prb["minutiae"]
        reg_desc = reg["descriptors"]
        prb_desc = prb["descriptors"]

        h = max(reg_gray.shape[0], prb_gray.shape[0])

        def _resize_to_h(img, target_h):
            ratio = target_h / img.shape[0]
            return cv2.resize(
                img,
                (int(img.shape[1] * ratio), target_h),
                interpolation=cv2.INTER_AREA,
            )

        reg_r = _resize_to_h(reg_gray, h)
        prb_r = _resize_to_h(prb_gray, h)

        canvas = np.zeros((h, reg_r.shape[1] + prb_r.shape[1], 3), dtype=np.uint8)
        canvas[:, : reg_r.shape[1]] = cv2.cvtColor(reg_r, cv2.COLOR_GRAY2BGR)
        canvas[:, reg_r.shape[1] :] = cv2.cvtColor(prb_r, cv2.COLOR_GRAY2BGR)

        sx1 = reg_r.shape[1] / reg_gray.shape[1]
        sy1 = h / reg_gray.shape[0]
        sx2 = prb_r.shape[1] / prb_gray.shape[1]
        sy2 = h / prb_gray.shape[0]

        for x, y, kind in reg_min:
            p = (int(x * sx1), int(y * sy1))
            color = (0, 0, 255) if kind == "endpoint" else (255, 128, 0)
            cv2.circle(canvas, p, 3, color, -1)
        for x, y, kind in prb_min:
            p = (int(x * sx2) + reg_r.shape[1], int(y * sy2))
            color = (0, 0, 255) if kind == "endpoint" else (255, 128, 0)
            cv2.circle(canvas, p, 3, color, -1)

        if reg_desc is not None and prb_desc is not None:
            good = self._match(reg_desc, prb_desc)
            for m in good[:80]:
                q = reg_min[m.queryIdx]
                t = prb_min[m.trainIdx]
                p1 = (int(q[0] * sx1), int(q[1] * sy1))
                p2 = (int(t[0] * sx2) + reg_r.shape[1], int(t[1] * sy2))
                cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(
            canvas, "Registered", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            canvas, "Probe", (reg_r.shape[1] + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        return canvas