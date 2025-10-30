#!/usr/bin/env python3
import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="SSIM 图像对比（可选自动配准）", layout="wide")

st.title("SSIM 图像对比（可选自动配准）")
st.caption("上传两张图片，基于 SSIM（结构相似度）计算差异并可视化（红色高亮 + 热力图）。可选对第二张图进行 ORB 配准后再比较。")

with st.sidebar:
    st.header("参数")
    align = st.checkbox("先自动配准（ORB+单应性）", value=False)
    thresh = st.slider("差异阈值（用于红色高亮）", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    st.caption("阈值越大，高亮区域越少（更严格）。")
    st.divider()
    st.caption("注意：页面会把第二张图缩放到与第一张图相同尺寸。")

col1, col2 = st.columns(2)
with col1:
    f1 = st.file_uploader("基准图像", type=["png", "jpg", "jpeg", "bmp", "webp"], key="img1")
with col2:
    f2 = st.file_uploader("对比图像", type=["png", "jpg", "jpeg", "bmp", "webp"], key="img2")

def imdecode_to_bgr(uploaded_file):
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片")
    return img  # BGR

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def align_orb(im1, im2, max_features=5000, good_match_percent=0.15):
    im1_gray = to_gray(im1)
    im2_gray = to_gray(im2)
    orb = cv2.ORB_create(nfeatures=max_features)
    k1, d1 = orb.detectAndCompute(im1_gray, None)
    k2, d2 = orb.detectAndCompute(im2_gray, None)
    if d1 is None or d2 is None:
        return im2, np.eye(3)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    if len(matches) < 4:
        return im2, np.eye(3)
    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(4, int(len(matches) * good_match_percent))
    matches = matches[:num_good]
    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        return im2, np.eye(3)
    h, w = im1.shape[:2]
    im2_aligned = cv2.warpPerspective(im2, H, (w, h))
    return im2_aligned, H

def diff_ssim(a, b):
    grayA = to_gray(a)
    grayB = to_gray(b)
    score, diff = ssim(grayA, grayB, full=True)
    diff = (1 - diff)  # larger values = more different
    return float(score), diff

def make_overlay(base, comp, diff_map, thresh=0.15):
    dm = diff_map
    dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-8)
    mask = (dm >= thresh).astype(np.uint8) * 255
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    comp_copy = comp.copy()
    red = np.zeros_like(comp_copy)
    red[..., 2] = 255  # BGR -> red
    overlay = cv2.addWeighted(comp_copy, 1.0, red, 0.4, 0, dtype=cv2.CV_8U)
    result = np.where(mask3 == 255, overlay, comp_copy)
    heat = cv2.applyColorMap((dm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return result, heat, mask

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def stack_three(a, b, c):
    h = max(a.shape[0], b.shape[0], c.shape[0])
    def pad(img):
        if img.shape[0] == h:
            return img
        top = (h - img.shape[0]) // 2
        bottom = h - img.shape[0] - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    a2, b2, c2 = pad(a), pad(b), pad(c)
    return np.hstack([a2, b2, c2])

if f1 and f2:
    try:
        im1 = imdecode_to_bgr(f1)
        im2 = imdecode_to_bgr(f2)
    except Exception as e:
        st.error(f"读取图片失败：{e}")
        st.stop()

    # 将第二张图缩放到与第一张图一致
    h1, w1 = im1.shape[:2]
    if im2.shape[:2] != (h1, w1):
        im2 = cv2.resize(im2, (w1, h1), interpolation=cv2.INTER_AREA)

    if align:
        im2, _ = align_orb(im1, im2)

    score, diff = diff_ssim(im1, im2)
    overlay, heat, _ = make_overlay(im1, im2, diff, thresh=thresh)
    vis = stack_three(im1, overlay, heat)

    st.subheader("结果")
    st.metric("SSIM 相似度（1=完全相同）", f"{score:.4f}")
    st.write("左：原图1 / 中：差异高亮 / 右：热力图")
    st.image(bgr_to_rgb(vis), use_column_width=True)

    # 下载拼接图
    vis_rgb = bgr_to_rgb(vis)
    ok, png = cv2.imencode(".png", cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button("下载拼接图（PNG）", data=png.tobytes(), file_name="ssim_diff_result.png", mime="image/png")

st.markdown("---")
st.caption("基于 SSIM 的结构性差异检测；如勾选自动配准，会先进行 ORB+单应性配准再比较。")
