import cv2
import numpy as np
import streamlit as st

# Asumsikan df sudah ada, dengan kolom 'path' dan 'label'
# Contoh: df = pd.DataFrame({'path': [...], 'label': [...])

# Fungsi buat GLCM manual
def compute_glcm_manual(image, dx, dy, levels=256):
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    rows, cols = image.shape

    # Tentukan batas i dan j agar tidak out of bounds
    if dy >= 0:
        i_range = range(rows - dy)
    else:
        i_range = range(-dy, rows)

    if dx >= 0:
        j_range = range(cols - dx)
    else:
        j_range = range(-dx, cols)

    for i in i_range:
        for j in j_range:
            row_val = image[i, j]
            col_val = image[i + dy, j + dx]
            glcm[row_val, col_val] += 1

    return glcm + glcm.T  # Buat symmetric (optional)

# Fungsi hitung fitur dari GLCM
def compute_glcm_features(glcm):
    glcm = glcm.astype(np.float64)
    glcm_sum = glcm.sum()
    if glcm_sum == 0:
        return [0] * 6
    glcm /= glcm_sum  # Normalisasi

    levels = glcm.shape[0]
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')

    contrast = np.sum((i - j) ** 2 * glcm)
    dissimilarity = np.sum(np.abs(i - j) * glcm)
    homogeneity = np.sum(glcm / (1.0 + (i - j) ** 2))
    asm = np.sum(glcm ** 2)
    energy = np.sqrt(asm)

    mi = np.sum(i * glcm)
    mj = np.sum(j * glcm)
    si = np.sqrt(np.sum((i - mi) ** 2 * glcm))
    sj = np.sqrt(np.sum((j - mj) ** 2 * glcm))
    if si > 0 and sj > 0:
        correlation = np.sum((i - mi) * (j - mj) * glcm) / (si * sj)
    else:
        correlation = 0

    return [contrast, dissimilarity, homogeneity, asm, energy, correlation]

# Arah GLCM manual: (dx, dy)
directions = {
    '0': (1, 0),
    '45': (1, -1),
    '90': (0, 1),
    '135': (-1, -1)
}
def get_feat(image):
    features = []
    for suffix, (dx, dy) in directions.items():
        glcm = compute_glcm_manual(image, dx, dy)
        contrast, dissimilarity, homogeneity, asm, energy, correlation = compute_glcm_features(glcm)
        features.extend([contrast, dissimilarity, homogeneity, asm, energy, correlation])
    
    return np.array(features)
