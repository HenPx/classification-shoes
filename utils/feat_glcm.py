import numpy as np
def compute_glcm(image, distance=1, angle=0):
    """
    Fungsi ini menghitung GLCM untuk satu arah dan satu jarak secara manual.
    Hanya mendukung citra grayscale dalam format numpy array.
    """
    rows, cols = image.shape
    levels = 256  # Untuk 8-bit grayscale

    # Matriks GLCM
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    # Arah: angle 0 derajat (horizontal)
    dr, dc = 0, distance

    for r in range(rows):
        for c in range(cols - distance):
            i = int(image[r, c])
            j = int(image[r + dr, c + dc])
            glcm[i, j] += 1

    # Normalisasi
    glcm = glcm / glcm.sum()
    return glcm

def extract_glcm_features(image):
    """
    Menghitung fitur statistik dari GLCM:
    - Contrast
    - Homogeneity
    - Energy
    - Correlation
    """
    glcm = compute_glcm(image)

    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))

    contrast = np.sum(glcm * (i - j) ** 2)
    homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))
    energy = np.sum(glcm ** 2)

    # Menghindari pembagian dengan nol
    mean_i = np.sum(i * glcm)
    mean_j = np.sum(j * glcm)
    std_i = np.sqrt(np.sum(((i - mean_i) ** 2) * glcm))
    std_j = np.sqrt(np.sum(((j - mean_j) ** 2) * glcm))

    if std_i * std_j == 0:
        correlation = 1.0
    else:
        correlation = np.sum(((i - mean_i) * (j - mean_j) * glcm)) / (std_i * std_j)

    return np.array([contrast, homogeneity, energy, correlation])