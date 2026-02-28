import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans

# ==========================================
# GANTI SESUAI NAMA FILE GAMBAR KAMU
# ==========================================
image_paths = [
    "Terang.jpeg",
    "Normal.jpeg",
    "Redup.jpeg"
]

# ==========================================
# FUNGSI KUANTISASI
# ==========================================
def uniform_quantization(image, levels=16):
    step = 256 // levels
    return (image // step) * step

def cluster_quantization(image, k=16):
    shape = image.shape
    pixels = image.reshape((-1,1))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    return clustered.reshape(shape).astype(np.uint8)

def mse(original, compressed):
    return np.mean((original - compressed) ** 2)

def psnr(original, compressed):
    error = mse(original, compressed)
    if error == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(error))

# ==========================================
# ANALISIS SEMUA GAMBAR
# ==========================================

for idx, path in enumerate(image_paths):

    print("\n====================================")
    print(f"ANALISIS GAMBAR {idx+1}: {path}")
    print("====================================")

    img = cv2.imread(path)
    if img is None:
        print("Gambar tidak ditemukan!")
        exit()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # =============================
    # KONVERSI MODEL WARNA
    # =============================
    start = time.time()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    t_gray = time.time() - start

    start = time.time()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    t_hsv = time.time() - start

    start = time.time()
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    t_lab = time.time() - start

    print(f"Waktu Gray : {t_gray:.5f} detik")
    print(f"Waktu HSV  : {t_hsv:.5f} detik")
    print(f"Waktu LAB  : {t_lab:.5f} detik")

    # =============================
    # KUANTISASI SEMUA MODEL
    # =============================

    # GRAY
    gray_uniform = uniform_quantization(gray)
    gray_cluster = cluster_quantization(gray)

    # HSV (channel V)
    hsv_v = hsv[:,:,2]
    hsv_uniform = uniform_quantization(hsv_v)
    hsv_cluster = cluster_quantization(hsv_v)

    # LAB (channel L)
    lab_l = lab[:,:,0]
    lab_uniform = uniform_quantization(lab_l)
    lab_cluster = cluster_quantization(lab_l)

    # =============================
    # HITUNG ERROR
    # =============================

    print("\n--- ERROR ANALYSIS ---")
    print("Gray Uniform MSE :", mse(gray, gray_uniform))
    print("Gray Cluster MSE :", mse(gray, gray_cluster))
    print("Gray Uniform PSNR:", psnr(gray, gray_uniform))
    print("Gray Cluster PSNR:", psnr(gray, gray_cluster))

    # =============================
    # MEMORI & KOMPRESI
    # =============================

    original_memory = img_rgb.nbytes
    quantized_memory = gray_uniform.nbytes // 2  # simulasi 4-bit
    compression_ratio = original_memory / quantized_memory

    print("\n--- MEMORY ANALYSIS ---")
    print(f"Memori Asli        : {original_memory/1024:.2f} KB")
    print(f"Memori Kuantisasi  : {quantized_memory/1024:.2f} KB")
    print(f"Rasio Kompresi     : {compression_ratio:.2f}:1")

    # =============================
    # SEGMENTASI HSV (contoh objek merah)
    # GANTI RANGE JIKA WARNA BERBEDA
    # =============================

    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 50, 50])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 + mask2
    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    detected_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    detection_percentage = (detected_pixels / total_pixels) * 100

    print("\n--- SEGMENTATION ---")
    print(f"Persentase area terdeteksi: {detection_percentage:.2f}%")

    # =============================
    # VISUALISASI
    # =============================

    plt.figure(figsize=(16,12))

    plt.subplot(4,4,1)
    plt.imshow(img_rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(4,4,2)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray")
    plt.axis("off")

    plt.subplot(4,4,3)
    plt.imshow(gray_uniform, cmap='gray')
    plt.title("Gray Uniform")
    plt.axis("off")

    plt.subplot(4,4,4)
    plt.imshow(gray_cluster, cmap='gray')
    plt.title("Gray Cluster")
    plt.axis("off")

    plt.subplot(4,4,5)
    plt.imshow(hsv[:,:,0], cmap='hsv')
    plt.title("Hue")
    plt.axis("off")

    plt.subplot(4,4,6)
    plt.imshow(hsv_uniform, cmap='gray')
    plt.title("HSV Uniform")
    plt.axis("off")

    plt.subplot(4,4,7)
    plt.imshow(lab_l, cmap='gray')
    plt.title("LAB L")
    plt.axis("off")

    plt.subplot(4,4,8)
    plt.imshow(lab_uniform, cmap='gray')
    plt.title("LAB Uniform")
    plt.axis("off")

    plt.subplot(4,4,9)
    plt.hist(gray.ravel(), bins=256)
    plt.title("Hist Gray Asli")

    plt.subplot(4,4,10)
    plt.hist(gray_uniform.ravel(), bins=16)
    plt.title("Hist Gray Uniform")

    plt.subplot(4,4,11)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask HSV")
    plt.axis("off")

    plt.subplot(4,4,12)
    plt.imshow(segmented)
    plt.title("Segmented")
    plt.axis("off")

    plt.tight_layout()
    plt.show()