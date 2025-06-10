import cv2
import numpy as np
import os

IMG_SIZE = (100, 100)

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE).flatten()
            images.append(img)
            filenames.append(img_path)
    return np.array(images).T, filenames  # shape: (n_pixels, n_samples)

def compute_mean_face(X):
    return np.mean(X, axis=1).reshape(-1, 1)

def power_iteration(A, num_iter=1000, tol=1e-6):
    n = A.shape[1]
    v = np.random.rand(n, 1)
    v /= np.linalg.norm(v)
    last_lambda = 0
    for _ in range(num_iter):
        Av = A @ v
        lambda_ = np.linalg.norm(Av)
        v = Av / lambda_
        if np.abs(lambda_ - last_lambda) < tol:
            break
        last_lambda = lambda_
    return lambda_, v

def compute_top_k_eigenfaces(A, k):
    C = A.T @ A
    eigenvectors = []
    for _ in range(k):
        _, v = power_iteration(C)
        eigenvectors.append(v)
        C -= (v @ v.T) * (v.T @ C @ v)
    eigenfaces = A @ np.hstack(eigenvectors)
    return eigenfaces

def recognize(face_test, mean_face, eigenfaces, projections, filenames):
    test_centered = face_test.reshape(-1, 1) - mean_face
    test_proj = eigenfaces.T @ test_centered
    dists = np.linalg.norm(projections - test_proj, axis=0)
    idx = np.argmin(dists)
    return filenames[idx], dists[idx]
