import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

## QImT_GM: Quantum Image Transformation with Gaussian Mixtures.
class QImT_GM:

    def __init__(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Image not found")

        self.original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.reconstructed = None

        self.mus = None
        self.sigmas = None
        self.weights = None

        self.kraus_ops = None


    def run_segmentation(self, k_clusters, sharpening_gamma=2.0):

        pixels = self.original.reshape(-1, 1)

        gmm = GaussianMixture(n_components=k_clusters, random_state=0)
        gmm.fit(pixels)

        gmm = GaussianMixture(
            n_components=k_clusters,
            covariance_type='full',
            random_state=0
        )

        gmm.fit(pixels)

        self.mus = gmm.means_.flatten()
        self.sigmas = np.sqrt(gmm.covariances_.flatten())
        self.weights = gmm.weights_
        order = np.argsort(self.mus)

        self.mus = self.mus[order]
        self.sigmas = self.sigmas[order]
        self.weights = self.weights[order]

        intensity = np.arange(256)

        G = []

        for mu, sigma, w in zip(self.mus, self.sigmas, self.weights):

            gaussian = w * np.exp(
                -(intensity - mu)**2 / (2 * sigma**2)
            )

            G.append(gaussian)

        G = np.array(G)


        G_norm = G / np.sum(G, axis=0)


        G_norm = G_norm ** sharpening_gamma
        G_norm = G_norm / np.sum(G_norm, axis=0)


        kraus_ops = []

        for g in G_norm:

            M = np.diag(np.sqrt(g))

            kraus_ops.append(M)

        self.kraus_ops = kraus_ops

        prob_maps = []

        for M in kraus_ops:

            diag_elements = np.diag(M @ M.T)

            prob = diag_elements[self.original]

            prob_maps.append(prob)

        prob_maps = np.array(prob_maps)


        reconstructed = np.zeros_like(self.original, dtype=np.float64)

        for k, mu in enumerate(self.mus):

            reconstructed += mu * prob_maps[k]

        self.reconstructed = np.clip(reconstructed,0,255).astype(np.uint8)


        print("EM segmentation finished")
        print("Means:", self.mus)


    def calculate_metrics(self):

        mse = np.mean(
            (self.original.astype(np.float64)
            - self.reconstructed.astype(np.float64))**2
        )

        psnr = 10 * np.log10((255**2)/mse) if mse != 0 else float('inf')

        ssim_val = ssim(self.original, self.reconstructed, data_range=255)

        ent_orig = shannon_entropy(self.original)
        ent_rec = shannon_entropy(self.reconstructed)

        ent_change = ent_rec - ent_orig

        percent_change = (ent_change/ent_orig)*100 if ent_orig != 0 else 0


        return {
            "PSNR": round(psnr,2),
            "SSIM": round(ssim_val,4),
            "Entropy Change": round(ent_change,4),
            "Percent Change": round(percent_change,2)
        }


    def plot_images(self):

        plt.figure(figsize=(10,6))

        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(self.original, cmap='gray')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.title(f"Reconstructed (K={len(self.mus)})")
        plt.imshow(self.reconstructed, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    def plot_results(self):

        metrics = self.calculate_metrics()

        plt.figure(figsize=(12,5))

        plt.hist(self.original.ravel(),
                 bins=256,
                 range=(0,256),
                 alpha=0.5,
                 label='Original')

        plt.hist(self.reconstructed.ravel(),
                 bins=256,
                 range=(0,256),
                 alpha=0.5,
                 label='Reconstructed')

        plt.title(
            f"K={len(self.mus)} | "
            f"PSNR={metrics['PSNR']} | "
            f"SSIM={metrics['SSIM']} | "
            f"%Entropy Change ={metrics['Percent Change']:.2f}%"
        )

        plt.legend()

        plt.tight_layout()
        plt.show()
