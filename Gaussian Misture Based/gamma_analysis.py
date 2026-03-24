from GaussianKraus_EM import QuantumSegmentationEM
import matplotlib.pyplot as plt


class GammaAnalysis:

    def __init__(self, image_paths):
        """
        image_paths: list of image file paths
        """

        self.image_paths = image_paths

        default_labels = ["Lena", "Peppers", "Barbara", "100", "1001"]

        if len(image_paths) > len(default_labels):
            raise ValueError("More images than predefined labels.")

        self.labels = {
            path: default_labels[i]
            for i, path in enumerate(image_paths)
        }

        self.gamma_values = []

        # Store results per image
        self.results = {
            path: {
                "psnr": [],
                "ssim": [],
                "percent": []
            } for path in image_paths
        }

    def run_gamma_sweep(self, k_clusters, gamma_range=(1, 49)):

        self.gamma_values = list(range(gamma_range[0], gamma_range[1] + 1, 2))

        for path in self.image_paths:

            print(f"\nProcessing image: {self.labels[path]}")

            model = QuantumSegmentationEM(path)

            psnr_vals = []
            ssim_vals = []
            percent_vals = []

            for gamma in self.gamma_values:

                print(f"  Gamma = {gamma}")

                model.run_segmentation(k_clusters, sharpening_gamma=gamma)
                metrics = model.calculate_metrics()

                psnr_vals.append(metrics["PSNR"])
                ssim_vals.append(metrics["SSIM"])
                percent_vals.append(metrics["Percent Change"])

            self.results[path]["psnr"] = psnr_vals
            self.results[path]["ssim"] = ssim_vals
            self.results[path]["percent"] = percent_vals

    def plot_psnr_analysis(self):

        if not self.gamma_values:
            raise ValueError("Run gamma sweep first.")

        plt.figure()

        for path in self.image_paths:
            plt.plot(
                self.gamma_values,
                self.results[path]["psnr"],
                label=self.labels[path]
            )

        plt.xlabel(r"$\gamma$ (Sharpening Parameter)", fontsize=15)
        plt.ylabel("PSNR (dB)", fontsize=15)

        plt.title("PSNR vs Gamma", fontsize=15)
        plt.legend()
        plt.grid()

        plt.show()

    def plot_ssim_analysis(self):

        if not self.gamma_values:
            raise ValueError("Run gamma sweep first.")

        plt.figure()

        for path in self.image_paths:
            plt.plot(
                self.gamma_values,
                self.results[path]["ssim"],
                label=self.labels[path]
            )

        plt.xlabel(r"$\gamma$ (Sharpening Parameter)", fontsize=15)
        plt.ylabel("SSIM", fontsize=15)

        plt.title("SSIM vs Gamma", fontsize=15)
        plt.legend()
        plt.grid()

        plt.show()

    def plot_entropy_analysis(self):

        if not self.gamma_values:
            raise ValueError("Run gamma sweep first.")

        plt.figure()

        for path in self.image_paths:
            plt.plot(
                self.gamma_values,
                self.results[path]["percent"],
                label=self.labels[path]
            )

        plt.xlabel(r"$\gamma$ (Sharpening Parameter)", fontsize=15)
        plt.ylabel("% Entropy Change", fontsize=15)

        plt.title("% Entropy Change vs Gamma", fontsize=15)
        plt.legend()
        plt.grid()

        plt.show()



image_list = [
        r"C:...\The-256-256-Lena-image.png",  
        r"C:...\peppers.jpg",   
        r"C:...\barbara.bmp",   
        r"C:...\100.jpg",       
        r"C:...\1001.jpg"]

analysis = GammaAnalysis(image_list)
analysis.run_gamma_sweep(
        k_clusters=3,
        gamma_range=(1, 49)
    )

    # Plot results
    analysis.plot_psnr_analysis()
    analysis.plot_ssim_analysis()
    analysis.plot_entropy_analysis()
