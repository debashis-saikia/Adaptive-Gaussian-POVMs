from GaussianKraus_EM import QuantumSegmentationEM
import matplotlib.pyplot as plt


class KAnalysis:

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

        self.k_values = []

        self.results = {
            path: {
                "psnr": [],
                "ssim": [],
                "percent": []
            } for path in image_paths
        }

    def run_k_sweep(self, k_range=(1, 14), gamma=2):

        self.k_values = list(range(k_range[0], k_range[1] + 1, 2))

        for path in self.image_paths:

            print(f"\nProcessing image: {self.labels[path]}")

            model = QuantumSegmentationEM(path)

            psnr_vals = []
            ssim_vals = []
            percent_vals = []

            for k in self.k_values:

                print(f"  k = {k}")

                model.run_segmentation(k, sharpening_gamma=gamma)
                metrics = model.calculate_metrics()

                psnr_vals.append(metrics["PSNR"])
                ssim_vals.append(metrics["SSIM"])
                percent_vals.append(metrics["Percent Change"])

            self.results[path]["psnr"] = psnr_vals
            self.results[path]["ssim"] = ssim_vals
            self.results[path]["percent"] = percent_vals


    def plot_psnr_analysis(self):

        if not self.k_values:
            raise ValueError("Run k sweep first.")

        plt.figure()

        for path in self.image_paths:
            plt.plot(
                self.k_values,
                self.results[path]["psnr"],
                label=self.labels[path]
            )

        plt.xlabel("k (Number of Clusters)", fontsize=15)
        plt.ylabel("PSNR (dB)", fontsize=15)

        plt.title("PSNR vs k ($\gamma$ = 2)", fontsize=15)
        plt.legend()
        plt.grid()

        plt.show()

    def plot_ssim_analysis(self):

        if not self.k_values:
            raise ValueError("Run k sweep first.")

        plt.figure()

        for path in self.image_paths:
            plt.plot(
                self.k_values,
                self.results[path]["ssim"],
                label=self.labels[path]
            )

        plt.xlabel("k (Number of Clusters)", fontsize=15)
        plt.ylabel("SSIM", fontsize=15)

        plt.title("SSIM vs k ($\gamma$ = 2)", fontsize=15)
        plt.legend()
        plt.grid()

        plt.show()

    def plot_entropy_analysis(self):

        if not self.k_values:
            raise ValueError("Run k sweep first.")

        plt.figure()

        for path in self.image_paths:
            plt.plot(
                self.k_values,
                self.results[path]["percent"],
                label=self.labels[path]
            )

        plt.xlabel("k (Number of Clusters)", fontsize=15)
        plt.ylabel("% Entropy Change", fontsize=15)

        plt.title("Entropy Change vs k ($\gamma$ = 2)", fontsize=15)
        plt.legend()
        plt.grid()

        plt.show()


image_list = [r"C:...\The-256-256-Lena-image.png",  r"C:...\peppers.jpg",  r"C:...\barbara.bmp",  r"C:...\100.jpg",  r"C:...1001.jpg" ]

analysis = KAnalysis(image_list)
analysis.run_k_sweep(k_range=(1, 13), gamma=2)
analysis.plot_psnr_analysis()
analysis.plot_ssim_analysis()
analysis.plot_entropy_analysis()
