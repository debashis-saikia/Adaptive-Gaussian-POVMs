import cv2
import numpy as np

def save_salt_pepper_noise(input_path, output_path, prob=0.02):
    image = cv2.imread(input_path, 0)   
    noisy = np.copy(image)
    thres = 1 - prob
    # Salt and Pepper noise
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = np.random.random()

            if r < prob:
                noisy[i][j] = 0        # pepper
            elif r > thres:
                noisy[i][j] = 255      # salt

    # Save the noisy image
    cv2.imwrite(output_path, noisy)

    print("Noisy image saved successfully!")

save_salt_pepper_noise("The-256-256-Lena-image.png", "noisy_output.jpg", 0.02)
