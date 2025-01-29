import numpy
import os
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
from skimage.transform import resize
import cvxpy


def ready_image(image_path, size):
    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    image = skimage.io.imread(image_path)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3: #means it's rgb
        image = rgb2gray(image)
    
    # Resize the image
    image = resize(image, size)
    
    return skimage.img_as_float(image)

def random_sampling(image, sampling_ratio):
    mask = numpy.random.rand(*image.shape) < sampling_ratio
    sampled_image = numpy.zeros_like(image)
    sampled_image[mask] = image[mask]
    return mask, sampled_image

def dct_matrix(size):
    basis = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == 0:
                basis[i, j] = numpy.sqrt(1 / size)
            else:
                basis[i, j] = numpy.sqrt(2 / size) * numpy.cos(numpy.pi * i * (2 * j + 1) / (2 * size))
    return basis


def rebuild_image(mask, sampled_image):
    m, n = sampled_image.shape
    #flatten for ease of calculation (I think it'll get flattened anyway later, but want to be safe)
    x = sampled_image.flatten()
    mask = mask.flatten()

    #get variables from Medium Article
    C = numpy.diag(mask.astype(int))
    Psi = dct_matrix(m * n)   
    C_Psi = C @ Psi
    y = C @ x
    s = cvxpy.Variable(m * n)

    constraint = [C_Psi @ s == y]
    problem = cvxpy.Minimize(cvxpy.norm1(s))
    problem = cvxpy.Problem(problem, constraint)
    #Attempt at doing it like in the medium article below, doesn't really work
    #goal = cvxpy.Minimize(cvxpy.norm(C_Psi @ s - y, 2) ** 2 + 0.1 * cvxpy.norm1(s))
    #problem = cvxpy.Problem(goal)
    problem.solve()

    x_attempt = Psi @ s.value
    return x_attempt.reshape(m, n)

#Make pretty with nice layout stuff (thanks AI)
def visualize_results(original, sampled, reconstructed):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Image")
    axes[1].imshow(sampled, cmap='gray')
    axes[1].set_title("Sampled Image")
    axes[2].imshow(reconstructed, cmap='gray')
    axes[2].set_title("Reconstructed Image")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Main function
def main(image_path, sampling_ratio, size=(64, 64)):
    og_image = ready_image(image_path, size)
    mask, sampled_image = random_sampling(og_image, sampling_ratio)
    reconstructed_image = rebuild_image(mask, sampled_image)
    visualize_results(og_image, sampled_image, reconstructed_image)

# Run on terminal launch
if __name__ == "__main__":
    image_path = input("Enter the path to an image: ").strip()
    ratio = float(input("Enter the sampling ratio: ").strip())
    main(image_path, ratio)
