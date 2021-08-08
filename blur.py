import numpy as np
from scipy.ndimage import variance
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize

def get_laplace(img_url):
    # Load image
    path = img_url
    img = io.imread(path)

    # Resize image
    img = resize(img, (400, 600))

    # Grayscale image
    img = rgb2gray(img)

    # Edge detection
    edge_laplace = laplace(img, ksize=3)

    # return {
    #     'laplace': edge_laplace,
    #     'variance': variance(edge_laplace),
    #     'max_val': np.amax(edge_laplace)
    # }
    return (variance(edge_laplace), np.amax(edge_laplace))

# Print output
# print(
#     get_laplace(
#         'test_data/blur_dataset_scaled/defocused_blurred/244_HONOR-7X_F.jpg'))
# print(f"Variance: {variance(edge_laplace)}")
# print(f"Maximum : {np.amax(edge_laplace)}")
