import numpy as np
import os
from sklearn import preprocessing, svm
from blur import get_laplace

#get all images in a directory
blurry_images_dir = r'test_data/defocused_blurred/'
sharp_images_dir = r'test_data/sharp/'
blurry_imgs = []
sharp_imgs = []
for file in os.listdir(blurry_images_dir):
    blurry_imgs.append(os.path.join(blurry_images_dir, file))

for file in os.listdir(sharp_images_dir):
    sharp_imgs.append(os.path.join(sharp_images_dir, file))

#process images to get laplace var and max values
blurry_laplaces = []
sharp_laplaces = []

for img in blurry_imgs:
    img_var, img_max = get_laplace(img)
    blurry_laplaces.append((img_var, img_max))

for img in sharp_imgs:
    img_var, img_max = get_laplace(img)
    sharp_laplaces.append((img_var, img_max))

# # start with the results from the previous script
# sharp_laplaces = [ (variance(edge_laplace_sharp_1), np.amax(edge_laplace_sharp_1)), ... ]
# blurry_laplaces = [ (variance(edge_laplace_blurry_1), np.amax(edge_laplace_blurry_1)), ... ]

# set class labels (non-blurry / blurry) and prepare features
y = np.concatenate((np.ones((len(sharp_laplaces), )), np.zeros((len(blurry_laplaces), ))), axis=0)
laplaces = np.concatenate((np.array(sharp_laplaces), np.array(blurry_laplaces)), axis=0)

# scale features
laplaces = preprocessing.scale(laplaces)

# train the classifier (support vector machine)
clf = svm.SVC(kernel='linear', C=100000)
clf.fit(laplaces, y)

# print parameters
print(f'CLF: {clf}')
print(f'Weights: {clf.coef_[0]}')
print(f'Intercept: {clf.intercept_}')

# # make sample predictions
# clf.predict([[0.00040431, 0.1602369]])  # result: 0 (blurred)
# clf.predict([[0.00530690, 0.7531759]])  # result: 1 (sharp)
