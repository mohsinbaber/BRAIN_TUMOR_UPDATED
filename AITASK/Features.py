import numpy as np
import cv2
import pandas
import os, sys
from pathlib import Path
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from scipy import ndimage
from skimage.filters import gabor
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
import warnings
from numpy.linalg import norm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

i = 1
mal = 1


list = os.listdir('C:/Users/LENOVO/Desktop/dataset/train/Glioma') # dir is your directory path
i = len(list)
print (i)

base = Path('C:/Users/LENOVO/Desktop/dataset/train/Glioma')
properties = ['contrast','energy','homogeneity','correlation','dissimilarity']
files = base.iterdir()

arr = [[] for _ in range(i)]
imageNames = [" "]*i
lab_arr = [0]*i
counter = 0
counter2 = 0
counter4 = 0
for items in files:
    if items.is_file():
        new_path1 = base.joinpath(items.name)
        a,b = new_path1.name.split('.')
        imageName = a.rsplit('/', 1)[0]
        imageNames[counter4] = imageName
        img = cv2.imread(str(new_path1))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #calculating lbp features
        feat_lbp = local_binary_pattern(gray_image, 8, 1, 'uniform')
        lbp_hist, _ = np.histogram(feat_lbp, 8)
        lbp_hist = np.array(lbp_hist, dtype=float)
        lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
        lbp_energy = np.sum(lbp_prob ** 2)
        lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))

        #calculating gabor features
        gaborFilt_real, gaborFilt_imag = gabor(gray_image, frequency=0.6)
        gaborFilt = (gaborFilt_real ** 2 + gaborFilt_imag ** 2) // 2
        gabor_hist, _ = np.histogram(gaborFilt, 8)
        gabor_hist = np.array(gabor_hist, dtype=float)
        gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
        gabor_energy = np.sum(gabor_prob ** 2)
        gabor_entropy = -np.sum(np.multiply(gabor_prob, np.log2(gabor_prob)))

        distances = [1, 2, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = greycomatrix(gray_image,
                            distances=distances,
                            angles=angles,
                            symmetric=True,
                            normed=True)

        arr[counter].append(greycoprops(glcm, properties[0])[0, 0])
        arr[counter].append(greycoprops(glcm, properties[1])[0, 0])
        arr[counter].append(greycoprops(glcm, properties[2])[0, 0])
        arr[counter].append(greycoprops(glcm, properties[3])[0, 0])
        arr[counter].append(greycoprops(glcm, properties[4])[0, 0])
        arr[counter].append(shannon_entropy(gray_image))
        arr[counter].append(-np.sum(glcm * np.log2(glcm + (glcm == 0))))
        arr[counter].append(ndimage.variance(gray_image))
        arr[counter].append(lbp_energy)
        arr[counter].append(lbp_entropy)
        arr[counter].append(gabor_energy)
        arr[counter].append(gabor_entropy)
        #arr[counter].append(mal)
        lab_arr[counter2] = mal

    counter = counter + 1
    counter2 = counter2 + 1
    counter4 = counter4 + 1

list1 = os.listdir('C:/Users/LENOVO/Desktop/dataset/train/Meningioma') # dir is your directory path
f = len(list1)
print (f)
base1 = Path('C:/Users/LENOVO/Desktop/dataset/train/Meningioma')
properties1 = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']
files1 = base1.iterdir()

arr1 = [[] for _ in range(f)]
lab_arr1 = [0]*f
imageNames2 = [" "]*f
counter1 = 0
counter3 = 0
counter5 = 0
non_mal = 0
for items1 in files1:
    if items1.is_file():
        new_path2 = base1.joinpath(items1.name)
        a, b = new_path2.name.split('.')
        imageName1 = a.rsplit('/', 1)[0]
        imageNames2[counter5] = imageName1
        img1 = cv2.imread(str(new_path2))
        gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # calculating lbp features
        feat_lbp = local_binary_pattern(gray_image1, 8, 1, 'uniform')
        lbp_hist, _ = np.histogram(feat_lbp, 8)
        lbp_hist = np.array(lbp_hist, dtype=float)
        lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
        lbp_energy = np.sum(lbp_prob ** 2)
        lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))

        # calculating gabor features
        gaborFilt_real, gaborFilt_imag = gabor(gray_image1, frequency=0.6)
        gaborFilt = (gaborFilt_real ** 2 + gaborFilt_imag ** 2) // 2
        gabor_hist, _ = np.histogram(gaborFilt, 8)
        gabor_hist = np.array(gabor_hist, dtype=float)
        gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
        gabor_energy = np.sum(gabor_prob ** 2)
        gabor_entropy = -np.sum(np.multiply(gabor_prob, np.log2(gabor_prob)))


        distances = [1, 2, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = greycomatrix(gray_image1,
                                distances=distances,
                                angles=angles,
                                symmetric=True,
                                normed=True)

        arr1[counter1].append(greycoprops(glcm, properties[0])[0, 0])
        arr1[counter1].append(greycoprops(glcm, properties[1])[0, 0])
        arr1[counter1].append(greycoprops(glcm, properties[2])[0, 0])
        arr1[counter1].append(greycoprops(glcm, properties[3])[0, 0])
        arr1[counter1].append(greycoprops(glcm, properties[4])[0, 0])
        arr1[counter1].append(shannon_entropy(gray_image1))
        arr1[counter1].append(-np.sum(glcm * np.log2(glcm + (glcm == 0))))
        arr1[counter1].append(ndimage.variance(gray_image1))
        arr1[counter1].append(lbp_energy)
        arr1[counter1].append(lbp_entropy)
        arr1[counter1].append(gabor_energy)
        arr1[counter1].append(gabor_entropy)
        #arr1[counter1].append(non_mal)
        lab_arr1[counter3] = non_mal

    counter1 = counter1 + 1
    counter3 = counter3 + 1
    counter5 = counter5 + 1



final = imageNames + imageNames2

label_vector = np.array(lab_arr + lab_arr1)
label_vector.shape = (i+f,1)
feature_vector = np.array(arr + arr1)
np.set_printoptions(suppress=True)

#for l in feature_vector:

    #print(l)


x = feature_vector
y = label_vector

from sklearn.datasets.samples_generator import make_blobs

features = ['Contrast', 'Energy', 'Homogeneity', 'Correlation', 'Dissimilarity','Shannon Entropy', 'Simple Entropy', 'Variance']

r = 0
sp = 131

from sklearn.model_selection import train_test_split
from neupy import algorithms
from sklearn import metrics

#for std in [200]:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    #model = algorithms.PNN(std=std,verbose=False)#LinearSVC(random_state=None)
    #model.train(x_train,y_train)
    #print(".......................")

    #y_pred = model.predict(x_test)
    #print(y_pred)
    #print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
    #print("Precision:",metrics.precision_score(y_test,y_pred))
    #print("Recall:",metrics.recall_score(y_test,y_pred))
#y_predicted = model.predict(x_test)
#print(metrics.accuracy_score(y_test, y_predicted))
#model.fit(x_train,y_train)

model = XGBClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = metrics.accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

list2 = os.listdir('C:/Users/LENOVO/Desktop/dataset/test') # dir is your directory path
z = len(list2)
print (z)
base3 = Path('C:/Users/LENOVO/Desktop/dataset/test')
properties2= ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']
files2 = base3.iterdir()

arr2 = [[] for _ in range(z)]

imageNames2 = [" "]*z
counter6 = 0
counter7 = 0
for items2 in files2:
    if items2.is_file():
        new_path3 = base3.joinpath(items2.name)
        a, b = new_path3.name.split('.')
        imageName1 = a.rsplit('/', 1)[0]
        imageNames2[counter7] = imageName1
        img4 = cv2.imread(str(new_path3))
        gray_image1 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

        #calculating lbp features
        feat_lbp = local_binary_pattern(gray_image1, 8, 1, 'uniform')
        lbp_hist, _ = np.histogram(feat_lbp, 8)
        lbp_hist = np.array(lbp_hist, dtype=float)
        lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
        lbp_energy = np.sum(lbp_prob ** 2)
        lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))

        #calculating gabor features
        gaborFilt_real, gaborFilt_imag = gabor(gray_image1, frequency=0.6)
        gaborFilt = (gaborFilt_real ** 2 + gaborFilt_imag ** 2) // 2
        gabor_hist, _ = np.histogram(gaborFilt, 8)
        gabor_hist = np.array(gabor_hist, dtype=float)
        gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
        gabor_energy = np.sum(gabor_prob ** 2)
        gabor_entropy = -np.sum(np.multiply(gabor_prob, np.log2(gabor_prob)))

        distances = [1, 2, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = greycomatrix(gray_image1,
                                distances=distances,
                                angles=angles,
                                symmetric=True,
                                normed=True)

        arr2[counter6].append(greycoprops(glcm, properties[0])[0, 0])
        arr2[counter6].append(greycoprops(glcm, properties[1])[0, 0])
        arr2[counter6].append(greycoprops(glcm, properties[2])[0, 0])
        arr2[counter6].append(greycoprops(glcm, properties[3])[0, 0])
        arr2[counter6].append(greycoprops(glcm, properties[4])[0, 0])
        arr2[counter6].append(shannon_entropy(gray_image1))
        arr2[counter6].append(-np.sum(glcm * np.log2(glcm + (glcm == 0))))
        arr2[counter6].append(ndimage.variance(gray_image1))
        arr2[counter6].append(lbp_energy)
        arr2[counter6].append(lbp_entropy)
        arr2[counter6].append(gabor_energy)
        arr2[counter6].append(gabor_entropy)

    counter6 = counter6 + 1
    counter7 = counter7+1

test_features = np.array(arr2)

