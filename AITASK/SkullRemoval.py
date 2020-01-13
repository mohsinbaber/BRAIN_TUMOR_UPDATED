# k.png is the file with dimensions similar to original image. We will save segmented image in its original dimensions.
import cv2
import os,io
from subprocess import call
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog
import pyodbc
import tkinter.ttk as new
from tkcalendar import DateEntry,Calendar
from tkinter import messagebox
from PIL import Image
from tkinter import ttk
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import xgboost
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from scipy import ndimage
from skimage.filters import gabor
from matplotlib import pyplot as plt


conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-U8LFE56;'
                      'Database=BrainTumor;'
                      'Trusted_Connection=yes;')
flag = -1
tumor = -1
root = Tk()
refPt = []
cropping = False
menubar = Menu(root)
root.config(menu=menubar)

def rec():
    # initial first frame of image
    img_re2 = Image.open('rec.jpg')
    img_re2 = img_re2.resize((120, 120))
    img_re2.save("abc2.ppm", "ppm")
    init_img = PhotoImage(file='abc2.ppm')
    init_label = Label(image=init_img)
    init_label.image = init_img
    init_label.place(x=120, y=150)
    text = Label(root, text="Brain Image-Axial View")
    text.place(x=120, y=280)

    # initial second frame of image
    init_img2 = PhotoImage(file='abc2.ppm')
    init_label2 = Label(image=init_img2)
    init_label2.image = init_img2
    init_label2.place(x=320, y=150)
    text = Label(root, text="Transformed Image-Color Map")
    text.place(x=295, y=280)

    #initial third frame of image
    init_img2 = PhotoImage(file='abc2.ppm')
    init_label2 = Label(image=init_img2)
    init_label2.image = init_img2
    init_label2.place(x=520, y=150)
    text = Label(root, text="Axial View-Seg1")
    text.place(x=540, y=280)

    #initial fourth frame of image
    init_img2 = PhotoImage(file='abc2.ppm')
    init_label2 = Label(image=init_img2)
    init_label2.image = init_img2
    init_label2.place(x=720, y=150)
    text = Label(root, text="Axial View-Seg2")
    text.place(x=740, y=280)

rec()
separator = new.Separator(root,orient="vertical")
separator.place(x=920,y=100,height=520)

separator2 = new.Separator(root,orient="horizontal")
separator2.place(x=50,y=100,width=870)

separator3 = new.Separator(root,orient="vertical")
separator3.place(x=50,y=101,height=520)

separator4 = new.Separator(root,orient="horizontal")
separator4.place(x=50,y=620,width=871)

adpat = Label(root,text="Add Patient Information",font=(None,15,'underline'))
adpat.place(x=1000,y=65)

idd = Label(root,text="Pat. ID:",font=(None,10))
idd.place(x=930,y=100)
iddText =  Text(root,width=35,height=0)
iddText.place(x=995,y=100)
iddText.config(state="disabled")

def get_id():
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(CAST(SUBSTRING(ID, 4, len(ID)-3) AS int)) FROM BrainTumor.dbo.PatientInfo')
    result = cursor.fetchall()
    cou = 0
    idd = ""
    for row in result:
        idd = row[0]
    if (idd == None):
        idAdd = "2001"
    else:
        # a, numb = idd.split('-')
        idFinal = int(idd) + 1
        idAdd = "200" + str(idFinal)
    return idAdd

iddText.config(state="normal")
iddText.insert(END,get_id())
iddText.config(state="disabled")

name = Label(root,text="Name:",font=(None,10))
name.place(x=930,y=140)
nameText = Text(root,width=35,height=0)
nameText.place(x=995,y=140)

age = Label(root,text="Date of Birth:",font=(None,10))
age.place(x=930,y=180)

dayVar=StringVar(root)
monVar=StringVar(root)
yearVar=StringVar(root)
dayVar.set('Day')
monVar.set('Month')
yearVar.set('Year')
dropdown1 = ttk.Combobox(root,width=8,textvariable=dayVar,state="readonly",values=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'])
dropdown2 = ttk.Combobox(root,width=8,textvariable=monVar,state="readonly",values=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
dropdown3 = ttk.Combobox(root,width=8,textvariable=yearVar,state="readonly",values=['2019','2018','2017','2016','2015','2014','2013','2012','2011','2010','2009','2008','2007','2006','2005','2004','2003','2002','2001','2000','1999','1998','1997','1996','1995','1994','1993','1992','1991','1990','1989','1988','1987','1986','1985','1984','1983','1982','1981','1980','1979','1978','1977','1976','1975','1974','1973','1972','1971','1970','1969','1968','1967','1966','1965','1964','1963','1962','1961','1960','1959','1958','1957','1956','1955','1954','1953','1952','1951','1950','1949','1948','1947','1946','1945','1944','1943','1942','1941','1940','1939','1938','1937','1936','1935','1934','1933','1932','1931','1930','1929','1928','1927','1926','1925','1924','1923','1922','1921','1920','1919','1918','1917','1916','1915','1914','1913','1912','1911','1910'])
dropdown1.place(x=1020,y=180)
dropdown2.place(x=1115,y=180)
dropdown3.place(x=1210,y=180)

gender = Label(root,text="Gender:",font=(None,10))
gender.place(x=930,y=220)

tkvar = StringVar(root)
tkvar.set("L")
rad1 = Radiobutton(root,text="Male",variable=tkvar,value="Male")
rad1.place(x=995,y=220)
rad2 = Radiobutton(root,text="Female",variable=tkvar,value="Female")
rad2.place(x=1065,y=220)
rad3= Radiobutton(root,text="Others",variable=tkvar,value="Others")
rad3.place(x=1145,y=220)

contact = Label(root,text="Contact:",font=(None,10))
contact.place(x=930,y=260)
contactText = Text(root,width=35,height=0)
contactText.place(x=995,y=260)

dov = Label(root,text="Date of Appointment:",font=(None,10))
dov.place(x=930,y=300)
cal = DateEntry(root, width=33, background='darkblue',
                    foreground='white', borderwidth=2)
cal.place(x=1060,y=300)

cancer = Label(root,text="Cancer Type:",font=(None,10))
cancer.place(x=930,y=340)
cancers = {'Malignant(Glioblastoma/Glioma)', 'Benign(Meningioma)','Normal Brain(Non-Cancerous)'}
cancerVar = StringVar(root)
cancerVar.set('Choose Tumor Type')
dropdown = OptionMenu(root,cancerVar,*cancers)
dropdown.config(width=38)
dropdown.place(x=1010,y=337)

stage = Label(root,text="Stage:",font=(None,10))
stage.place(x=930,y=380)
stages = ['Normal Brain(Non-Cancerous)','Stage 1','Stage 2','Stage 3','Stage 4']
stageVar = StringVar(root)
stageVar.set('Choose Tumor Stage')
dropdown = OptionMenu(root,stageVar,*stages)
dropdown.config(width=38)
dropdown.place(x=1010,y=375)

pres = Label(root,text="Prescription:",font=(None,10))
pres.place(x=930,y=410)
presText = Text(root,width=44,height=7)
presText.place(x=930,y=430)


# upload initial image selected by user
def upload_img(file):
        if(file != ""):
            img_re = Image.open(file)
            img_re = img_re.resize((120, 120))
            img_re.save("abc.ppm", "ppm")
            img_re.save("tumor.ppm","ppm")
            imga = PhotoImage(file="abc.ppm")
            imgF = Label(image=imga)
            imgF.image = imga
            imgF.place(x=120, y=150)
            global flag
            flag = 0

            #transformed image
            im_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
            t_img = cv2.resize(im_color,(120, 120))
            cv2.imwrite("transform.ppm",t_img)
            t_imga = PhotoImage(file="transform.ppm")
            imgL = Label(image=t_imga)
            imgL.image = t_imga
            imgL.place(x=320,y=150)

            #segmentation 1
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            plt.hist(gray.ravel(), 256)

            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

            colormask = np.zeros(img.shape, dtype=np.uint8)
            colormask[thresh != 0] = np.array((0, 0, 255))
            blended = cv2.addWeighted(img, 0.7, colormask, 0.1, 0)

            ret, markers = cv2.connectedComponents(thresh)

            # Get the area taken by each component. Ignore label 0 since this is the background.
            marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
            # Get label of largest component by area
            largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
            # Get pixels which correspond to the brain
            brain_mask = markers == largest_component

            brain_out = img.copy()
            # In a copy of the original image, clear those pixels that don't correspond to the brain
            brain_out[brain_mask == False] = (0, 0, 0)

            brain_mask = np.uint8(brain_mask)
            kernel = np.ones((8, 8), np.uint8)
            closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

            cv2.imwrite('output.png', img)
            brain_out = cv2.imread('output.png')
            # In a copy of the original image, clear those pixels that don't correspond to the brain
            brain_out[closing == False] = (0, 0, 0)
            cv2.imwrite('out2.png', brain_out)

            global filename2
            th, dst = cv2.threshold(cv2.imread('out2.png'), 127, 255, cv2.THRESH_TOZERO)
            filename2 = 'seg1.png'
            seg_re = cv2.resize(dst,(120, 120))
            cv2.imwrite(filename2, seg_re)
            seg1_imga = PhotoImage(file="seg1.png")
            imgS = Label(image=seg1_imga)
            imgS.image = seg1_imga
            imgS.place(x=520, y=150)


            #segmentation 2
            def cvt_image_colorspace(image, colorspace=cv2.COLOR_BGR2GRAY):
                return cv2.cvtColor(image, colorspace)

            def median_filtering(image, kernel_size=3):

                return cv2.medianBlur(image, kernel_size)

            def apply_threshold(image, **kwargs):

                threshold_method = kwargs['threshold_method']
                max_value = kwargs['pixel_value']
                threshold_flag = kwargs.get('threshold_flag', None)
                if threshold_flag is not None:
                    ret, thresh1 = cv2.adaptiveThreshold(image, max_value, threshold_method, cv2.THRESH_BINARY,
                                                         kwargs['block_size'], kwargs['const'])
                else:
                    ret, thresh1 = cv2.threshold(image, kwargs['threshold'], max_value, threshold_method)
                return thresh1

            def sobel_filter(img, x, y, kernel_size=3):
                return cv2.Sobel(img, cv2.CV_8U, x, y, ksize=kernel_size)

            image = cv2.imread(file)

            # Step one - grayscale the image
            grayscale_img = cvt_image_colorspace(image)

            # Step two - filter out image
            median_filtered = median_filtering(grayscale_img, 5)

            # Step 3a - apply Sobel filter
            img_sobelx = sobel_filter(median_filtered, 1, 0)
            img_sobely = sobel_filter(median_filtered, 0, 1)

            # Adding mask to the image
            img_sobel = img_sobelx + img_sobely + grayscale_img

            # Step 3b - apply erosion + dilation
            # apply erosion and dilation to show only the part of the image having more intensity - tumor region
            # that we want to extract
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            erosion = cv2.morphologyEx(median_filtered, cv2.MORPH_ERODE, kernel)

            dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel)

            new_thresholding = apply_threshold(dilation, **{"threshold": 160,
                                                            "pixel_value": 255,
                                                            "threshold_method": cv2.THRESH_BINARY})

            s2_img = cv2.resize(new_thresholding, (120, 120))
            cv2.imwrite(r'seg2.png', s2_img)
            s2_imga = PhotoImage(file="seg2.png")
            imgSe = Label(image=s2_imga)
            imgSe.image = s2_imga
            imgSe.place(x=720, y=150)

            img_rgb = cv2.imread('seg1.png')
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            template = cv2.imread('seg2.png', 0)
            w, h = template.shape[::-1]

            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

            cv2.imwrite('res.png', img_rgb)
        else:
            messagebox.showerror("Error","Image not selected!")

        train_data()

def train_data():
    # training data and saving it in pickle file

    dir = 'tumor_features.csv'
    col_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'Class']
    dataset = pd.read_csv(dir, names=col_names)
    X = dataset.drop('Class', axis=1)
    y = dataset['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)

    # save the model to disk
    filename = 'trained_model.pkl'
    joblib.dump(model, filename)

    # load the model from disk
    loaded_model = joblib.load(filename)

    y_pred = loaded_model.predict(X_test)

    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    predictions = [round(value) for value in y_pred]

    accuracy = metrics.accuracy_score(y_test, predictions)
    mse = mean_absolute_error(y_test, y_pred)

def show_result():
    cont_Text.config(state="normal")
    ener_Text.config(state="normal")
    homo_Text.config(state="normal")
    corr_Text.config(state="normal")
    diss_Text.config(state="normal")
    shen_Text.config(state="normal")
    sien_Text.config(state="normal")
    var_Text.config(state="normal")
    lbpen_Text.config(state="normal")
    lbpent_Text.config(state="normal")
    gaen_Text.config(state="normal")
    gaent_Text.config(state="normal")
    tot_Text.config(state="normal")
    pa_Text.config(state="normal")
    er_Text.config(state="normal")
    cont_Text.delete("1.0", "end-1c")
    ener_Text.delete("1.0", "end-1c")
    homo_Text.delete("1.0", "end-1c")
    corr_Text.delete("1.0", "end-1c")
    diss_Text.delete("1.0", "end-1c")
    shen_Text.delete("1.0", "end-1c")
    sien_Text.delete("1.0", "end-1c")
    var_Text.delete("1.0", "end-1c")
    lbpen_Text.delete("1.0", "end-1c")
    lbpent_Text.delete("1.0", "end-1c")
    gaen_Text.delete("1.0", "end-1c")
    gaent_Text.delete("1.0", "end-1c")
    tot_Text.delete("1.0", "end-1c")
    pa_Text.delete("1.0", "end-1c")
    er_Text.delete("1.0", "end-1c")

    arr = [[]*12]
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']
    if(flag == 0):
        img = cv2.imread("tumor.ppm")
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # calculating lbp features
        feat_lbp = local_binary_pattern(gray_image, 8, 1, 'uniform')
        lbp_hist, _ = np.histogram(feat_lbp, 8)
        lbp_hist = np.array(lbp_hist, dtype=float)
        lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
        lbp_energy = np.sum(lbp_prob ** 2)
        lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))

        # calculating gabor features
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
        cont_Text.config(state="normal")
        ener_Text.config(state="normal")
        homo_Text.config(state="normal")
        corr_Text.config(state="normal")
        diss_Text.config(state="normal")
        shen_Text.config(state="normal")
        sien_Text.config(state="normal")
        var_Text.config(state="normal")
        lbpen_Text.config(state="normal")
        lbpent_Text.config(state="normal")
        gaen_Text.config(state="normal")
        gaent_Text.config(state="normal")

        cont_Text.insert(END, str((greycoprops(glcm, properties[0])[0, 0])*100))
        ener_Text.insert(END, str((greycoprops(glcm, properties[1])[0, 0])*100))
        homo_Text.insert(END, str((greycoprops(glcm, properties[2])[0, 0])*100))
        corr_Text.insert(END,str((greycoprops(glcm, properties[3])[0, 0])*100))
        diss_Text.insert(END,str((greycoprops(glcm, properties[4])[0, 0])*100))
        shen_Text.insert(END,str(shannon_entropy(gray_image)*100))
        sien_Text.insert(END,str((-np.sum(glcm * np.log2(glcm + (glcm == 0))))*100))
        var_Text.insert(END,str(ndimage.variance(gray_image)*100))
        lbpen_Text.insert(END,lbp_energy*100)
        lbpent_Text.insert(END,lbp_entropy*100)
        gaen_Text.insert(END,gabor_energy*100)
        gaent_Text.insert(END, gabor_entropy * 100)
        cont_Text.config(state="disabled")
        ener_Text.config(state="disabled")
        homo_Text.config(state="disabled")
        corr_Text.config(state="disabled")
        diss_Text.config(state="disabled")
        shen_Text.config(state="disabled")
        sien_Text.config(state="disabled")
        var_Text.config(state="disabled")
        lbpen_Text.config(state="disabled")
        lbpent_Text.config(state="disabled")
        gaen_Text.config(state="disabled")
        gaent_Text.config(state="disabled")

        arr[0].append(greycoprops(glcm, properties[0])[0, 0])
        arr[0].append(greycoprops(glcm, properties[1])[0, 0])
        arr[0].append(greycoprops(glcm, properties[2])[0, 0])
        arr[0].append(greycoprops(glcm, properties[3])[0, 0])
        arr[0].append(greycoprops(glcm, properties[4])[0, 0])
        arr[0].append(shannon_entropy(gray_image))
        arr[0].append(-np.sum(glcm * np.log2(glcm + (glcm == 0))))
        arr[0].append(ndimage.variance(gray_image))
        arr[0].append(lbp_energy)
        arr[0].append(lbp_entropy)
        arr[0].append(gabor_energy)
        arr[0].append(gabor_entropy)

        col_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        np.savetxt("test_tumor_features.csv", arr, delimiter=",")
        dir = "test_tumor_features.csv"
        dataset = pd.read_csv(dir, names=col_names)
        dir2 = "tumor_features.csv"
        col_names1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12','Class']
        train = pd.read_csv(dir2,names=col_names1)
        y = train['Class']


        X = dataset[col_names]


        with open('trained_model.pkl', 'rb') as saved:
            loaded_model = joblib.load(saved)
        y_prob = loaded_model.predict_proba(X)
        y_pred = loaded_model.predict(X)
        global tumor
        if(int(y_pred[0]) == 0):
            tumor = 0
            tot_Text.config(background="#B1F5D4")
            tot_Text.config(state="normal")
            tot_Text.insert(END,"Benign")
            tot_Text.config(state="disabled")
            pa_Text.config(state="normal")
            pa_Text.insert(END,str((y_prob[0:, 0][0])*100))
            pa_Text.config(state="disabled")
            er_Text.config(state="normal")
            er_Text.insert(END,str(1.0 - y_prob[0:, 0][0]))
            er_Text.config(state="disabled")
            #print("Probability: "+str((y_prob[0:, 0][0])*100)+"\nError Rate: "+str(1.0 - y_prob[0:, 0][0]))
        elif(int(y_pred[0]) == 1):
            tumor = 1
            tot_Text.config(background="#F5B1B1")
            tot_Text.config(state="normal")
            tot_Text.insert(END, "Malignant")
            tot_Text.config(state="disabled")
            pa_Text.config(state="normal")
            pa_Text.insert(END, str((y_prob[0:, 1][0]) * 100))
            pa_Text.config(state="disabled")
            er_Text.config(state="normal")
            er_Text.insert(END, str(1.0 - y_prob[0:, 1][0]))
            er_Text.config(state="disabled")
            #print("Probability: "+str((y_prob[0:, 1][0])*100)+"\nError Rate: "+str(1.0 - y_prob[0:, 1][0]))

    elif(flag == 1):
        img = cv2.imread("browse_database.ppm")
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # calculating lbp features
        feat_lbp = local_binary_pattern(gray_image, 8, 1, 'uniform')
        lbp_hist, _ = np.histogram(feat_lbp, 8)
        lbp_hist = np.array(lbp_hist, dtype=float)
        lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
        lbp_energy = np.sum(lbp_prob ** 2)
        lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))

        # calculating gabor features
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

        cont_Text.config(state="normal")
        ener_Text.config(state="normal")
        homo_Text.config(state="normal")
        corr_Text.config(state="normal")
        diss_Text.config(state="normal")
        shen_Text.config(state="normal")
        sien_Text.config(state="normal")
        var_Text.config(state="normal")
        lbpen_Text.config(state="normal")
        lbpent_Text.config(state="normal")
        gaen_Text.config(state="normal")
        gaent_Text.config(state="normal")
        cont_Text.insert(END, str((greycoprops(glcm, properties[0])[0, 0]) * 100))
        ener_Text.insert(END, str((greycoprops(glcm, properties[1])[0, 0]) * 100))
        homo_Text.insert(END, str((greycoprops(glcm, properties[2])[0, 0]) * 100))
        corr_Text.insert(END, str((greycoprops(glcm, properties[3])[0, 0]) * 100))
        diss_Text.insert(END, str((greycoprops(glcm, properties[4])[0, 0]) * 100))
        shen_Text.insert(END, str(shannon_entropy(gray_image) * 100))
        sien_Text.insert(END, str((-np.sum(glcm * np.log2(glcm + (glcm == 0)))) * 100))
        var_Text.insert(END, str(ndimage.variance(gray_image) * 100))
        lbpen_Text.insert(END, lbp_energy * 100)
        lbpent_Text.insert(END, lbp_entropy * 100)
        gaen_Text.insert(END, gabor_energy * 100)
        gaent_Text.insert(END, gabor_entropy * 100)
        cont_Text.config(state="disabled")
        ener_Text.config(state="disabled")
        homo_Text.config(state="disabled")
        corr_Text.config(state="disabled")
        diss_Text.config(state="disabled")
        shen_Text.config(state="disabled")
        sien_Text.config(state="disabled")
        var_Text.config(state="disabled")
        lbpen_Text.config(state="disabled")
        lbpent_Text.config(state="disabled")
        gaen_Text.config(state="disabled")
        gaent_Text.config(state="disabled")

        arr[0].append(greycoprops(glcm, properties[0])[0, 0])
        arr[0].append(greycoprops(glcm, properties[1])[0, 0])
        arr[0].append(greycoprops(glcm, properties[2])[0, 0])
        arr[0].append(greycoprops(glcm, properties[3])[0, 0])
        arr[0].append(greycoprops(glcm, properties[4])[0, 0])
        arr[0].append(shannon_entropy(gray_image))
        arr[0].append(-np.sum(glcm * np.log2(glcm + (glcm == 0))))
        arr[0].append(ndimage.variance(gray_image))
        arr[0].append(lbp_energy)
        arr[0].append(lbp_entropy)
        arr[0].append(gabor_energy)
        arr[0].append(gabor_entropy)

        col_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        np.savetxt("test_tumor_features.csv", arr, delimiter=",")
        dir = "test_tumor_features.csv"
        dataset = pd.read_csv(dir, names=col_names)
        dir2 = "tumor_features.csv"
        col_names1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'Class']
        train = pd.read_csv(dir2, names=col_names1)
        y = train['Class']

        X = dataset[col_names]

        with open('trained_model.pkl', 'rb') as saved:
            loaded_model = joblib.load(saved)
        y_prob = loaded_model.predict_proba(X)
        y_pred = loaded_model.predict(X)

        if (int(y_pred[0]) == 0):
            tumor = 0
            tot_Text.config(background="#B1F5D4")
            tot_Text.config(state="normal")
            tot_Text.insert(END, "Benign")
            tot_Text.config(state="disabled")
            pa_Text.config(state="normal")
            pa_Text.insert(END, str((y_prob[0:, 0][0]) * 100))
            pa_Text.config(state="disabled")
            er_Text.config(state="normal")
            er_Text.insert(END, str(1.0 - y_prob[0:, 0][0]))
            er_Text.config(state="disabled")
        elif (int(y_pred[0]) == 1):
            tumor = 1
            tot_Text.config(background="#F5B1B1")
            tot_Text.config(state="normal")
            tot_Text.insert(END, "Benign")
            tot_Text.config(state="disabled")
            pa_Text.config(state="normal")
            pa_Text.insert(END, str((y_prob[0:, 1][0]) * 100))
            pa_Text.config(state="disabled")
            er_Text.config(state="normal")
            er_Text.insert(END, str(1.0 - y_prob[0:, 1][0]))
            er_Text.config(state="disabled")

def upload_img2(file):
    if(file !=""):
        # transformed image
        im_gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        t_img = cv2.resize(im_color, (120, 120))
        cv2.imwrite("transform.ppm", t_img)
        t_imga = PhotoImage(file="transform.ppm")
        imgL = Label(image=t_imga)
        imgL.image = t_imga
        imgL.place(x=320, y=150)

        # segmentation 1
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.hist(gray.ravel(), 256)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        colormask = np.zeros(img.shape, dtype=np.uint8)
        colormask[thresh != 0] = np.array((0, 0, 255))
        blended = cv2.addWeighted(img, 0.7, colormask, 0.1, 0)

        ret, markers = cv2.connectedComponents(thresh)

        # Get the area taken by each component. Ignore label 0 since this is the background.
        marker_area = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]
        # Get label of largest component by area
        largest_component = np.argmax(marker_area) + 1  # Add 1 since we dropped zero above
        # Get pixels which correspond to the brain
        brain_mask = markers == largest_component

        brain_out = img.copy()
        # In a copy of the original image, clear those pixels that don't correspond to the brain
        brain_out[brain_mask == False] = (0, 0, 0)

        brain_mask = np.uint8(brain_mask)
        kernel = np.ones((8, 8), np.uint8)
        closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite('output.png', img)
        brain_out = cv2.imread('output.png')
        # In a copy of the original image, clear those pixels that don't correspond to the brain
        brain_out[closing == False] = (0, 0, 0)
        cv2.imwrite('out2.png', brain_out)

        global filename2
        th, dst = cv2.threshold(cv2.imread('out2.png'), 127, 255, cv2.THRESH_TOZERO)
        filename2 = 'seg1.png'
        seg_re = cv2.resize(dst, (120, 120))
        cv2.imwrite(filename2, seg_re)
        seg1_imga = PhotoImage(file="seg1.png")
        imgS = Label(image=seg1_imga)
        imgS.image = seg1_imga
        imgS.place(x=520, y=150)

        # segmentation 2
        def cvt_image_colorspace(image, colorspace=cv2.COLOR_BGR2GRAY):
            return cv2.cvtColor(image, colorspace)

        def median_filtering(image, kernel_size=3):

            return cv2.medianBlur(image, kernel_size)

        def apply_threshold(image, **kwargs):

            threshold_method = kwargs['threshold_method']
            max_value = kwargs['pixel_value']
            threshold_flag = kwargs.get('threshold_flag', None)
            if threshold_flag is not None:
                ret, thresh1 = cv2.adaptiveThreshold(image, max_value, threshold_method, cv2.THRESH_BINARY,
                                                     kwargs['block_size'], kwargs['const'])
            else:
                ret, thresh1 = cv2.threshold(image, kwargs['threshold'], max_value, threshold_method)
            return thresh1

        def sobel_filter(img, x, y, kernel_size=3):
            return cv2.Sobel(img, cv2.CV_8U, x, y, ksize=kernel_size)

        image = cv2.imread(file)

        # Step one - grayscale the image
        grayscale_img = cvt_image_colorspace(image)

        # Step two - filter out image
        median_filtered = median_filtering(grayscale_img, 5)

        # Step 3a - apply Sobel filter
        img_sobelx = sobel_filter(median_filtered, 1, 0)
        img_sobely = sobel_filter(median_filtered, 0, 1)

        # Adding mask to the image
        img_sobel = img_sobelx + img_sobely + grayscale_img


        # Step 3b - apply erosion + dilation
        # apply erosion and dilation to show only the part of the image having more intensity - tumor region
        # that we want to extract
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        erosion = cv2.morphologyEx(median_filtered, cv2.MORPH_ERODE, kernel)

        dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel)

        # Step 4 - apply thresholding

        new_thresholding = apply_threshold(dilation, **{"threshold": 160,
                                                        "pixel_value": 255,
                                                        "threshold_method": cv2.THRESH_BINARY})

        s2_img = cv2.resize(new_thresholding, (120, 120))
        cv2.imwrite(r'seg2.png', s2_img)
        s2_imga = PhotoImage(file="seg2.png")
        imgSe = Label(image=s2_imga)
        imgSe.image = s2_imga
        imgSe.place(x=720, y=150)

        img_rgb = cv2.imread('seg1.png')
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('seg2.png', 0)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

        cv2.imwrite('res.png', img_rgb)
    else:
        messagebox.showerror("Error", "Image not selected!")

    train_data()

# upload segmented image in the other image box
def upload_seg_img(file):
    img_re = Image.open(file)
    img_re = img_re.resize((300, 300))
    img_re.save("abc.ppm", "ppm")
    imga = PhotoImage(file="abc.ppm")
    imgF = Label(image=imga)
    imgF.image = imga
    imgF.place(x=550, y=50)


def browse():
    global fname
    fname = filedialog.askopenfilename()
    upload_img(fname)

subMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="New Brain Image", command=browse)

def browse_database():
    def search_img():
        sql = "SELECT PSI FROM BrainTumor.dbo.PatientInfo WHERE ID=?"
        val = srchBar.get("1.0", "end-1c")
        cursor = conn.cursor()
        cursor.execute(sql,val)
        res = cursor.fetchall()
        if (srchBar.get("1.0", "end-1c").startswith("PH-") == True):
            for row in res:
                image = io.BytesIO(row[0])
                img_re = Image.open(image)
                img_re = img_re.resize((120, 120))
                img_re.save("browse_database.ppm", "ppm")
                imga = PhotoImage(file="browse_database.ppm")
                imgF = Label(image=imga)
                imgF.image = imga
                imgF.place(x=120, y=150)
                global flag
                flag=1
                upload_img2("browse_database.ppm")
        else:
            messagebox.showerror("Error", "Patient ID must always start with \"PH-\"")
        srchBar.destroy()
        srchBtn.destroy()
    srchBar = Text(root,width=15,height=0)
    srchBar.place(x=120,y=300)
    srchBtn = Button(root,text="Search",width=16,height=0, command=search_img)
    srchBtn.place(x=120,y=330)

loadMenu1 = Menu(subMenu)
openMenu = Menu(loadMenu1)
openMenu.add_command(label="From desktop",command=browse)
openMenu.add_command(label="From database",comman=browse_database)
loadMenu1.add_cascade(label="Brain image",menu=openMenu)
subMenu.add_cascade(label="Open",menu=loadMenu1)

def abt_us():
    tkinter.messagebox.showinfo("About Us", "This application segments and detect brain tumors in an MRI Image. ")

def validateContact(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# saves text and segmented image files
def save_data():
    try:
        ans = messagebox.askyesno("Saving Data","Save Data?")
        if(ans == True):
            cursor = conn.cursor()
            cursor.execute('SELECT MAX(CAST(SUBSTRING(ID, 4, len(ID)-3) AS int)) FROM BrainTumor.dbo.PatientInfo')
            result = cursor.fetchall()
            cou = 0
            idd=""
            for row in result:
                idd =row[0]
            if (idd == None):
                idAdd = "2001"
            else:
                idFinal = int(idd) + 1
                idAdd = "200" + str(idFinal)
            dob = dayVar.get() + "-" + monVar.get() + "-" + yearVar.get()
            a, b, year = dob.split("-")
            #calyear = datetime.datetime.now().year - int(year)
            if(len(nameText.get("1.0", "end-1c")) == 0 or dayVar.get() == 'Day' or monVar.get()=='Month' or yearVar.get()=='Year' or str(tkvar.get()) == "L" or len(contactText.get("1.0", "end-1c")) == 0 or len(str(cal.get_date())) == 0 or cancerVar.get() == 'Choose Tumor Type' or stageVar.get() == 'Choose Tumor Stage' or len(presText.get("1.0", "end-1c")) == 0):
                messagebox.showerror("Error", "One or more fields are empty!")
            else:
                if(os.path.exists('tumor.ppm') == True):
                    if(validateContact(contactText.get("1.0", "end-1c")) == True and len(contactText.get("1.0", "end-1c")) == 11and (contactText.get("1.0", "end-1c").startswith("03") == True or contactText.get("1.0", "end-1c").startswith("0423"))):
                        image = Image.open('tumor.ppm')
                        blob=open('tumor.ppm', 'rb').read()
                        cursor1 = conn.cursor()
                        sql = "INSERT INTO BrainTumor.dbo.PatientInfo (ID,name,dob,gender,contact,DOV,cancer,PSI,stage,prescription)VALUES (?,?,?,?,?,?,?,?,?,?)"
                        val = (idAdd,nameText.get("1.0", "end-1c"),dob,
                               tkvar.get(),contactText.get("1.0", "end-1c"),str(cal.get_date()),cancerVar.get(),blob,stageVar.get(),presText.get("1.0", "end-1c"))
                        cursor1.execute(sql,val)

                        cursor2 = conn.cursor()
                        sql1 = "INSERT INTO BrainTumor.dbo.PatientHist (ID,name,dob,gender,contact,DOV,cancer,PSI,stage,prescription)VALUES (?,?,?,?,?,?,?,?,?,?)"
                        val1 = (
                            idAdd, nameText.get("1.0", "end-1c"), dob,
                            tkvar.get(), contactText.get("1.0", "end-1c"), str(cal.get_date()), cancerVar.get(), blob,
                            stageVar.get(), presText.get("1.0", "end-1c"))
                        cursor2.execute(sql1, val1)
                        conn.commit()

                        if(messagebox.askokcancel("Saving Data","Data saved successfully!")):
                            nameText.delete("1.0", "end-1c")
                            dayVar.set("Day")
                            monVar.set("Month")
                            yearVar.set("Year")
                            tkvar.set("L")
                            contactText.delete("1.0", "end-1c")
                            cal.delete(0,END)
                            cancerVar.set("Choose Tumor Type")
                            stageVar.set("Choose Tumor Stage")
                            presText.delete("1.0", "end-1c")
                            cont_Text.config(state="normal")
                            ener_Text.config(state="normal")
                            homo_Text.config(state="normal")
                            corr_Text.config(state="normal")
                            diss_Text.config(state="normal")
                            shen_Text.config(state="normal")
                            sien_Text.config(state="normal")
                            var_Text.config(state="normal")
                            lbpen_Text.config(state="normal")
                            lbpent_Text.config(state="normal")
                            gaen_Text.config(state="normal")
                            gaent_Text.config(state="normal")
                            tot_Text.config(state="normal")
                            pa_Text.config(state="normal")
                            er_Text.config(state="normal")
                            cont_Text.delete("1.0", "end-1c")
                            ener_Text.delete("1.0", "end-1c")
                            homo_Text.delete("1.0", "end-1c")
                            corr_Text.delete("1.0", "end-1c")
                            diss_Text.delete("1.0", "end-1c")
                            shen_Text.delete("1.0", "end-1c")
                            sien_Text.delete("1.0", "end-1c")
                            var_Text.delete("1.0", "end-1c")
                            lbpen_Text.delete("1.0", "end-1c")
                            lbpent_Text.delete("1.0", "end-1c")
                            gaen_Text.delete("1.0", "end-1c")
                            gaent_Text.delete("1.0", "end-1c")
                            tot_Text.delete("1.0", "end-1c")
                            pa_Text.delete("1.0", "end-1c")
                            er_Text.delete("1.0", "end-1c")
                            tot_Text.config(background="#FFFFFF")
                            iddText.config(state="normal")
                            iddText.delete("1.0", "end-1c")
                            iddText.insert(END, get_id())
                            iddText.config(state="disabled")
                            rec()
                        else:
                            nameText.delete("1.0", "end-1c")
                            dayVar.set("Day")
                            monVar.set("Month")
                            yearVar.set("Year")
                            tkvar.set("L")
                            contactText.delete("1.0", "end-1c")
                            cal.delete(0, END)
                            cancerVar.set("Choose Tumor Type")
                            stageVar.set("Choose Tumor Stage")
                            presText.delete("1.0", "end-1c")
                            cont_Text.config(state="normal")
                            ener_Text.config(state="normal")
                            homo_Text.config(state="normal")
                            corr_Text.config(state="normal")
                            diss_Text.config(state="normal")
                            shen_Text.config(state="normal")
                            sien_Text.config(state="normal")
                            var_Text.config(state="normal")
                            lbpen_Text.config(state="normal")
                            lbpent_Text.config(state="normal")
                            gaen_Text.config(state="normal")
                            gaent_Text.config(state="normal")
                            tot_Text.config(state="normal")
                            pa_Text.config(state="normal")
                            er_Text.config(state="normal")
                            cont_Text.delete("1.0", "end-1c")
                            ener_Text.delete("1.0", "end-1c")
                            homo_Text.delete("1.0", "end-1c")
                            corr_Text.delete("1.0", "end-1c")
                            diss_Text.delete("1.0", "end-1c")
                            shen_Text.delete("1.0", "end-1c")
                            sien_Text.delete("1.0", "end-1c")
                            var_Text.delete("1.0", "end-1c")
                            lbpen_Text.delete("1.0", "end-1c")
                            lbpent_Text.delete("1.0", "end-1c")
                            gaen_Text.delete("1.0", "end-1c")
                            gaent_Text.delete("1.0", "end-1c")
                            tot_Text.delete("1.0", "end-1c")
                            pa_Text.delete("1.0", "end-1c")
                            er_Text.delete("1.0", "end-1c")
                            tot_Text.config(background="#FFFFFF")
                            iddText.config(state="normal")
                            iddText.delete("1.0", "end-1c")
                            iddText.insert(END, get_id())
                            iddText.config(state="disabled")
                            rec()
                    else:
                        messagebox.showerror("Error","Phone number is either incomplete or incorrect!\nIt must be of format 03xxxxxxxxx or 0423xxxxxxx")
                else:
                    messagebox.showerror("Error","Brain tumor image not selected!")
    except:
        messagebox.showerror("","Technical error occured")

def save_crop_img():
    f = filedialog.asksaveasfile(mode="w", defaultextension=".txt")
    if f is None:
        return
    save_text = textBox.get(1.0, END)
    f.write(save_text)
    a, b = f.name.split('.')
    imageFile = a.rsplit('/', 1)[1]
    newFilename = f.name.rsplit('/', 1)[0]
    cropImg = cv2.imread('ROI.png')
    cv2.imwrite(newFilename + '/' + imageFile + '.png', cropImg)
    f.close()

def crop_work():
    def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            cropping = False

            # draw a rectangle around the region of interest
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)

    image = cv2.imread(fname)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            cv2.destroyAllWindows()
            break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imwrite("ROI.png", roi)
        cv2.waitKey(0)

    # close all open windows
    cv2.destroyAllWindows()


def crop_img():
    crop_wind = tkinter.Toplevel(root)
    crop_wind.geometry('850x650')
    crop_wind.title("Crop Image")

    img_re4 = Image.open(fname)
    img_re4 = img_re4.resize((300, 300))
    img_re4.save("abc4.ppm", "ppm")
    init_img4 = PhotoImage(file='abc4.ppm')
    init_label4 = Label(crop_wind, image=init_img4)
    init_label4.image = init_img4
    init_label4.place(x=100, y=50)
    text4 = Label(crop_wind, text="Original Image")
    text4.place(x=220, y=355)

    crop_work()

    # initial second frame of image
    img_re5 = Image.open('ROI.png')
    img_re5.save("abc5.ppm", "ppm")
    init_img3 = PhotoImage(file='abc5.ppm')
    init_label3 = Label(crop_wind,image=init_img3)
    init_label3.image = init_img3
    init_label3.place(x=450, y=50)
    text5 = Label(crop_wind, text="Cropped Image")
    text5.place(x=550, y=355)

    crop = Button(crop_wind, text="Save", height=2, width=20, command = save_crop_img)
    crop.place(x=450,y=550)

def del_data():
    call(["python","PatientDelDoctor.py"])

def upd_data():
    call(["python", "PatientUpdDoctor.py"])

def log_out():
    if(messagebox.askyesno("Logging Out","Do you wish to logout?") == True):
        root.destroy()
        call(["python", "Login.py"])

subMenu.add_command(label="Crop Image", command=crop_img)
subMenu.add_command(label="Exit", command=log_out)

subMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Patient Information",menu=subMenu)
subMenu.add_command(label="Delete Patient Information", command=del_data)
subMenu.add_command(label="Update Patient Information", command=upd_data)

def view_pat_data():
    call(["python","ViewPatientData.py"])
def view_pat_hist():
    call(["python", "ViewPatientHist.py"])

subMenu = Menu(menubar,tearoff=0)
menubar.add_cascade(label="View",menu=subMenu)
subMenu.add_command(label="Patient Data",command=view_pat_data)
subMenu.add_command(label="Patient History",command=view_pat_hist)

subMenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help", menu=subMenu)
subMenu.add_command(label="About us", command=abt_us)

root.geometry('1300x650')
root.title("Brain Tumor")

img_re2 = Image.open('hos_logo_SkullRemoval.jpg')
img_re2 = img_re2.resize((90, 90))
img_re2.save("hos_logo_SkullRemoval.ppm", "ppm")
init_img = PhotoImage(file='hos_logo_SkullRemoval.ppm')
init_label = Label(image=init_img)
init_label.image = init_img
init_label.place(x=250, y=0)


text = Label(root, text="Department of Neurology and Neuroscience", font=(None,20,'underline'))
text.place(x=350,y=0)

text = Label(root, text="Brain Tumor Detection System", font=(None,20,'underline'))
text.place(x=430,y=50)

viewLabel = Button(root,text="View Patient History",font=(None,10,'underline'),height=0,width=0,bd=0,fg="#0000FF",command=view_pat_hist)
viewLabel.place(x=930,y=550)

saveBtn = Button(root, text="Save Data", height=2, width=20, command=save_data)
saveBtn.place(x=1050, y=576)

featuresLab = Label(root, text="Features", font=(None,15,'underline'))
featuresLab.place(x=430,y=320)

width = 8
#contrast,energy,homogeneity,correlation,dissimilarity,shanon entropy,simple entropy,variance,lbp energy, lbp entropy, gabor energy , gabor entropy
contLab = Label(root, text="Contrast", font=(None,10))
contLab.place(x=155,y=360)
cont_Text = Text(root,width=width,height=0)
cont_Text.place(x=150, y=380)

enerLab = Label(root, text="Energy", font=(None,10))
enerLab.place(x=160,y=430)
ener_Text = Text(root,width=width,height=0)
ener_Text.place(x=150, y=450)

homoLab = Label(root, text="Homogeneity", font=(None,10))
homoLab.place(x=255,y=360)
homo_Text = Text(root,width=width,height=0)
homo_Text.place(x=260, y=380)

corrLab = Label(root, text="Correlation", font=(None,10))
corrLab.place(x=260,y=430)
corr_Text = Text(root,width=width,height=0)
corr_Text.place(x=260, y=450)

dissLab = Label(root, text="Dissimilarity", font=(None,10))
dissLab.place(x=365,y=360)
diss_Text = Text(root,width=width,height=0)
diss_Text.place(x=370, y=380)

shenLab = Label(root, text="Shennon Entropy", font=(None,10))
shenLab.place(x=350,y=430)
shen_Text = Text(root,width=width,height=0)
shen_Text.place(x=370, y=450)

sienLab = Label(root, text="Entropy", font=(None,10))
sienLab.place(x=490,y=360)
sien_Text = Text(root,width=width,height=0)
sien_Text.place(x=480, y=380)

varLab = Label(root, text="Variance", font=(None,10))
varLab.place(x=485,y=430)
var_Text = Text(root,width=width,height=0)
var_Text.place(x=480, y=450)

lbpenLab = Label(root, text="LBP Energy", font=(None,10))
lbpenLab.place(x=585,y=360)
lbpen_Text = Text(root,width=width,height=0)
lbpen_Text.place(x=590, y=380)

lbpentLab = Label(root, text="LBP Entropy", font=(None,10))
lbpentLab.place(x=585,y=430)
lbpent_Text = Text(root,width=width,height=0)
lbpent_Text.place(x=590, y=450)

gaenLab = Label(root, text="Gabor Energy", font=(None,10))
gaenLab.place(x=695,y=360)
gaen_Text = Text(root,width=width,height=0)
gaen_Text.place(x=700, y=380)

gaentLab = Label(root, text="Gabor Entropy", font=(None,10))
gaentLab.place(x=695,y=430)
gaent_Text = Text(root,width=width,height=0)
gaent_Text.place(x=700, y=450)

res = Label(root, text="Results", font=(None,15,'underline'))
res.place(x=435,y=490)

tot = Label(root, text="Type of Tumor:", font=(None,10))
tot.place(x=80,y=535)
tot_Text = Text(root,width=20,height=0)
tot_Text.place(x=170,y=535)

pa = Label(root, text="Prediction Accuracy:", font=(None,10))
pa.place(x=340,y=535)
pa_Text = Text(root,width=20,height=0)
pa_Text.place(x=465,y=535)

er = Label(root, text="Error Rate:", font=(None,10))
er.place(x=630,y=535)
er_Text = Text(root,width=20,height=0)
er_Text.place(x=700,y=535)

show_res_Btn = Button(root, text="Show Results", height=2, width=20, command=show_result)
show_res_Btn.place(x=400, y=575)
def disable_event():
    if(messagebox.askyesno("Logging Out","Do you wish to log out?")):
        root.destroy()
        call(["python", "Login.py"])
    else:
        pass

textBox = Text(root, width=50, height=10)
textBox.place(x=880, y=850)
root.protocol("WM_DELETE_WINDOW", disable_event)
root.resizable(0,0)
root.mainloop()