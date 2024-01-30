# Standard imports
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import colorsys
import base64
from io import BytesIO

######################
#DECLARE BLOB DETECTOR
######################
params = cv2.SimpleBlobDetector_Params() 
# Set Area filtering parameters 
params.filterByArea = True
params.minArea = 40
params.minThreshold = 20;    # the graylevel of images
params.maxThreshold = 255;
params.filterByColor = True
params.blobColor = 255
# Set Circularity filtering parameters 
params.filterByCircularity = False 
params.minCircularity = 0.9
# Set Convexity filtering parameters 
params.filterByConvexity = False
params.minConvexity = 0.2
# Set inertia filtering parameters 
params.filterByInertia = False
params.minInertiaRatio = 0.01
# Create a detector with the parameters 
detector = cv2.SimpleBlobDetector_create(params) 
######################
cwd = os.getcwd()
baseFilePath = str(cwd).replace("\\","\\\\")+"\\Images"
#baseFilePath = 'C:\\Python\\Python37-32\\_DiamondPro\\Images'

def main():
    with open(baseFilePath+'\\SKU_target.txt', 'r') as myfile:
          data = myfile.read()
    SKU = str(data).replace(' ','').replace('SKU','')
    #SKU = '7272263'
    df_SKU = runForSKU(SKU)
    print(df_SKU.head(99))
    chartDict1 = plotHSVforDF(df_SKU)
    chartDict2 = plotBlobsScatter(df_SKU)
    chartDict1.update(chartDict2)
    savePlotsAsHTML(SKU, chartDict1)
    

def savePlotsAsHTML(SKU, chartDict):
    skuPath = baseFilePath+'\\SKU'+SKU
    html = 'SKU: '+SKU+' !!'
    for key in chartDict:
        #print(key, '->', chartDict[key])
        html = html + str(key) + '<img src=\'data:image/png;base64,{}\'>'.format(chartDict[key]) + 'Some more html'

    writepath = skuPath+'\\outputReport_'+SKU+'.html'
    mode = 'w' if os.path.exists(writepath) else 'w' #'a'
    with open(writepath,mode) as f:
        f.seek(0)
        f.write(html)
        f.truncate()


def runForSKU(SKU):
    skuPath = baseFilePath+'\\SKU'+SKU
    frameID = 0
    #df_FramesList = []
    df_SKU = pd.DataFrame( columns=['DiamondID','FrameID', 'Area', 'xcoord', 'ycoord', 'color'])
    for filename in os.listdir(skuPath):
        if "outputReport_" not in filename:
            print(filename)
            imgPath = skuPath+'\\'+filename
            df_frame = getBlobMetricsForFrame(imgPath, frameID, SKU).reset_index(drop=True)
            print(df_frame.size)
            df_SKU = pd.concat([df_SKU, df_frame],ignore_index=True, axis=0).reset_index(drop=True)
            frameID += 1
            #cv2.imshow(SKU,img)
            key = cv2.waitKey(25)#pauses for some seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
    return df_SKU
        

def maskandBlobDetect(img, lowerHSV, upperHSV, blobdetector, hsv):
    mask = cv2.inRange(hsv, lowerHSV, upperHSV)
    result = cv2.bitwise_and(img,img, mask= mask)
    h, s, v = cv2.split(result)
    v1 = cv2.bitwise_and(v,v, mask= mask)
    kpoints = blobdetector.detect(v1)
    #cv2.imshow('mask',mask)
    print(type(kpoints))
    print(len(kpoints))
    return kpoints, result

def base64StringToJPG(imgstring):
    imgdata = base64.b64decode(imgstring)
    print(type(imgdata))
    return imgdata;

def getBlobMetricsForFrame(imgPath, frameID, SKU):
# Read image
#img = cv2.imread('C:\\Python\\Python37-32\\_DiamondPro\\Images\\SKU7272263\\2aafb17d-773e-4367-b597-9a191b171a7a.jpg')
    img = cv2.imread(imgPath)
    cv2.imshow(imgPath,img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ##First Pass Saturation & Value filtering## - removes whites, greys, blacks
    lower_HSV = np.array([0,40,100]) #(Hue, Saturation, Value)
    upper_HSV = np.array([255,255,255])
    mask = cv2.inRange(hsv, lower_HSV, upper_HSV)
    res = cv2.bitwise_and(img,img, mask= mask)
    #cv2.imshow('res',res)
    ############

    lower_HSV = np.array([0,0,0]) #(Hue, Saturation, Value) #Orange
    upper_HSV = np.array([80,255,255]) #Oranges Only
    keypoints_orange, oranges = maskandBlobDetect(res, lower_HSV, upper_HSV, detector, hsv)
    #cv2.imshow('oranges',oranges)

    lower_HSV = np.array([80,0,0]) #(Hue, Saturation, Value) #Blues
    upper_HSV = np.array([115,255,255]) 
    keypoints_blue, blues = maskandBlobDetect(res, lower_HSV, upper_HSV, detector, hsv)
    #cv2.imshow('blues',blues)

    lower_HSV = np.array([115,0,0]) #(Hue, Saturation, Value) #PINKS
    upper_HSV = np.array([255,255,255]) #Purples Only
    keypoints_pink, pinks = maskandBlobDetect(res, lower_HSV, upper_HSV, detector, hsv)
    #cv2.imshow('pinks',pinks)

    #Detect blobs 
    #keypoints = detector.detect(v1)
    keypoints = keypoints_pink + keypoints_orange + keypoints_blue
    print(type(keypoints))
    print(len(keypoints))
    print(res.shape)

    diamondID = SKU
    frameID = frameID
    df_frame = pd.DataFrame( columns=['DiamondID','FrameID', 'Area', 'xcoord', 'ycoord', 'color'])
    for k in keypoints:
        #print('area: ',  k.size)
        #print('coord: ', k.pt)
        color = res[int(k.pt[1]), int(k.pt[0])]
        #print('color RGB: ',color)
        #df_frame.append([[diamondID, frameID, k.size, k.pt, color]], ignore_index = True)
        #df_frame =df_frame.append([[diamondID, frameID, k.size, k.pt[0], k.pt[1], color]], ignore_index = True)
        diction = {'DiamondID': diamondID
                   , 'FrameID': int(frameID)
                   , 'Area': k.size
                   , 'xcoord': k.pt[0]
                   , 'ycoord': k.pt[1]
                   , 'color': color  }
        df_frame =df_frame._append(diction, ignore_index=True)
        

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints"+str(frameID), im_with_keypoints)
    
    #print(df_frame.head(25))
    return df_frame


def plotHSVforDF(df):
    hue = []
    saturation = []
    value = []
    for r in df['color']:
        print(r)
        h_s_v =  colorsys.rgb_to_hsv(r[0], r[1], r[2])
        hue.append(h_s_v[0]*180.0)
        saturation.append(h_s_v[1]*100.0)
        value.append(h_s_v[2])

    hueHist = np.histogram(hue, bins=range(0,180))
    saturationHist = np.histogram(saturation, bins=range(0,100))
    valueHist = np.histogram(value, bins=range(0,255))

    # Hue
    #print(hueHist[0])
    plt.figure(figsize=(20, 2))
    data = hueHist[0]
    y_pos = np.arange(len(data))
    plt.xlim([0, 180])
    colours = [colors.hsv_to_rgb((i/180, 1, 0.9)) for i in range(0, 180)]
    plt.bar(y_pos, data,  color=colours, edgecolor=colours, width=1)
    plt.title('Hue')
    tmpfileHue = BytesIO()
    plt.savefig(tmpfileHue, format='png')
    encodedHue = base64.b64encode(tmpfileHue.getvalue()).decode('utf-8')

    # Saturation
    #print(saturationHist[0])
    plt.figure(figsize=(20, 2))
    data = saturationHist[0]
    y_pos = np.arange(len(data))
    plt.xlim([0, 100])
    colours = [colors.hsv_to_rgb((.33, i/100, 1)) for i in range(0, 100)]
    plt.bar(y_pos, data,  color=colours, edgecolor=colours, width=1)
    plt.title('Saturation')
    tmpfileSat = BytesIO()
    plt.savefig(tmpfileSat, format='png')
    encodedSat = base64.b64encode(tmpfileSat.getvalue()).decode('utf-8')

    # Value
    #print(valueHist[0])
    plt.figure(figsize=(20, 2))
    data = valueHist[0]
    y_pos = np.arange(len(data))
    plt.xlim([0, 255])
    colours = [colors.hsv_to_rgb((0.0, 0.0, i/255)) for i in range(0, 255)]
    plt.bar(y_pos, data,  color=colours, edgecolor=colours, width=1)
    plt.title('Value')
    tmpfileVal = BytesIO()
    plt.savefig(tmpfileVal, format='png')
    encodedVal = base64.b64encode(tmpfileVal.getvalue()).decode('utf-8')

    ##########
    #plt.show()
    ##########
    return {'Hue': encodedHue, 'Saturation':encodedSat, 'Value':encodedVal}

def cnvrtColors(clr):
    #print(clr)
    clrHsv = colorsys.rgb_to_hsv(clr[0], clr[1], clr[2])
    clrHsv=(clrHsv[0],1,255)
    clrRGB = colorsys.hsv_to_rgb(clrHsv[0], clrHsv[1], clrHsv[2])
    print(clrRGB)
    clrHex = '#{:02x}{:02x}{:02x}'.format( int(clrRGB[0]), int(clrRGB[1]) , int(clrRGB[2]) )
    print(clrHex)
    return clrHex

def getHue(clr):
    #print(clr)
    hue = colorsys.rgb_to_hsv(clr[0], clr[1], clr[2])[0]
    return hue
    

def plotBlobsScatter(df):
    x = df['color'].apply(getHue)
    y = df['Area']
    size = ((df['Area']*df['Area'] / 4))
    colors = df['color'].apply(cnvrtColors)

    plt.figure(figsize=(15, 15))
    plt.scatter(x, y, s=size, c=colors)
    plt.title('FireSpots')
    plt.xlabel('Hue [0,1] spectrum')
    plt.xticks(np.arange(0, 1, step=0.1))
    plt.ylabel('FireShine Area (pixels)^2')
    #plt.show()

    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return {'ScatterPlot':encoded}

#def calculateKeyFrameArea(img):

############
main()
input("Press enter to exit...")
###########
