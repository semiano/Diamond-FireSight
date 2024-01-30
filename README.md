Application uses OpenCV for Python to detect color refactions from cut diamonds.  Often referred to as "Fire", there is a quality to diamonds which varies for each specimen.  Currently there is no quantification for this metric, it is only described qualitatively.  This software attempts quantify it. 

Inputs:
 - raw input images should be stored in directory .\images\SKU####\
 - an output report file will be generated in the same directory.  The output file is a histogram of color blobs detected, size, saturation, hue, value.  The blobs represent detect color refractions within the diamond.
 - SKU_target.txt should contain the "SKU####" name that the program will process

Example output report:
![image](https://github.com/semiano/SEMServer/assets/6520366/2c4e3d02-2ebd-4dd2-9a95-ad7a5d1816e0)
![image](https://github.com/semiano/SEMServer/assets/6520366/ea938e0e-174c-45c0-aa8d-601284ea1b91)
