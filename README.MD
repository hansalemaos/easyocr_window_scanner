# Don't install this module without reading the instructions!!

## The best OCR engine

In my opinion, EasyOCR is the best OCR engine out there.  
The results are much better than Tesseract's OCR, but it is much slower.  
**And, unfortunately, it is a bit tricky to install.**

Example - **Bluestacks** [adb/hwnd]

<div align="left">
      <a href="https://www.youtube.com/watch?v=hDRwmzbrvaY">
         <img src="https://img.youtube.com/vi/hDRwmzbrvaY/0.jpg" style="width:100%;">
      </a>
</div>

##### Step 1

If you have a GPU and Windows (by the way, this module is only for Windows), you should install torch and torchvision first to get a good speed up  

*From https://github.com/JaidedAI/EasyOCR  
Note 1: For Windows, please install torch and torchvision first by following the official instructions here https://pytorch.org. On the pytorch website, be sure to select the right CUDA version you have. If you intend to run on CPU mode only, select CUDA = None.*

##### Step 2

```python
pip install easyocr_window_scanner
```

##### Step 3

EasyOCR installs the headless (light) version of opencv that doesn't even have cv2.imshow, so  
EasyOCR2Df.scan_window_with_adb / EasyOCR2Df.scan_window_with_hwnd don't work!

```python
To fix this problem, do this:
pip uninstall opencv-python
pip uninstall opencv-python-headless
pip install --upgrade --force-reinstall opencv-python==4.5.5.62
```

##### Step 4

You might get the following error when you import the module:  
cv2.error: Unknown C++ exception from OpenCV code.  

It seems to be a very common issue with EasyOCR  
https://github.com/opencv/opencv-python/issues/602#issuecomment-1070290110  

To fix this problem, you have to open:  
**"C:\Users\YOURUSERNAME\anaconda3\envs\YOURENV\Lib\site-packages\easyocr\craft_utils.py"**
and change:  
nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)  
to:  
nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8))

##### Step 5

```python
#We are finally ready to go

from easyocr_window_scanner import EasyOCR2Df

#create an instance, you can pass any argument you would pass to the EasyOCR module
easyo = EasyOCR2Df(["en"])

#Taking screenshots using adb, and do OCR on the fly
easyo.scan_window_with_adb(adb_path=r"C:\ProgramData\adb\adb.exe",adb_serial='localhost:5555',quit_key='q')

#Taking screenshots using the windows handle, and do OCR on the fly
easyo.scan_window_with_hwnd(hwnd=None, regular_expression='[bB]lue[sS]tacks.*', quit_key="q")

#If you just want to have the OCR results as a DataFrame, use:
easyo.start_ocr_to_df(r'https://i.ytimg.com/vi/fa82Qpw6lyE/hqdefault.jpg').get_results_df()
Out[5]:
                               text      conf  ...  middle_x  middle_y
0     No stop signs, speedin' limit  0.842704  ...       148        90
1       Nobody's gonna slow me down  0.744984  ...       159       112
2       Like a wheel, gonna spin it  0.966272  ...       141       136
3     Nobody's gonna mess me 'round  0.994361  ...       165       157
4               Satan! Paid my dues  0.708869  ...       153       180
5         Playin' in a rockin' band  0.979712  ...       133       202
6                  Mama' Look at me  0.559171  ...       147       224
7  Tm on my way t0 the promise land  0.703882  ...       174       247
8                               Hey  0.999969  ...        53       179
9                               Hey  0.999027  ...        52       224
[10 rows x 14 columns]

#If you want to have the image of the OCR results without showing it, use:
easyo.start_ocr_to_df(r'https://i.ytimg.com/vi/fa82Qpw6lyE/hqdefault.jpg').draw_easyocr_results().get_results_picture()
Out[7]:
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
        ...
```
