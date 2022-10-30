import random
from time import sleep
from typing import Any, Union
import keyboard
import numpy as np
from a_pandas_ex_plode_tool import pd_add_explode_tools
import pandas as pd
from windows_adb_screen_capture import ScreenShots

pd_add_explode_tools()
import easyocr
import requests
import os
import re
import cv2
from a_cv2_imshow_thread import add_imshow_thread_to_cv2

add_imshow_thread_to_cv2()


def open_image_in_cv(image, channels_in_output=None):
    if isinstance(image, str):
        if os.path.exists(image):
            if os.path.isfile(image):
                image = cv2.imread(image)
        elif re.search(r"^.{1,10}://", str(image)) is not None:
            x = requests.get(image).content
            image = cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    elif "PIL" in str(type(image)):
        image = np.array(image)
    else:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

    if channels_in_output is not None:
        if image.shape[-1] == 4 and channels_in_output == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[-1] == 3 and channels_in_output == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            pass
    return image


class EasyOCR2Df:
    r"""

    In my opinion, EasyOCR is the best OCR engine out there.
    The results are much better than Tesseract's OCR, but it is much slower than Tesseract.
    And, unfortunately, it is a bit tricky to install.

    Step 1)
    If you have a GPU and Windows (by the way, this module is only for Windows), you should install torch and torchvision first to get a good speed up

    From https://github.com/JaidedAI/EasyOCR
    Note 1: For Windows, please install torch and torchvision first by following the official instructions here https://pytorch.org. On the pytorch website, be sure to select the right CUDA version you have. If you intend to run on CPU mode only, select CUDA = None.

    Step 2)
    pip install easyocr_window_scanner

    Step 3)
    EasyOCR installs the headless (light) version of opencv that doesn't even have cv2.imshow, so
    EasyOCR2Df.scan_window_with_adb / EasyOCR2Df.scan_window_with_hwnd don't work!

    To fix this problem, do this:
    pip uninstall opencv-python
    pip uninstall opencv-python-headless
    pip install --upgrade --force-reinstall opencv-python==4.5.5.62

    Step 4)
    You might get the following error when you import the module:
    cv2.error: Unknown C++ exception from OpenCV code.

    It seems to be a very common issue with EasyOCR
    https://github.com/opencv/opencv-python/issues/602#issuecomment-1070290110

    To fix this problem, you have to open:
    "C:\Users\YOURUSERNAME\anaconda3\envs\YOURENV\Lib\site-packages\easyocr\craft_utils.py"
    and change:
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    to:
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8))

    Step 5)
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
"""

    def __init__(self, *args, **kwargs):
        self.reader = easyocr.Reader(*args, **kwargs)
        self.readyimage = np.array([])
        self.resultpicture = np.array([])
        self.df = pd.DataFrame()

    def draw_easyocr_results(self):
        image = self.readyimage.copy()
        for key, item in self.df.iterrows():
            fontdistance1 = 5
            fontdistance2 = 20
            if item["text"].strip() == "":
                continue
            tmp_tl_x = item["point_0_x"]
            tmp_tl_y = item["point_0_y"]
            tmp_br_x = item["width"]
            tmp_br_y = item["height"]
            conf = item["conf"]
            text = item["text"]
            r_, g_, b_ = (
                random.randrange(50, 255),
                random.randrange(50, 255),
                random.randrange(50, 255),
            )

            image = cv2.rectangle(
                image,
                (tmp_tl_x, tmp_tl_y),
                (tmp_br_x + tmp_tl_x, tmp_br_y + tmp_tl_y),
                (r_, g_, b_),
                1,
            )

            try:
                image = cv2.putText(
                    image,
                    str(conf),
                    (tmp_tl_x, tmp_tl_y - fontdistance1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    2,
                )
            except Exception:
                fontdistance1 = 0
                fontdistance2 = 0
                image = cv2.putText(
                    image,
                    str(conf),
                    (tmp_tl_x, tmp_tl_y - fontdistance1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    2,
                )

            image = cv2.putText(
                image,
                str(conf),
                (tmp_tl_x, tmp_tl_y - fontdistance1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (r_, g_, b_),
                1,
            )
            image = cv2.putText(
                image,
                text,
                (tmp_tl_x, tmp_tl_y - fontdistance2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 0, 0),
                3,
            )
            image = cv2.putText(
                image,
                text,
                (tmp_tl_x, tmp_tl_y - fontdistance2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (r_, g_, b_),
                1,
            )
        self.resultpicture = image.copy()
        return self

    def start_ocr_to_df(self, image: Any):
        self.readyimage = open_image_in_cv(image, channels_in_output=3)
        result = self.reader.readtext(self.readyimage)
        df = pd.DataFrame(result)
        df[1] = df[1].astype("string")
        coords = df[0].s_explode_lists_and_tuples(0)
        singlepoints = [
            df[[x for x in df.columns[1:]]]
            .copy()
            .rename(columns={1: "text", 2: "conf"})
        ]
        for ini, col in enumerate(coords.columns):
            subcoords = coords[col].s_explode_lists_and_tuples(col)
            subcoords.columns = [f"point_{ini}_x", f"point_{ini}_y"]
            for col2 in subcoords:
                subcoords[col2] = subcoords[col2].astype(np.uint16)
            singlepoints.append(subcoords.copy())
        df = pd.concat(singlepoints, axis=1)
        df["width"] = df["point_1_x"] - df["point_0_x"]
        df["height"] = df["point_2_y"] - df["point_0_y"]
        df["middle_x"] = df["point_0_x"] + (df["width"] // 2)
        df["middle_y"] = df["point_0_y"] + (df["height"] // 2)
        df["width"] = df["width"].astype(np.uint16)
        df["height"] = df["height"].astype(np.uint16)
        df["middle_x"] = df["middle_x"].astype(np.uint16)
        df["middle_y"] = df["middle_y"].astype(np.uint16)
        self.df = df.copy()
        return self

    def show_results(self, window_name: str = ""):
        cv2.imshow_thread(
            window_name=window_name, image=self.resultpicture, sleep_time=None
        )
        return self

    def get_results_df(self):
        return self.df

    def get_results_picture(self):
        return self.resultpicture

    def get_original_picture(self):
        return self.readyimage

    def scan_window_with_adb(
        self,
        adb_path: str = r"C:\ProgramData\adb\adb.exe",
        adb_serial: str = "localhost:5555",
        quit_key: str = "q",
        sleep_time: Union[float, int] = 0.05,
        print_df=True,
    ):
        def activate_stop():
            nonlocal stop
            stop = True

        stop = False
        sc2 = ScreenShots(hwnd=None, adb_path=adb_path, adb_serial=adb_serial)
        sc2.imshow_adb(sleep_time=sleep_time, quit_key=quit_key)
        sleep(1)
        sc2.enable_show_edited_images()
        keyboard.add_hotkey(quit_key, activate_stop)
        while not stop:
            screenshot_window = sc2.imget_adb()
            ocrresults = (
                self.start_ocr_to_df(screenshot_window)
                .draw_easyocr_results()
                .get_results_picture()
            )
            sc2.show_edited_image(ocrresults)
            if print_df:
                print("____________________")

                print(self.df[["text", "conf"]])
        keyboard.remove_all_hotkeys()
        return self

    def scan_window_with_hwnd(
        self,
        hwnd: Union[None, int] = None,
        regular_expression: Union[None, str] = None,
        quit_key="q",
        sleep_time: Union[float, int] = 0.05,
        print_df=True,
    ):
        def activate_stop():
            nonlocal stop
            stop = True

        stop = False

        sc2 = ScreenShots(hwnd=hwnd)
        if hwnd is None:
            sc2.find_window_with_regex(regular_expression)
        sc2.imshow_hwnd(sleep_time=sleep_time, quit_key=quit_key)
        sleep(1)
        sc2.enable_show_edited_images()
        keyboard.add_hotkey(quit_key, activate_stop)
        while not stop:
            screenshot_window = sc2.imget_hwnd()
            ocrresults = (
                self.start_ocr_to_df(screenshot_window)
                .draw_easyocr_results()
                .get_results_picture()
            )
            sc2.show_edited_image(ocrresults)
            if print_df:
                print("____________________")
                print(self.df[["text", "conf"]])
        keyboard.remove_all_hotkeys()
        return self
