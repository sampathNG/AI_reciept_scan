from fastapi import APIRouter, Depends, FastAPI,HTTPException
from imutils.perspective import four_point_transform
import numpy as np
import pytesseract
import argparse
import imutils
import cv2
import re
import os
router = APIRouter()
@router.get("/")
async def hello():
    return {"message": "Hello World"}

@router.get("/ocr")
async def ocr():
    image_path = os.path.join(os.getcwd(),"controllers", "images", "receipt.jpg")

    orig = cv2.imread(image_path)
    image = orig.copy()
    image = imutils.resize(image, width=500)
    ratio = orig.shape[1] / float(image.shape[1])
    # 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
    edged = cv2.Canny(blurred, 75, 200)
    # 
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    receiptCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the receipt
        if len(approx) == 4:
            receiptCnt = approx
            break
    # if the receipt contour is empty then our script could not find the
    # outline and we should be notified
        if receiptCnt is None:
            raise Exception(("Could not find receipt outline."))
    # 
    # 
    reciept = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)
    # cv2.imshow("Receipt Transform", imutils.resize(receipt, width=500))
    # cv2.waitKey(0)
    options = "--psm 4"
    # text = pytesseract.image_to_string(
    #     cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB),
    #     config=options)
    src = cv2.imread(image_path) 

    text = pytesseract.image_to_string(
        cv2.cvtColor(src, cv2.COLOR_BGR2RGB),
        config=options)
    text = pytesseract.image_to_string(image_path)
    # text = pytesseract.image_to_string(receipt)
    # show the raw output of the OCR process
    print(reciept)
    print("\n")
    # print(receipt)
    return {"message": "Hello World"}



@router.get("/ocrr")

# def perform_ocr(img: np.ndarray):

def perform_ocr():
    image_path = os.path.join(os.getcwd(),"controllers", "images", "receipt.jpg")

    # img_orig = cv2.imdecode(image_path, cv2.IMREAD_COLOR)

    img_orig = cv2.imread(image_path,cv2.IMREAD_COLOR)
    image = img_orig.copy()
    image = imutils.resize(image, width=500)
    ratio = img_orig.shape[1] / float(image.shape[1])

    # convert the image to grayscale, blur it slightly, and then apply
    # edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (
            5,
            5,
        ),
        0,
    )
    edged = cv2.Canny(blurred, 75, 200)

    # find contours in the edge map and sort them by size in descending
    # order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the receipt outline
    receiptCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the receipt
        if len(approx) == 4:
            receiptCnt = approx
            break

    # if the receipt contour is empty then our script could not find the
    # outline and we should be notified
    if receiptCnt is None:
        raise Exception(
            (
                "Could not find receipt outline. "
                "Try debugging your edge detection and contour steps."
            )
        )

    # apply a four-point perspective transform to the *original* image to
    # obtain a top-down bird's-eye view of the receipt
    receipt = four_point_transform(img_orig, receiptCnt.reshape(4, 2) * ratio)

    # apply OCR to the receipt image by assuming column data, ensuring
    # the text is *concatenated across the row* (additionally, for your
    # own images you may need to apply additional processing to cleanup
    # the image, including resizing, thresholding, etc.)
    options = "--psm 4"
    text = pytesseract.image_to_string(
        cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB), config=options
    )
    print(text)
    return text