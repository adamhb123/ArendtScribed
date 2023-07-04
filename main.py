"""
Programmatically converting captions of a black & white film with white captions to a transcript.

The film is an interview with Hannah Arendt.

Source: https://www.youtube.com/watch?v=dVSRJC4KAiE
"""

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
import cv2
from mss import mss
from PIL import Image
import pytesseract
from pprint import pprint
import language_tool_python

lang_tool = language_tool_python.LanguageTool('en-US')

def best_interpretation(interpretations):
    #print(min([len(lang_tool.check(x)) for x in interpretations]))
    mode = max(set(interpretations), key=interpretations.count)
    return mode

def main():
    bounding_box = {'top': 775, 'left': 1820, 'width': 1070, 'height': 175}
    sct = mss()
    final_transcript = []
    interpretations = []
    i = 0
    last_thresh = None
    similarity_threshold = 0.9
    thresh_std_threshold = 50
    while True:
        sct_img = sct.grab(bounding_box)
        img_arr = np.array(sct_img)
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 250,255, cv2.THRESH_BINARY)
        im_as_str = pytesseract.image_to_string(thresh)
        interpret = im_as_str.replace('\n',' ').replace('\x0c','')
        interpretations.append(interpret)
        cv2.imshow('screen', thresh)
        #print(interpret)
        if last_thresh is not None and thresh.std() > thresh_std_threshold:
            score, diff = structural_similarity(thresh, last_thresh, full=True)
            if score < similarity_threshold:
                final_transcript.append(best_interpretation(interpretations))
                interpretations.clear()
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
        last_thresh = thresh.copy()

    print(final_transcript)
    # Save to CSV
    df = pd.DataFrame(final_transcript,columns=["Phrase"])
    df.to_csv("transcription.csv")
    return final_transcript

if __name__=="__main__":
    main()