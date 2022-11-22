import sample
import cv2
from build_vocab import Vocabulary
import json
import requests
import urllib
from playsound import playsound
from gtts import gTTS
import os

headers = {'Content-type': 'application/json',
           'token': '-wGcnpMasnRrNWs3Grdc5vWV9l0FkIQrQ6BHbiPBB0UY1h2eI89NBD-O-mpaCvAl'}

try:

    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union

    import pandas as pd
    import streamlit as st
except Exception as e:
    print(e)

STYLE = """
<h1>ĐỒ ÁN MÔN MACHINE LEARNING WITH LARGE DATASET CỦA NHÓM 5</h1>
<style>
img {
    max-width: 100%;
}
body {
    margin: 0px;
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 400;
    line-height: 1.6;
    color: rgb(250, 250, 250);
    background-color:Blue;
    text-size-adjust: 100%;
    -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
    -webkit-font-smoothing: auto;
    text-align:center;
}
button {
background-color: Orange;
}

h1 {color: Gray;}
p {color: Red;
     font-size: 40px
}
</style>
"""
image = ''


class FileUpload(object):
    def __init__(self):
        self.fileTypes = ["csv", "png", "jpg", "jpeg"]

    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        file_uploaded = 0

        #st.info(__doc__)
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))
            return
        content = file.getvalue()
        file_uploaded = 1

        if isinst   ance(file, BytesIO):
            show_file.image(file)
        else:
            data = pd.read_csv(file)
            st.dataframe(data.head(10))
        file.close()

        image_path = str(file.name)
        encoder_path = "encoder-50-20.ckpt"
        decoder_path = "decoder-50-20.ckpt"
        vocab_path = "preprocessed_vocab_.pkl"

        if (file_uploaded == 1):
            returned_sentence = sample.run_inference(image_path, encoder_path, decoder_path, vocab_path)
            st.write(returned_sentence)
            tts = gTTS(text=returned_sentence, lang='vi')
            tts.save("Siri.mp3")
            st.audio("Siri.mp3")
            os.remove("Siri.mp3")



if __name__ == "__main__":
    helper = FileUpload()
    helper.run()