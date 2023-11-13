# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

import io
from io import StringIO

#import libraries for multiSOCIAL toolbox
import os 
import pandas as pd
import numpy as np
import csv
import subprocess
import pathlib

#a few more libraries
import torch
import whisper
from tempfile import NamedTemporaryFile
from scipy.io.wavfile import read
import librosa

import opensmile
import base64


LOGGER = get_logger(__name__)

#main app run
def run():
    st.set_page_config(
        page_title="MultiSOCIAL Homepage",
        page_icon="👋",
    )
    st.write("# Welcome to MultiSOCIAL toolbox!")

    #disable for now will use for converting data later
    #st.sidebar.success("Select a demo above.")

    st.markdown(
    """
        
        We present a new effort to assemble open-source tools into a single platform for multimodal interaction data
        the MultiSOCIAL Toolbox (or the MULTImodal timeSeries Open-SourCe Interaction Analysis Library).
          
        The toolbox enables any researcher who has video files of any kind of interaction to extract time-series data in three modalities:
        - body movement (non-verbal behavior); 
        - transcripts (what was said during interaction); 
        - acoustic prosodic characteristics (how it was said).
        
    """
    )

    st.subheader("If you have a video, follow the steps below")

    #message after uploading the file
    # source: https://github.com/tyler-simons/backgroundremoval/blob/main/bg_remove.py
    

   
    def convert_vid2audio(video_upload, audio_file):
        command = "ffmpeg -i {video} -acodec pcm_s16le -ar 16000 -ac 2 {audio}".format(video=video_upload, audio=audio_file)
        subprocess.call(command,shell=True)

    #get transcript using whisper
    def get_transcript(upload):
        torch.cuda.is_available()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        #this chooses the model language to english
        model = whisper.load_model("base.en", device=DEVICE)
        
        #st.write(upload)
        audio, sr = librosa.load(upload)
        text = model.transcribe(audio)  

        st.download_button("Download your transcript", text['text'], "transcript.txt", "audio/wav")        

    video_upload = st.file_uploader("Upload your video file here", type=["wav"], accept_multiple_files=False,label_visibility="visible")

    if video_upload:
        get_transcript(video_upload)
    
    
    #get acoustic-prosodic features with opensmile
    def get_audio_features(upload):

        feature_set_name = opensmile.FeatureSet.ComParE_2016
        feature_level_name=opensmile.FeatureLevel.LowLevelDescriptors

        #define csv file name
        csv_name = 'acoustic_prosodic_features_' + upload.name.split('.')[0] + '.csv'

        # we want to save file names to keep track of what file has been analyszed
        #file_names_columns = ['File_name']

        smile = opensmile.Smile(
        feature_set=feature_set_name,
        feature_level=feature_level_name
        )

        #feature column names for csv file
        feature_names = smile.feature_names


        # complete final column names
        audio, sr = librosa.load(upload)
        st.write(upload)
            
        file_id = [str(upload.name)]
        #feature = smile.process_file(upload)
        feature = smile.process_signal(audio,sr)
        
        #storing the statistical values of the features, we store mean here
        mean_feature = np.mean(feature, axis = 0).tolist()
        #st.write(mean_feature)
            
        #file and feature
        id_and_feature_columns = ['file_id'] + feature_names
        id_and_feature = [file_id + mean_feature]
        st.write(id_and_feature_columns)
        st.write(id_and_feature)

        df = pd.DataFrame(id_and_feature,columns =id_and_feature_columns)
        final_df = df.to_csv(index=False)
        #b64 = base64.b64encode(df.encode()).decode()

        st.download_button("Download your audio features", final_df, 'acoustic_prosodic_features_' + upload.name.split('.')[0] + '.csv', "txt/csv")
        
    audio_upload = st.file_uploader("Upload your audio file here", type=["wav"], accept_multiple_files=False,label_visibility="visible")

    if audio_upload:
        #bytes_data = audio_upload.getvalue()
        #
        # st.write(bytes_data)
        get_audio_features(audio_upload)



        





















    #st.button("Convert video to audio", type="secondary")

    #st.button("Split your video screen", type="secondary")

    #st.button("Get body movement", type="primary")

    #st.button("Get transcript", type="primary")

    #st.button("Get acoustic-prosodic", type="primary")


if __name__ == "__main__":
    run()
