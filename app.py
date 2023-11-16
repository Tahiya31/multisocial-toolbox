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

# import libraries for video processing
import cv2
import mediapipe as mp
import csv
import tempfile


#a few more libraries
import torch
import whisper
from tempfile import NamedTemporaryFile
from scipy.io.wavfile import read
import librosa

import opensmile
import base64
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp


# TO DO: 
# allow selecting feature type and level for opensmile
# allow choice of language, model for whisper
# allow multiple file upload


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

    st.subheader("I want transcript from an audio file")

    #message after uploading the file
    # source: https://github.com/tyler-simons/backgroundremoval/blob/main/bg_remove.py
    def convert_vid2audio(video_upload, audio_file):
        audio_file = subprocess.run(["ffmpeg", "-i", video_upload.name, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "2", "pipe:"], stdout=subprocess.PIPE).stdout
        #subprocess.run('ffmpeg -i {$video_upload.name} {$audio_file}',shell=True)
        
        #subprocess.call(command, shell=True)
        #clip = mp.VideoFileClip(video_upload)
        #audio_file = clip.audio.write_audiofile("converted_" + video_upload.name.split('.')[0]+".wav", codec = 'wav')
        
        st.download_button("Download your audio", audio_file, "converted_"+video_upload.name.split('.')[0]+".wav", "audio/wav")        

    #with st.sidebar:
        #st.subheader("Data Preparation")

        #st.markdown("Convert video (mp4) to audio (wav)")
        #video_upload = st.file_uploader("Upload your video file here", type=["mp4"], accept_multiple_files=False,label_visibility="visible", key= 'video_audio')
        
        #if video_upload:
            #audio_file = tempfile.NamedTemporaryFile(delete=False)
            #audio_file = "converted_" + video_upload.name.split('.')[0]+".wav"
            #convert_vid2audio(video_upload, audio_file)

    #st.divider()
    
        
    #get transcript using whisper
    def get_transcript(upload):
        torch.cuda.is_available()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        #this chooses the model language to english
        model = whisper.load_model("base.en", device=DEVICE)
        
        #st.write(upload)
        audio, sr = librosa.load(upload)
        text = model.transcribe(audio)  

        st.download_button("Download your transcript", text['text'], "transcript_"+upload.name.split('.')[0]+".txt", "audio/wav")        

    sound_upload = st.file_uploader("Upload your audio file here", type=["wav"], accept_multiple_files=False,label_visibility="visible", key= 'transcript')

    if sound_upload:
        get_transcript(sound_upload)
    
    st.divider()

    st.subheader("I want acoustic-prosodic characteristics from an audio file")

    
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
        #st.write(id_and_feature_columns)
        #st.write(id_and_feature)

        df = pd.DataFrame(id_and_feature,columns =id_and_feature_columns)
        final_df = df.to_csv(index=False)
        #b64 = base64.b64encode(df.encode()).decode()

        st.download_button("Download your audio features", final_df, 'acoustic_prosodic_features_' + upload.name.split('.')[0] + '.csv', "txt/csv")
        
    audio_upload = st.file_uploader("Upload your audio file here", type=["wav"], accept_multiple_files=False,label_visibility="visible", key= 'speech')

    if audio_upload:
        #bytes_data = audio_upload.getvalue()
        #
        # st.write(bytes_data)
        get_audio_features(audio_upload)

    st.divider()

    st.subheader("I want movement from a video file")

    def get_movement(movement_upload):
        # Initialize MediaPipe Pose and Drawing utilities
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(movement_upload.read())

        # Open the video file
        cap = cv2.VideoCapture(tfile.name)

        frame_number = 0
        csv_List = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            result = pose.process(frame_rgb)

            # Draw the pose landmarks on the frame
            if result.pose_landmarks:
                #print(result.pose_landmarks.landmark)
                #mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Add the landmark coordinates to the list and print them
                #write_landmarks_to_csv(result.pose_landmarks.landmark, frame_number, csv_data)
                for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
                    csv_List.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
                
                #poseMarkList = []
                #for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    #print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
                    #poseMarkList.append([mp_pose.PoseLandmark(idx).name.landmark.x, landmark.y, landmark.z])

                # stratify the list
                #flat_list = [item for sublist in poseMarkList for item in sublist]
                #flat_list.insert([0, frame_number])
                #csv_List.append(flat_list)

            # Display the frame
            #cv2.imshow('MediaPipe Pose', frame)

            # Exit if 'q' key is pressed
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break

            frame_number += 1

        #cap.release()
        #cv2.destroyAllWindows()

        # Save the CSV data to a file
        #with open(output_csv, 'a+', newline='') as csvfile:
            #csv_writer = csv.writer(csvfile)
            #csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
            #csv_writer.writerows(csv_List)

        df = pd.DataFrame(csv_List, columns = ['frame_number', 'landmark', 'x', 'y', 'z']) 
        final_df = df.to_csv(index=False)
        st.download_button("Download your movement features", final_df, 'movement_features_' + movement_upload.name.split('.')[0] + '.csv', "txt/csv")

    movement_upload = st.file_uploader("Upload your video file here", type=["mp4"], accept_multiple_files=False,label_visibility="visible", key= 'movement')
    
    if movement_upload:
        #output_csv = 'movement_'+movement_upload.name.split('.')[0]+'.csv'
        get_movement(movement_upload)

    st.divider()

    st.subheader("I have feedback/ideas for this toolbox") 
    st.link_button("Sure, let's go to the feedback form", "https://forms.gle/SPojkeqLhXsKVJK1A", help=None, type="secondary", disabled=False, use_container_width=False)




# misc: sharing the app with others: https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace
if __name__ == "__main__":
    run()
