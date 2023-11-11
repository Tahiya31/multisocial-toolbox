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

from io import StringIO

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to MultiSOCIAL toolbox!")

    #st.sidebar.success("Select a demo above.")


    st.markdown(
        """
        
        We present a new effort to assemble open-source tools into a single platform for multimodal interaction data
        the MultiSOCIAL Toolbox (or the MULTImodal timeSeries Open-SourCe Interaction Analysis Library).
          While these tools exist in separate packages for scientists with programming abilities
          our goal is to expand access to scholars with limited (or even non-existent) programming experience
            and to accelerate discovery through a unified multimodal data processing pipeline. 
            The toolbox enables any researcher who has video files of any kind of interaction to extract time-series data in three modalities:
          - body movement (non-verbal behavior); 
          - transcripts (what was said during interaction); 
          - acoustic prosodic characteristics (how it was said).
        
    """
    )

    st.header("If you have a video, follow the steps below")

    st.file_uploader("Upload your video file", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None,  disabled=False, label_visibility="visible")

    st.button("Convert video to audio", type="secondary")

    st.button("Split you video screen", type="secondary")

    st.button("Get body movement", type="primary")

    st.button("Get transcript", type="primary")

    st.button("Get acoustics-prosodic", type="primary")

if __name__ == "__main__":
    run()
