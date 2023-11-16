# Welcome to MultiSOCIAL toolbox! An Open-Source Library for Quantifying Multimodal Social Interaction

Veronica Romero, Department of Psychology and Davis Institute for Artificial Intelligence,  Colby College, USA, vcromero@colby.edu, Tahiya Chowdhury, Davis Institute for Artificial Intelligence,  Colby College, USA, tchowdhu@colby.edu, Alexandra Paxton, Department of Psychological Science & Center for the Ecological Study of Perception and Action,  University of Connecticut, USA, alexandra.paxton@uconn.edu.

In everyday life, most of our experiences of social interaction are multimodal, but in our research, most of our studies of social interaction focus on only one kind of communication behavior. However, by reducing the dimensionality of our data, we are limited in our ability to capture the dynamics of real-world social behavior. A major reason for this unimodal focus is technological: Until recently, the richness of human behavior in just minutes of face-to-face interaction could take over an hour of meticulous hand-coding, transcription, and annotation, but advances in computing power and software innovation are changing that. Here, we present a new effort to assemble open-source tools into a single platform for multimodal interaction data: the MultiSOCIAL Toolbox (or the MULTImodal timeSeries Open-SourCe Interaction Analysis Library). While these tools exist in separate packages for scientists with programming abilities, our goal is to expand access to scholars with limited (or even non-existent) programming experience and to accelerate discovery through a unified multimodal data processing pipeline. The toolbox enables any researcher who has video files of any kind of interaction to extract time-series data in three modalities: body movement (non-verbal behavior); transcripts (what was said during interaction); and acoustic prosodic characteristics (how it was said).


# Tell me how to use this toolbox

This toolbox is created using streamlit, a Python based web app environment.

1. First, create a codespace to make your own copy of the app. To do this:
   - click on the green button (as shown below)
  
   - click on `Codespace`. Click on the `...` beside Codespace for settings.
   - Choose `New with options` to create a new codespace that will host the app.
   - Choose a `Machine Type` (preferrably 4 core, 16 GB RAM if you are planning to process video data)
   - Click `Create Codespace`.

2. You will see a black screen while the codespace sets up the web app environment. Click on `View logs` should show the progress. (this step may take some type)
  
3. If the codespace set up is successful, you should see Visual Studio environment that will allow you to interact with the app files.

# Ok, how do I use the app for processing my data?

Let's get that app running on our browser! To do this:

1. Click on `TERMINAL` in the lower panel on your screen.
2. type `pip install -r requirements.txt` and hit `Enter`. This will install all the software packages we are using behind our app.
3. Once everything is installed, type `streamlit run app.py` to run the app.
4. If everything went smoothly above (hopefully!), you should see the app interface appearing on the right panel

   The app is ready to use! **Welcome to MultiSOCIAL Toolbox**. 

# Common issues and workarounds

1. `The codespace is taking too long to setup` -- If it does not take you to the Visual studio symbol within a few minutes, feel free to refresh your browser tab.
2. `The codespace is taking too long to setup` -- Codespace responds best on Google Chrome compared to other browsers (e.g., Safari).
3. `I am getting an error in the TERMINAL tab about whisper and torch.` -- You can install these two packages by running `pip install torch torch-utils whisper` in the terminal.
4. `I am having issue with openCV when importing cv2` -- You may want to update pip first (`pip install --upgrade pip`), and then run `pip install -r requirements.txt` again to resolve this.


