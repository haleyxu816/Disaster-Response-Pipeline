# Disaster-Response-Pipeline

### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [How to use the code](#use)
4. [Data Source](#source)

## Project Motivation<a name="motivation"></a>

During disaster events, many messsages were sent. In this project, I want to create an ETL and machine learning pipeline to  classify disaster messages, so that receivers can send the messages to an appropriate disaster relief agency.

Moreover, this project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## File Descriptions <a name="files"></a>

There are three folders:
1. data :   
   disaster_messages.csv & disaster_categories.csv : raw dataset  
   process_data.py : data cleaning pipeline
   
2. models :  
   train_classifier.py : text processing and machine learning pipeline  
   classifier.pkl : final model  
   
3. app :  
   files needed to render the webpage
   
## How to use the code<a name="use"></a>

### Build new model by using new dataset: 
If someone in the future comes with a revised or new dataset of messages, they should be able to easily create a new model just by running this code. process_data.py and train_classifier.py should be able to run with additional arguments specifying the files used for the data and model.  

Example code:  
    
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db   

    python train_classifier.py ../data/DisasterResponse.db classifier.pkl    


### Run the web app:
Open a new terminal window. You should already be in the workspace folder, but if not, then use terminal commands to navigate inside the folder with the run.py file.

Type in the command line:
    
    python run.py
    
Now, open another Terminal Window. Type:

    env|grep WORK
    
You should see the **SPACEID** and **SPACEDOMAIN**.
In a new web browser window, type in the following:

    https://SPACEID-3001.SPACEDOMAIN
    


## Data Source<a name="source"></a>
Must give credit to [Figure Eight](https://www.figure-eight.com/) for the data.
