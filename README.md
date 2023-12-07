# FOLDER STRUCTURE

- datasets: contains all datasets, CMT1_nodes is a smaller dataset, CMT14_nodes is a dataset with 121 cities and vrp_cristo5 is the dataset with 27 cities that has been generated after preprocessing and is the one being used by the code
- source_code: contains the source code vrp_app.py and a requirements.txt file
- project_report: contains the project report

# INSTRUCTIONS ON HOW TO COMPILE AND EXECUTE CODE

Step 1: Run the requirements.txt file using the command: ‘pip install -r requirements.txt’. This file installs all necessary packages in order to run the code </br>
Step 2: Update the path for the dataset in vrp.py line number 41, to the location where the dataset is present </br>
Step 3: Run the command, ‘streamlit run vrp.py’. This will open up the UI on your localhost </br>
Step 4: Interact with the UI. By default, the Google ORTools algorithm will run and provide a result on the UI. Next, pressing the ‘Run GA models’ button will sequentially run the pure GA model then the hybrid enhanced GA model, and then provide a graph showcasing the comparison of how pure GA and hybrid GA models have performed against the results obtained by Google ORTools </br>

# TEAM

G10: Akash Janardhan Srinivas, Sumukh Naveen Aradhya
CS255 Sec 01: Design and Analysis of Algorithms, Fall 2023
