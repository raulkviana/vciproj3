# VCI Project: Identify lego pieces

## Overview
This repository contains all the information used to develop a solution to identify lego pieces (color, ratio and position). The code was developed in Python.


## Index
* [Install](#install)
* [Main Pipeline](#mainPipeline)
* [Some Results](#someResults)
* [Folders](#folders)

<a name="mainPipeline"/>
<a name="someResults"/>
<a name="folders"/>
<a name="install"/>


## Install 
In order to install all the requirements to use our application you must run the following command on a terminal (in this directory):
```
pip install -r requirements.txt
```

## Main Pipeline 
![Our pipeline](/pipelineVCI.jpg "Pipeline for lego identification")

### Note
The color thresholding is done through calibrated colors obtained with the region growing algorithm.

## Some results

| Topic  | Result |
| ------------- |:-------------:|
| Region Growing | ![Region Growing](/regionGrowing.gif "Region Growing")|
| Color | ![Color](https://github.com/raulkviana/vciproj3/blob/main/Main%20Code/2%20and%203%20Iteration/Ratio/color_detection.jpg "Color")|
| Corners | ![Corners](https://github.com/raulkviana/vciproj3/blob/main/Main%20Code/2%20and%203%20Iteration/Ratio/imageWithCorners.jpg "Corners")|
| Ratio | ![Ratio identification](https://github.com/raulkviana/vciproj3/blob/main/Main%20Code/2%20and%203%20Iteration/Ratio/finalImage.jpg "Ratio")|

## Folders

| Folder  | Description |
| ------------- |:-------------:|
|Deliverables | Folder containing all the deliverables|
| Main Code | In this folder we have the working code we developed during <br> the semester |
| Testing | Folder containing all the algorithms used to develop <br> the main code and test  ideas and different approaches |
| dataset | Dataset made available by the professor|
| dataset2 | Dataset created by the team|
