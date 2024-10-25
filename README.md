# Sortify -- Face-Recognition-based-Image-Sorter
This is a small fun project which uses face recognition techniques to separate images from a large dataset into images of different people according to faces.

This repo contains all the code for my blog post on [medium](https://medium.com/analytics-vidhya/face-recognition-based-image-separator-408681f2360d)

# Contents
  1. Getting Started
  2. Requirements
  3. Files Included
  4. Running the code
  5. Results
  6. License
  7. Acnowledgements
  8. References
 
## Getting Started:
Before getting started first create directory in your system for this project and then add the following directories to it.
  1. Dataset (Contains a large image set which you want to separate)
  
  ![](images/dataset.PNG)
  
  2. People (Contains 1 image per person for whom you want to recognize images and separate)
  
  ![](images/people.PNG)
  
  3. output (Empty directory, the code will fill it with images of people once it has run successfully)
  
The folder structure looks as follows:

![](images/Untitled%20Diagram.png)
  
## Requirements:
Make sure you have already installed the following packages along with python 3.7:
  1. [opencv](https://pypi.org/project/opencv-python/) - Image processing library
  2. [pickle](https://pypi.org/project/pickle5/) - To store encodings
  3. [dlib](https://github.com/davisking/dlib) - ML toolkit for image based tasks
  4. [face_recognition](https://github.com/ageitgey/face_recognition) - Face Recognition api built on dlib

## Files Included:
All the neccessary code is inside [image_segmentation.py](image_segmentation.py) file.

[known_encodings.pickle](known_encodings.pickle) is my people dataset encodings file, you can create your own when you run the code.

## Running the code:
To run the code first add the neccessary images as stated in the getting started section.
After that open command prompt or terminal in your root directory and run the following command:
  ```
  python image_segmentation.py
  ```
Wait and once it's finished, check out the output folder.

## Results:
Once the code is run the output folder looks something like this:

![](images/output%20folder.PNG)

Atharv Folder:

![](images/atharv.jpeg)

Urjit Folder:

![](images/urjit.jpeg)

Group Folder:

![](images/group.PNG)

Unknown Folder:

![](images/unknown.PNG)


