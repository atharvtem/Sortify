#Neccessary Imports

import os
import cv2
import pickle
import numpy as np
import face_recognition

#Save encodings
def saveEncodings(encs,names,fname="encodings.pickle"):
    """
    Save encodings in a pickle file to be used in future.

    Parameters
    ----------
    encs : List of np arrays
        List of face encodings.
    names : List of strings
        List of names for each face encoding.
    fname : String, optional
        Name/Location for pickle file. The default is "encodings.pickle".

    Returns
    -------
    None.

    """
    
    data=[]
    d = [{"name": nm, "encoding": enc} for (nm, enc) in zip(names, encs)]
    data.extend(d)

    encodingsFile=fname
    
    # dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    f = open(encodingsFile, "wb")
    f.write(pickle.dumps(data))
    f.close()    

#Function to read encodings
def readEncodingsPickle(fname):
    """
    Read Pickle file.

    Parameters
    ----------
    fname : String
        Name of pickle file.(Full location)

    Returns
    -------
    encodings : list of np arrays
        list of all saved encodings
    names : List of Strings
        List of all saved names

    """
    
    data = pickle.loads(open(fname, "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]
    names=[d["name"] for d in data]
    return encodings,names

#Function to create encodings and get face locations
def createEncodings(image):
    """
    Create face encodings for a given image and also return face locations in the given image.

    Parameters
    ----------
    image : cv2 mat
        Image you want to detect faces from.

    Returns
    -------
    known_encodings : list of np array
        List of face encodings in a given image
    face_locations : list of tuples
        list of tuples for face locations in a given image

    """
    
    #Find face locations for all faces in an image
    face_locations = face_recognition.face_locations(image)
    
    #Create encodings for all faces in an image
    known_encodings=face_recognition.face_encodings(image,known_face_locations=face_locations)
    return known_encodings,face_locations

#Function to compare encodings
def compareFaceEncodings(unknown_encoding,known_encodings,known_names):
    """
    Compares face encodings to check if 2 faces are same or not.

    Parameters
    ----------
    unknown_encoding : np array
        Face encoding of unknown people.
    known_encodings : np array
        Face encodings of known people.
    known_names : list of strings
        Names of known people

    Returns
    -------
    acceptBool : Bool
        face matched or not
    duplicateName : String
        Name of matched face
    distance : Float
        Distance between 2 faces

    """
    duplicateName=""
    distance=0.0
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding,tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)
    distance=face_distances[best_match_index]
    if matches[best_match_index]:
        acceptBool=True
        duplicateName=known_names[best_match_index]
    else:
        acceptBool=False
        duplicateName=""
    return acceptBool,duplicateName,distance

#Save Image to new directory
def saveImageToDirectory(image,name,imageName):
    """
    Saves images to directory.

    Parameters
    ----------
    image : cv2 mat
        Image you want to save.
    name : String
        Directory where you want the image to be saved.
    imageName : String
        Name of image.

    Returns
    -------
    None.

    """
    path="C:/Blog/Blog 31/Face-Recognition-based-Image-Separator-master/Output"+name
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    cv2.imwrite(path+"/"+imageName,image)
    

def processKnownPeopleImages(path=r"C:/Blog/Blog 31/Face-Recognition-based-Image-Separator-master/People", saveLocation="./known_encodings.pickle"):
    """
    Process images of known people and create face encodings to compare in future.
    Each image should have just 1 face in it.

    Parameters
    ----------
    path : STRING, optional
        Path for known people dataset. Each image in this dataset should contain only 1 face.
    saveLocation : STRING, optional
        Path for storing encodings for known people dataset.

    Returns
    -------
    None.
    """
    
    known_encodings = []
    known_names = []
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)

        # Read image
        image = cv2.imread(imgPath)
        if image is None:
            print(f"Error: Image {img} not found or failed to load.")
            continue  # Skip to the next image if failed to load
        
        # Resize the image
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        
        # Extract name from filename
        name = os.path.splitext(img)[0]
        
        # Get locations and encodings
        encs, locs = createEncodings(image)
        
        # Assume only one face per image as per the docstring
        if encs:
            known_encodings.append(encs[0])
            known_names.append(name)
            
            # Draw rectangle around the face
            for loc in locs:
                top, right, bottom, left = loc
                cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
            
            # Show the image with the rectangle for verification
            cv2.imshow("Image", image)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        else:
            print(f"No face found in image {img}. Skipping.")
    
    # Save encodings to file
    saveEncodings(known_encodings, known_names, saveLocation)



def processDatasetImages(path="C:/Blog/Blog 31/Face-Recognition-based-Image-Separator-master/Dataset/", saveLocation="./dataset_encodings.pickle"):
    """
    Process image in dataset from where you want to separate images.
    It separates the images into directories of known people, groups, and any unknown people images.
    """

    # Read pickle file for known people to compare faces from
    people_encodings, names = readEncodingsPickle("./known_encodings.pickle")
    
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)

        # Read image
        image = cv2.imread(imgPath)
        if image is None:
            print(f"Error: Image {img} not found or failed to load.")
            continue  # Skip to the next image if failed to load
        
        orig = image.copy()
        
        # Resize
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        
        # Get locations and encodings
        encs, locs = createEncodings(image)
        
        # Save image to a group image folder if more than one face is in the image
        if len(locs) > 1:
            saveImageToDirectory(orig, "Group", img)
        
        # Processing image for each face
        knownFlag = False
        for i, loc in enumerate(locs):
            top, right, bottom, left = loc
            unknown_encoding = encs[i]
            
            acceptBool, duplicateName, distance = compareFaceEncodings(unknown_encoding, people_encodings, names)
            if acceptBool:
                saveImageToDirectory(orig, duplicateName, img)
                knownFlag = True
            else:
                # Draw a rectangle around unrecognized faces
                cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
        
        if knownFlag:
            print("Match Found")
        else:
            saveImageToDirectory(orig, "Unknown", img)
        
        # Show Image
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        
def main():
    datasetPath="C:/Blog/Blog 31/Face-Recognition-based-Image-Separator-master/Dataset/"
    peoplePath="C:/Blog/Blog 31/Face-Recognition-based-Image-Separator-master/People/"
    processKnownPeopleImages(path=peoplePath)
    processDatasetImages(path=datasetPath)
    print("Completed")

if __name__=="__main__":
    main()