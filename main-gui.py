from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import *
from tkinter import filedialog
from tkinter import *
import imutils
import cv2
from PIL import Image, ImageTk
import os,time
import argparse

#Setup
white 		= "#ffffff"
lightBlue2 	= "#adc5ed"
font 		= "Constantia"
fontButtons = (font, 12)
maxWidth  	= 800
maxHeight 	= 480



def openImageMaskDetector(): 

    def ImageMaskDetector():
        imagepath = EntryBox.get()
        if(imagepath == ""):
            Display_area.config(state=NORMAL)
            Display_area.insert(END, "\n"+"INFO:"+"Error 404 : No Such Image Found "+"\n",'warning')
            Display_area.yview(END)


        Display_area.config(state=NORMAL)
        Display_area.insert(END,"\n"+"[INFO] Loading Model....."+'\n')
        Display_area.yview(END)
        
       

        prototxt = "face_detector/deploy.prototxt"
        weightfile = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

        Display_area.config(state=NORMAL)
        Display_area.insert(END,"\n"+"[INFO] Model Loaded Successfully...."+'\n','success')
        Display_area.yview(END)

        Display_area.config(state=NORMAL)
        Display_area.insert(END,"\n"+"[INFO] computing face detections..."+'\n','success')
        Display_area.yview(END)


        Display_area.config(state=NORMAL)
        Display_area.insert(END,"\n"+"[INFO] Detecting Mask from Image...."+'\n','success')
        Display_area.yview(END)
        # Load Preconfigured readNet 
        net = cv2.dnn.readNet(prototxt, weightfile)

        # Load model file 
        modelpath = "mask_detector.model"
        model = load_model(modelpath)
        
        

        # Load Image for Detcting Mask 
        image = cv2.imread(imagepath)
        orig = image.copy()
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the face detections
        # print("[INFO] computing face detections...")
        
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                # pass the face through the model to determine if the face
                # has a mask or not
               
                (mask, withoutMask) = model.predict(face)[0]

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        # show the output image
        Display_area.config(state=NORMAL)
        Display_area.insert(END,"\n"+"[INFO] Showing Output...."+'\n','success')
        Display_area.yview(END)
        cv2.imshow("Output", image)
        cv2.waitKey(0)

    def UploadImage():
        path = EntryBox.get()
        frame = cv2.imread(path)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image).resize((300, 250))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

    def askdirectory():
        EntryBox.delete(0,END)
        file_selected = filedialog.askopenfilename()
        EntryBox.insert(0,file_selected) 
      
    # Toplevel object which will  
    # be treated as a new window 
    imagewindow = Toplevel(base) 
    imagewindow.resizable(width=False,height=False)

    # Image Frame Area
    lmain = tk.Label(imagewindow)
    lmain.place(x=100,y=70)

    # Display Area 
    Display_area = ScrolledText(imagewindow,width="58",height=20,bg="#f0f0f0")
    Display_area.tag_config('warning', background='tomato', foreground="black")
    Display_area.tag_config('success', background="SeaGreen1", foreground="black",font=("Times New Roman",14,"bold"))
    Display_area.place(x=10,y=320)

    # sets the title of the Toplevel widget 
    imagewindow.title("Detect face Mask From Images") 
  
    # sets the geometry of toplevel 
    imagewindow.geometry("500x500") 

    #Load Image to be detected
    Label(imagewindow,text="Select Image",bg="wheat1",font=("Verdana",10,"bold")).grid(row=0)

    Browse_btn = Button(imagewindow,text="Load Image", width="12",
                        bd=2, bg="green", activebackground="plum3", fg='#fff',command=askdirectory).place(x=410,y=0)

    Upload = Button(imagewindow, font=("Verdana", 8,"bold"),text="Upload", width="9", height=2,
                    bd=2, bg="palegreen", activebackground="plum1", fg='#000000'
                 ,command = UploadImage).place(x=150,y=30)

    rundetector = Button(imagewindow,text = "Run Detector", font = fontButtons, bg="#28B463", activebackground="#1A5276", fg='#ffffff', width = 10, height= 1,command=ImageMaskDetector)
    rundetector.place(x=250,y=30)

    EntryBox = Entry(imagewindow,width="30",bg="wheat3",fg="black",font=("Verdana",12))
    EntryBox.grid(row=0,column=1)



    # Configure Window
    imagewindow.configure(bg="wheat1")
    imagewindow.mainloop()



def openRealTimeDetector():
    
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)


    # Main Start 
    # Loading precinfigured files and model 
    prototxt = "face_detector/deploy.prototxt"
    weightfile = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    modelpath = "mask_detector.model"
    faceNet = cv2.dnn.readNet(prototxt, weightfile)

    maskNet = load_model(modelpath)

    # Initializing Camera (Web cam)
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()




    


base = tk.Tk()
base.title('Face Mask Detectorr')
base.geometry("500x250")
base.resizable(width=True,height=True)



Detect_Image = Button(base, font=("Verdana", 8,"bold"),text="Detect From Image", width="15", height=2,
                     bd=2, bg="palegreen", activebackground="plum1", fg='#000000'
                 ,command = openImageMaskDetector).place(x=125,y=80)
                 
realtime_detector = Button(base, font=("Verdana", 8,"bold"),text="Real Time Detector", width="15", height=2,
                     bd=2, bg="salmon1", activebackground="khaki2", fg='#000000'
                 ,command = openRealTimeDetector).place(x=280,y=80)

Display_area = ScrolledText(base,width="58",height=5,bg="AntiqueWhite2")
Display_area.place(x=10,y=150)
Display_area.tag_config('cyanbg', background="lightcyan", foreground="black")

Display_area.config(state=NORMAL)
Display_area.insert(END,"[INFO]: WebCam Loader might take a few minutes"+ '\n','cyanbg')
Display_area.yview(END)





base.configure(bg="wheat1")
base.mainloop()