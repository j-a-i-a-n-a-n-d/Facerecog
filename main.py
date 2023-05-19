import tkinter
from PIL import Image, ImageTk
from tkinter import *
from tkinter import messagebox
import os
import cv2
import csv
import numpy as np
import pandas as pd
import datetime
import time
from threading import Thread
from PIL import Image
from tkinter import filedialog
global window


def quit_window():
    if messagebox.askyesno("Quit", "You pressed the exit tab.Do you really want to quit to quit?"):
        window.destroy()


def contact():
    messagebox._show(title="Contact Me",
                     message="Find my contact attached in any of the files")


def about():
    messagebox._show(
        title="About", message="Simplying Attendance during Examinationz")

# webcam check


def webcamCheck():
    import cv2
    face_cascade = cv2.CascadeClassifier(
        'D:/Python_FaceDetection_Recognition/haarcascade_frontalface_default.xml')
    capture = cv2.VideoCapture(0)  # "https://192.168.1.23:8080/video"
    while True:
        _, img = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
        cv2.imshow('Webcam Check', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

# member window


def memberWindow():

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def captureFace():
        def submit():
            Id = Id_var.get()
            name = Name_var.get()
            Name_var.set("")
            Id_var.set("")
            if (is_number(Id) and name.isalpha()):
                cam = cv2.VideoCapture(0)
                harcascadePath = "D:/Python_FaceDetection_Recognition/haarcascade_frontalface_default.xml"
                detector = cv2.CascadeClassifier(harcascadePath)
                sampleNum = 0
                while (True):
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(
                        gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h),
                                      (10, 159, 255), 2)
                        sampleNum = sampleNum+1
                        cv2.imwrite("D:/Python_FaceDetection_Recognition/TrainingImages" + os.sep + name + "."+Id + '.' +
                                    str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                        cv2.imshow('frame', img)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
                    elif sampleNum > 20:  # sample limit 100
                        break
                loginframe.destroy()
                cam.release()
                cv2.destroyAllWindows()
                # res = "Images Saved for ID : " + Id + " Name : " + name
                header = ["Id", "Name"]
                row = [Id, name]
                if (os.path.isfile("D:/Python_FaceDetection_Recognition/StudentDetails.csv")):
                    with open("D:/Python_FaceDetection_Recognition/StudentDetails/StudentDetails.csv", 'a+') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(j for j in row)
                    csvFile.close()
                else:
                    with open("D:/Python_FaceDetection_Recognition/StudentDetails/StudentDetails.csv", 'a+') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(i for i in header)
                        writer.writerow(j for j in row)
                    csvFile.close()
            else:
                if (is_number(Id)):
                    messagebox.showwarning(
                        "Warning", "Enter Alphabetical Name without white spaces or numbers!", parent=Memberwindow)
                    captureFace()
                if (name.isalpha()):
                    messagebox.showwarning(
                        "Warning", "NUMERIC VALUE EXPECTED", parent=Memberwindow)
                    captureFace()

        loginframe = tkinter.Frame(Memberwindow, bg="black")
        loginframe.place(relx=0.5, rely=0.23, relwidth=0.40, relheight=0.52)
        IDLabel = Label(loginframe, text="Enter your ID", fg="white",
                        bg="#344A70", width=11, font=('times', 20, ' bold '))
        IDLabel.place(relx=0.05, rely=0.25)
        Id_var = StringVar()
        IDEntry = Entry(loginframe, textvariable=Id_var,
                        font=('times', 15, ' bold '))
        IDEntry.place(relx=0.5, rely=0.25, relwidth=0.45, relheight=0.1)
        NAMELabel = Label(loginframe, text="Enter your Name", fg="white",
                          bg="#344A70", width=13, font=('times', 20, ' bold '))
        NAMELabel.place(relx=0.04, rely=0.5)
        Name_var = StringVar()
        NAMEEntry = Entry(loginframe, textvariable=Name_var,
                          font=('times', 15, ' bold '))
        NAMEEntry.place(relx=0.5, rely=0.5, relwidth=0.45, relheight=0.1)
        SubmitButton = Button(loginframe, text="Submit",
                              command=submit, font=('times', 20, ' bold '))
        SubmitButton.place(relx=0.45, rely=0.75)

    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # create empty face list
        faces = []
        # create empty ID list
        Ids = []
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(Id)
        return faces, Ids

    # Function to train images saved in the desired file location

    def TrainImages():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        harcascadePath = "D:/Python_FaceDetection_Recognition/haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels(
            "D:/Python_FaceDetection_Recognition/TrainingImages")
        Thread(target=recognizer.train(faces, np.array(Id))).start()
        recognizer.save(
            "D:/Python_FaceDetection_Recognition/TrainingImageLabel"+os.sep+"Trainner.yml")
        messagebox.showinfo(
            "Images Trained", "Images Successfully Trained!", parent=Memberwindow)

    def recognize_attendence():
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        recognizer.read(
            "D:/Python_FaceDetection_Recognition/TrainingImageLabel/Trainner.yml")
        harcascadePath = "D:/Python_FaceDetection_Recognition/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        df = pd.read_csv("D:/Python_FaceDetection_Recognition/StudentDetails" +
                         os.sep+"StudentDetails.csv")
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height
        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            _, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize=(
                int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 100:
                    aa = df.loc[df['Id'] == Id]['Name'].values
                    confstr = "  {0}%".format(round(100 - conf))
                    tt = str(Id)+"-"+aa
                else:
                    Id = '  Unknown  '
                    tt = str(Id)
                    confstr = "  {0}%".format(round(100 - conf))

                if (100-conf) > 67:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(
                        ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(
                        ts).strftime('%H:%M:%S')
                    aa = str(aa)[2:-2]
                    attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

                tt = str(tt)[2:-2]
                print(type(tt),tt)
                if (100-conf) > 67:
                    tt = tt + " [Pass]"
                    cv2.putText(im, str(tt), (x+5, y-5),
                                font, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(im, str(tt), (x + 5, y - 5),
                                font, 1, (255, 255, 255), 2)

                if (100-conf) > 67:
                    cv2.putText(im, str(confstr), (x + 5, y + h - 5),
                                font, 1, (0, 255, 0), 1)
                elif (100-conf) > 50:
                    cv2.putText(im, str(confstr), (x + 5, y + h - 5),
                                font, 1, (0, 255, 255), 1)
                else:
                    cv2.putText(im, str(confstr), (x + 5, y + h - 5),
                                font, 1, (0, 0, 255), 1)

            attendance = attendance.drop_duplicates(
                subset=['Id'], keep='first')
            cv2.imshow('Attendance', im)
            if (cv2.waitKey(1) == ord('q')):
                break
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        fileName = "D:/Python_FaceDetection_Recognition/Attendance"+os.sep + \
            "Attendance_"+date+"_"+Hour+"-"+Minute+".csv"
        attendance.to_csv(fileName, index=False)
        messagebox.showinfo("Attendance Successful",
                            "Attendance Successful!", parent=Memberwindow)
        cam.release()
        cv2.destroyAllWindows()
    Memberwindow = Toplevel()
    Memberwindow.title("Member")
    Memberwindow.geometry("1280x720")
    Memberwindow.resizable(True, True)
    Memberwindow.configure(background='#344A70')

    message = tkinter.Label(Memberwindow, text="Member Login", fg="black",
                            bg="#344A70", width=60, height=1, font=('times', 29, ' bold '))
    message.place(x=5, y=20, relwidth=1)
    img1 = Image.open(
        "D:/Python_FaceDetection_Recognition/images/memberEmoticon.png")
    resized_image = img1.resize((400, 350), Image.LANCZOS)
    new_image = ImageTk.PhotoImage(resized_image)
    label1 = Label(Memberwindow, image=new_image)
    label1.image = new_image
    label1.place(relx=0.05, rely=0.23)
    frame1 = tkinter.Frame(Memberwindow, bg="black")
    frame1.place(relx=0.5, rely=0.23, relwidth=0.40, relheight=0.52)

    check = tkinter.Button(frame1, text="Check WebCam", command=webcamCheck, fg="black",
                           bg="light grey", activebackground="white", font=('times', 20, ' bold '))
    check.place(x=130, y=30, relwidth=0.55)
    cap_faces = tkinter.Button(frame1, text="Capture Your Face", command=captureFace,
                               fg="black", bg="light grey", activebackground="white", font=('times', 20, ' bold '))
    cap_faces.place(x=130, y=110, relwidth=0.55)
    train = tkinter.Button(frame1, text="Train Your Images", command=TrainImages, fg="black",
                           bg="light grey", activebackground="white", font=('times', 20, ' bold '))
    train.place(x=130, y=190, relwidth=0.55)
    recog_att = tkinter.Button(frame1, text="Click For Attendance", command=recognize_attendence,
                               fg="black", bg="light grey", activebackground="white", font=('times', 20, ' bold '))
    recog_att.place(x=130, y=270, relwidth=0.55)

    def homewindow():
        Memberwindow.destroy()
    HomeButton = Button(Memberwindow, text="Home", font=(
        'times', 20, ' bold '), command=homewindow)
    HomeButton.place(relx=0.67, rely=0.8)


# Function for the main window

def mainWindow():

    menubar = Menu(window)
    help = Menu(menubar, tearoff=0)
    help.add_command(label="Contact Us", command=contact)
    help.add_separator()
    help.add_command(label="Exit", command=quit_window)
    menubar.add_cascade(label="Help", menu=help)
    menubar.add_command(label="About", command=about)
    window.config(menu=menubar)
    message3 = tkinter.Label(window, text="Simplying Attendance During Examination",
                             fg="black", bg="#344A70", width=60, height=1, font=('times', 29, ' bold '))
    message3.place(x=5, y=20, relwidth=1)
    frame1 = tkinter.Frame(window, bg="black")
    frame1.place(relx=0.29, rely=0.23, relwidth=0.39, relheight=0.52)

    member = tkinter.Button(frame1, text="MEMBER", fg="black", command=memberWindow,
                            bg="light grey", width=11, activebackground="white", font=('times', 22, ' bold '))
    member.place(x=110, y=110, relwidth=0.55, relheight=0.15)

    quit = tkinter.Button(frame1, text="QUIT", command=quit_window, fg="black",
                          bg="light grey", width=11, activebackground="white", font=('times', 22, ' bold '))
    quit.place(x=110, y=270, relwidth=0.55, relheight=0.15)

    window.protocol("WM_DELETE_WINDOW", quit_window)
    window.mainloop()


window = tkinter.Tk()
window.title("Simplying Attendance During Examination")
window.geometry("1280x720")
window.resizable(True, True)
window.configure(background='#344A70')
mainWindow()
