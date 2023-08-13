import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from threading import Thread
import pyttsx3
import csv
import os.path
import time
import shutil
import cv2
import pygame
import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import pickle
from PIL import ImageTk, Image
from tkinter import *
import tkinter.filedialog as filedialog
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from playsound import playsound
import datetime
import pandas as pd 
from keras.models import load_model
from PIL import Image, ImageTk
from sklearn.preprocessing import LabelEncoder

#Xay dung ham
def save_data():
    student_id = txt.get()
    student_name = txt2.get()
    class_name = txt3.get()

    file_exists = os.path.isfile('data.csv')

    with open('data.csv', 'a', newline='') as file:
        fieldnames = ['student_id', 'student_name', 'class_name']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Nếu tệp CSV không tồn tại, thêm tiêu đề cho tệp
        if not file_exists:
            writer.writeheader()

        # Viết thông tin của sinh viên vào tệp
        writer.writerow({'student_id': student_id, 'student_name': student_name, 'class_name': class_name})

    txt.delete(0, tk.END)
    txt2.delete(0, tk.END)
    txt3.delete(0, tk.END)

def record_student_videos():
    # Load sound file
    sound_path = "sound.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)

    # Wait for sound file to finish playing
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass

    # load student names from CSV file
    with open('student_classes.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        student_names = [row[1] for row in reader]

    # create videos directory if not exists
    if not os.path.exists('videos'):
        os.mkdir('videos')

    # loop through each student name and record a video
    for student_name in student_names:
        # create directory for student if not exists
        student_dir = os.path.join('videos', student_name)
        if not os.path.exists(student_dir):
            os.mkdir(student_dir)
        
        video_path = os.path.join(student_dir, f"{student_name}.mp4")
        if os.path.exists(video_path):
            print(f"Video for {student_name} already exists, skipping...")
            continue

        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        # record video for 1 minute and 10 seconds
        start_time = time.time()
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                out.write(frame)
                elapsed_time = time.time() - start_time
                cv2.putText(frame, f"Recording time: {int(elapsed_time)}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('frame',frame)
                if elapsed_time >= 70:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Video for {student_name} has been saved.")

def capture_faces_from_videos():
    # load face classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # create dataset folder to store captured face images
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # loop through each directory in the videos folder
    for subdir, dirs, files in os.walk('videos'):
        # loop through each video file in the directory
        for file in files:
            # check if file is a video file
            if file.endswith('.mp4') or file.endswith('.avi'):
                # create subfolder with the video file name to store captured images
                video_name = os.path.splitext(file)[0]
                video_folder = os.path.join('dataset', video_name)
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

                # check if images for the video have already been captured
                if len(os.listdir(video_folder)) > 0:
                    continue

                # initialize video capture object
                video_file = os.path.join(subdir, file)
                cap = cv2.VideoCapture(video_file)

                # initialize counter to keep track of captured images
                count = 0

                while True:
                    # read a frame from the video
                    ret, frame = cap.read()

                    # if there's an error or the video has ended, break out of the loop
                    if not ret:
                        break

                    # detect faces in the frame
                    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

                    # loop through each detected face and save the image
                    for (x, y, w, h) in faces:
                        # save the face image to the video subfolder
                        cv2.imwrite(os.path.join(video_folder, 'face_{}.jpg'.format(count)), frame[y:y+h, x:x+w])

                        # increment the counter
                        count += 1

                    # display the frame
                    cv2.imshow('frame', frame)

                    # if the user presses the 'q' key, break out of the loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # if we've captured 500 images, break out of the loop
                    if count == 500:
                        break

                # release the video capture object and close the window
                cap.release()
                cv2.destroyAllWindows()

def split_data():
    # đường dẫn đến thư mục chứa dữ liệu
    base_dir = "dataset"

    #tạo danh sách các lớp từ thư mục chứa dữ liệu
    classes = os.listdir(base_dir)

    # Sử dụng LabelEncoder để mã hóa các nhãn
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(classes)

    # tạo hai thư mục "train" và "validation"
    train_dir = os.path.join("ExpData", "train")
    val_dir = os.path.join("ExpData", "validation")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    #vòng lặp qua các lớp
    for i, cl in enumerate(classes):
        # tạo đường dẫn đến thư mục ảnh của lớp hiện tại
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        print("{}: {} Images".format(cl, len(images)))
        # chia dữ liệu thành tập train và tập validation
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        # tạo thư mục tương ứng với lớp hiện tại trong thư mục train và validation
        cl_train_dir = os.path.join(train_dir, str(i))
        cl_val_dir = os.path.join(val_dir, str(i))
        if not os.path.exists(cl_train_dir):
            os.mkdir(cl_train_dir)
        if not os.path.exists(cl_val_dir):
            os.mkdir(cl_val_dir)
        # sao chép các tập train và validation tương ứng vào các thư mục vừa tạo
        for file in train_images:
            img = cv2.imread(file, 1)
            resized = cv2.resize(img, (150,150))
            cv2.imwrite(os.path.join(cl_train_dir, os.path.basename(file)), resized)
        for file in val_images:
            img = cv2.imread(file, 1)
            resized = cv2.resize(img, (150,150))
            cv2.imwrite(os.path.join(cl_val_dir, os.path.basename(file)), resized)

    # Lưu trữ thông tin mã hóa nhãn vào một tệp pickle
    with open("ExpData/labels.pkl", "wb") as f:
        pickle.dump(encoded_labels, f)

def run_image_classification_tf():
    # Data Generators
    batch_size = 128
    train_url = 'ExpData/train'
    validation_url = 'ExpData/validation'
    
    train = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
    )

    validation = ImageDataGenerator(rescale=1./255)

    train_dataset = train.flow_from_directory(
        train_url,
        target_size = (150,150),  # thay đổi kích thước ảnh đầu vào thành (224,224)
        batch_size = batch_size,
        class_mode = 'categorical'
    )

    validation_dataset = validation.flow_from_directory(
        validation_url,
        target_size = (150,150),  # thay đổi kích thước ảnh đầu vào thành (224,224)
        batch_size = batch_size,
        class_mode = 'categorical'
    )
    
    # Model
    input_shape = (150, 150, 3) # kích thước ảnh đầu vào
    num_classes = train_dataset.num_classes # số lượng lớp
    learning_rate = 0.001 # hệ số học tập

    base_model = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model_mobilenet = Model(inputs=base_model.input, outputs=predictions)

    model_mobilenet.summary()
    model_mobilenet.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train Model
    start_time = time.time()
    history = model_mobilenet.fit(train_dataset,batch_size=batch_size,epochs=50,verbose=1,validation_data=validation_dataset)
    end_time = time.time()
    training_time = end_time - start_time

    training_time_str = str(datetime.timedelta(seconds=training_time))
    print('Thời gian huấn luyện:', training_time_str)

    scores = model_mobilenet.evaluate(validation_dataset)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model_mobilenet.save('ExpData/Final_test_MobileNet.h5')

    # Plot Results
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()

def train_model_cnn():
    batch_size = 128
    train_url = 'ExpData/train'
    validation_url  = 'ExpData/validation'
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_dataset = train_datagen.flow_from_directory(
        train_url,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_dataset = validation_datagen.flow_from_directory(
        validation_url,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='categorical'
    )

    input_shape = (150,150,3)
    num_classes = train_dataset.num_classes
    learning_rate = 0.001

    model = Sequential()
    model.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same',input_shape=(150,150,3)))
    model.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
    model.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
    model.add(Conv2D(128,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(256,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
    model.add(Conv2D(256,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu',kernel_initializer='he_uniform',))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu',kernel_initializer='he_uniform',))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))

    model.summary()

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])

    start_time = time.time()
    history = model.fit(train_dataset, batch_size=batch_size, epochs=50, verbose=1, validation_data=validation_dataset)
    end_time = time.time()
    training_time = end_time - start_time

    training_time_str = str(datetime.timedelta(seconds=training_time))
    print('Thời gian huấn luyện:', training_time_str)

    scores = model.evaluate(validation_dataset)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    model.save('ExpData/Final_test_CNN.h5')

    # Plot Results
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()

def copy_class_images():
    # create directory for test images if not exists
    if not os.path.exists('ExpData/test'):
        os.makedirs('ExpData/test')

    # load class names from CSV file
    with open('student_classes.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        class_names = [row[1] for row in reader]

    # loop through each class name and copy the first image to the test directory
    for class_name in class_names:
        # create directory for class if not exists
        class_dir = os.path.join('ExpData/test', class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # check if an image already exists for this class
        img_path = os.path.join(class_dir, f"{class_name}.jpg")
        if os.path.exists(img_path):
            print(f"Image for {class_name} already exists, skipping...")
            continue

        # get the first image for this class
        class_images_dir = os.path.join('ExpData/train', class_name)
        class_images = os.listdir(class_images_dir)
        if len(class_images) > 0:
            src_path = os.path.join(class_images_dir, class_images[0])
            shutil.copy(src_path, img_path)
            print(f"Image for {class_name} has been copied.")
        else:
            print(f"No image found for {class_name}")

def predict_and_display_results():
    # Load the trained model
    model = load_model('ExpData/Final_test.h5')

    # load label pickle file
    with open('ExpData/classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    # Define the directory path containing the test images
    test_dir = 'ExpData/test'

    # Load student names from student_classes.csv
    student_names = []
    with open('student_classes.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            student_names.append(row[1])
    
    pygame.mixer.init()
    sound = pygame.mixer.Sound('soundGreeting.mp3')

    # Create a new window to display the results
    results_window = tk.Toplevel(root)
    results_window.title('Prediction Results')

    # Create a table to display the results
    table = ttk.Treeview(results_window, columns=('Student ID', 'Student Name', 'Class Name', 'Image Name', 'Prediction', 'Time'))
    table.heading('Student ID', text='Student ID')
    table.heading('Student Name', text='Student Name')
    table.heading('Class Name', text='Class Name')
    table.heading('Image Name', text='Image Name')
    table.heading('Prediction', text='Prediction')
    table.heading('Time', text='Time')
    table.pack(fill='both', expand=True)

    # Define the path for the CSV file
    csv_path = 'ExpData/predicted_images.csv'

    # Load the list of predicted images from the CSV file if it exists
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            predicted_images = [row[0] for row in reader]
    else:
        predicted_images = []
    
    # Initialize the total number of images to 0
    total_images = 0

    # Loop through all the subdirectories in the test directory
    for sub_dir in os.listdir(test_dir):
        sub_dir_path = os.path.join(test_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue  # Skip non-directory files

        # Loop through all the images in the subdirectory and make predictions
        for filename in os.listdir(sub_dir_path):
            # Increase the total number of images by 1
            total_images += 1

            # Check if the image has already been predicted
            if os.path.join(sub_dir, filename) in predicted_images:
                continue  # Skip this image

            # Load the image and preprocess it
            img_path = os.path.join(sub_dir_path, filename)
            img = load_img(img_path, target_size=(224, 224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            sound.play()

            while pygame.mixer.get_busy():
                time.sleep(0.1)

            '''# Make a prediction and print the label
            start_time = time.time()
            preds = model.predict(x)
            end_time = time.time()
            label_idx = np.argmax(preds, axis=1)[0]
            label = classes[label_idx]'''

            # Make a prediction and print the label
            preds = model.predict(x)
            label_idx = np.argmax(preds, axis=1)[0]
            label = classes[label_idx]

            # Check if student name in student_classes.csv matches label
            if label in student_names:
                # Find the student's row in student_classes.csv
                with open('student_classes.csv', 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip header row
                    for row in reader:
                        if row[1] == label:
                            student_id = row[0]
                            class_name = row[2]
                            break

                '''#Add the results to the table
                elapsed_time = round(end_time - start_time, 2)
                table.insert('', 'end', values=(student_id, label, class_name, os.path.join(sub_dir, filename), label, elapsed_time))'''

                # Add the results to the table
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                table.insert('', 'end', values=(student_id, label, class_name, os.path.join(sub_dir, filename), label, current_time))

                # Play sound if prediction is correct
                if label == sub_dir:
                    sound.play()
                
                # Add the image to the list of predicted images
                predicted_images.append(os.path.join(sub_dir, filename))

    # Write the list of predicted images to the CSV file
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image'])
        writer.writerows([[image] for image in predicted_images])
    
    '''# Print the total number of images
    print('Total number of images processed:', total_images)'''

    message = f"{total_images} images have been processed."
    message_label = tk.Label(results_window, text=message)
    message_label.pack()
        
    # Resize the columns to fit the data
    for column in table['columns']:
        table.column(column, width=200, stretch=True)

def drowsiness_detection():
    model = load_model('ExpData/Final_test_1.h5')
    path = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # load label pickle file
    with open('ExpData/classes_1.pkl', 'rb') as f:
        classes = pickle.load(f)

    # initialize text-to-speech engine
    engine = pyttsx3.init()

    def play_sound(status):
        engine.say("Hello, " + status)  # read out the name using text-to-speech
        engine.runAndWait()

    while True:
        Threshold= 5
        ret,frame = cap.read()
        frame = cv2.flip(frame, 1)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(faceCascade.empty())
        faces = faceCascade.detectMultiScale(frame,1.3,5,minSize=(100,100))

        if len(faces) == 0:
            status = 'no face'
        else:
            for x,y,w,h in faces:
                #roi_gray  = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                faces = faceCascade.detectMultiScale (roi_color)
                for (ex, ey, ew, eh) in faces:
                    face_roi = roi_color[ey: ey+eh, ex:ex + ew]
                    new_array = cv2.resize(face_roi, (150, 150))
                    X_input = np.array(new_array).reshape(-1,150, 150, 3).astype('float64')
                    Predict=np.argmax(model.predict(X_input),axis = -1)
                    status = classes[int(Predict)]
                    t = Thread(target=play_sound, args=(status,))
                    t.start()

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Use putText() method for
        # inserting text on video
        cv2.putText(frame,status,(100, 100),font,3,(0, 255, 0),2,cv2.LINE_AA)
        cv2.imshow('Drowsiness Detection Tutorial', frame)
        if cv2.waitKey(2) & 0xFF== ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tạo một instance của lớp Tk
root = tk.Tk()

# Tạo cửa sổ con
popup = tk.Toplevel()
popup.title("Thông báo")

# Thêm một nhãn vào cửa sổ con
message = "Chào mừng bạn đến với ứng dụng Tkinter của tôi!"
label = tk.Label(popup, text=message, padx=10, pady=10)
label.pack()

# Thêm một nút để tắt cửa sổ con
button = tk.Button(popup, text="OK", command=popup.destroy)
button.pack()

# Hiển thị cửa sổ con
popup.grab_set()
root.wait_window(popup)

# Đặt kích thước của cửa sổ chính
root.geometry("800x480")
root.resizable(True,True)
root.configure(background='#355454')

#main window------------------------------------------------
message3 = tk.Label(root, text="ỨNG DỤNG XÁC THỰC KHUÔN MẶT" ,fg="white",bg="#355454" ,width=60 ,height=1,font=('times', 29, ' bold '))
message3.place(x=10, y=10,relwidth=1)

#frames-------------------------------------------------
frame1 = tk.Frame(root, bg="white")
frame1.place(relx=0.11, rely=0.15, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(root, bg="white")
frame2.place(relx=0.51, rely=0.15, relwidth=0.39, relheight=0.80)

#frame_headder
fr_head1 = tk.Label(frame1, text="ĐĂNG KÝ KHUÔN MĂT", fg="white",bg="black" ,font=('times', 17, ' bold ') )
fr_head1.place(x=0,y=0,relwidth=1)

fr_head2 = tk.Label(frame2, text="XÁC THỰC THÔNG TIN", fg="white",bg="black" ,font=('times', 17, ' bold ') )
fr_head2.place(x=0,y=0,relwidth=1)

#registretion frame
lbl = tk.Label(frame1, text="Mã sinh viên:",width=20  ,height=1  ,fg="black"  ,bg="white" ,font=('times', 17, ' bold ') )
lbl.place(x=-70, y=40)

txt = tk.Entry(frame1,width=32 ,fg="black",bg="#e1f2f2",highlightcolor="#00aeff",highlightthickness=3,font=('times', 15, ' bold '))
txt.place(x=145, y=40,relwidth=0.5)

lbl2 = tk.Label(frame1, text="Họ và tên:",width=20  ,fg="black"  ,bg="white" ,font=('times', 17, ' bold '))
lbl2.place(x=-85, y=80)

txt2 = tk.Entry(frame1,width=32 ,fg="black",bg="#e1f2f2",highlightcolor="#00aeff",highlightthickness=3,font=('times', 15, ' bold ')  )
txt2.place(x=145, y=80,relwidth=0.5)

lbl3 = tk.Label(frame1, text="Lớp:",width=20  ,fg="black"  ,bg="white" ,font=('times', 17, ' bold '))
lbl3.place(x=-110, y=120)

txt3 = tk.Entry(frame1,width=32 ,fg="black",bg="#e1f2f2",highlightcolor="#00aeff",highlightthickness=3,font=('times', 15, ' bold ')  )
txt3.place(x=145, y=120,relwidth=0.5)

message0=tk.Label(frame1,text="Thực hiện các bước từ trái -> phải!",bg="white" ,fg="black"  ,width=39 ,height=1,font=('times', 16, ' bold '))
message0.place(x=-80,y=180)

#Attendance frame
'''lbl3 = tk.Label(frame2, text="Attendance Table",width=20  ,fg="black"  ,bg="white"  ,height=1 ,font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)'''

#BUTTONS----------------------------------------------

saveInfo = tk.Button(frame1, text="Lưu thông tin", command=save_data, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
saveInfo.place(x=7, y=210,relwidth=0.45)

recordVid = tk.Button(frame1, text="Quay video", command=record_student_videos, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
recordVid.place(x=160, y=210,relwidth=0.45)

capfaceVid = tk.Button(frame1, text="Tạo dữ liệu", command=capture_faces_from_videos, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
capfaceVid.place(x=7, y=260,relwidth=0.45)

splitData = tk.Button(frame1, text="Xử lý dữ liệu", command=split_data, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
splitData.place(x=160, y=260,relwidth=0.45)

trainData = tk.Button(frame1, text="Huấn luyện TF", command=run_image_classification_tf, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
trainData.place(x=1, y=310,relwidth=0.5)

trainDataCnn = tk.Button(frame1, text="Huấn luyện CNN", command=train_model_cnn, fg="black", bg="#00aeff", width=34, height=1, activebackground = "white", font=('times', 16, ' bold '))
trainDataCnn.place(x=155, y=310,relwidth=0.5)

trackImg = tk.Button(frame2, text="Lấy dữ liệu xác thực", command=copy_class_images, fg="black", bg="#00aeff", height=1, activebackground = "white" ,font=('times', 16, ' bold '))
trackImg.place(x=7,y=40,relwidth=0.6)

showInfo = tk.Button(frame2, text="Kiểm tra", command=predict_and_display_results, fg="black", bg="#00aeff", height=1, activebackground = "white" ,font=('times', 16, ' bold '))
showInfo.place(x=200,y=40,relwidth=0.3)

cameraImg = tk.Button(frame2, text="Camera xác thực", command=drowsiness_detection, fg="black", bg="#00aeff", height=1, activebackground = "white" ,font=('times', 16, ' bold '))
cameraImg.place(x=7,y=100,relwidth=0.6)

class App:
    def __init__(self, master):
        # Load animation GIF file
        self.anim = Image.open("gif3.gif")
        self.frames = []
        try:
            while True:
                frame = self.anim.copy()
                frame = frame.resize((200, 200)) # Thay đổi kích thước của frame
                self.frames.append(ImageTk.PhotoImage(frame))
                self.anim.seek(len(self.frames)) # Đọc từng frame
        except EOFError:
            pass

        # Tạo widget Label để hiển thị animation và đặt vị trí ở giữa cửa sổ giao diện
        self.label = tk.Label(master)
        self.label.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

        # Set animation as first frame
        self.idx = 0
        self.label.config(image=self.frames[self.idx])
        self.animate()

    def animate(self):
        self.idx += 1
        if self.idx == len(self.frames):
            self.idx = 0
        self.label.config(image=self.frames[self.idx])
        self.label.after(100, self.animate)

app = App(frame2)

# Bắt đầu vòng lặp chính của ứng dụng và hiển thị giao diện lên màn hình
root.mainloop()
