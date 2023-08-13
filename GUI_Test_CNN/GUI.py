import os
import csv
import time
import cv2
import pygame

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
        
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = f"{student_name}.mp4"
        video_path = os.path.join(student_dir, video_name)
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
    else:
        print(f"Video for {student_name} already exists, skipping...")
