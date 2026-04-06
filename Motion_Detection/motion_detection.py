import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import datetime
import os
import time
RECORD_DIR = &quot;recordings&quot;
SNAP_DIR = &quot;snapshots&quot;
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)
WIDTH = 640
HEIGHT = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
model = YOLO(&quot;yolov8n.pt&quot;)
fgbg = cv2.createBackgroundSubtractorMOG2()
recording = False
manual_recording = False
out = None

last_motion_time = 0
zone_points = []
root = tk.Tk()
root.title(&quot;Ultimate AI Motion Detection System&quot;)
video_label = tk.Label(root)
video_label.grid(row=0, column=0, columnspan=3)
min_area = tk.IntVar(value=800)
threshold_val = tk.IntVar(value=25)
record_duration = tk.IntVar(value=10)
ai_enabled = tk.BooleanVar(value=True)
ttk.Label(root, text=&quot;Min Motion Area&quot;).grid(row=1,column=0)
ttk.Scale(root, from_=100, to=5000, variable=min_area,
orient=&quot;horizontal&quot;).grid(row=1,column=1)
ttk.Label(root, text=&quot;Threshold&quot;).grid(row=2,column=0)
ttk.Scale(root, from_=1, to=100, variable=threshold_val,
orient=&quot;horizontal&quot;).grid(row=2,column=1)
ttk.Label(root, text=&quot;Record Seconds&quot;).grid(row=3,column=0)
ttk.Scale(root, from_=5, to=60, variable=record_duration,
orient=&quot;horizontal&quot;).grid(row=3,column=1)
ttk.Checkbutton(root, text=&quot;Enable AI Person Detection&quot;,
variable=ai_enabled).grid(row=4,column=0)
def snapshot():
ret, frame = cap.read()
if ret:
name = datetime.datetime.now().strftime(&quot;%Y%m%d_%H%M%S&quot;) + &quot;.jpg&quot;
path = os.path.join(SNAP_DIR, name)
cv2.imwrite(path, frame)
def toggle_record():
global manual_recording
manual_recording = not manual_recording
def open_recordings():
files = os.listdir(RECORD_DIR)

win = tk.Toplevel(root)
win.title(&quot;Recordings&quot;)
for f in files:
ttk.Label(win, text=f).pack()
ttk.Button(root,text=&quot;Snapshot&quot;,command=snapshot).grid(row=5,column=0)
ttk.Button(root,text=&quot;Manual Record&quot;,command=toggle_record).grid(row=5,column=1)
ttk.Button(root,text=&quot;Browse
Recordings&quot;,command=open_recordings).grid(row=5,column=2)
def process_frame(frame):
motion_detected = False
mask = fgbg.apply(frame)
_, thresh = cv2.threshold(mask, threshold_val.get(), 255, cv2.THRESH_BINARY)
thresh = cv2.dilate(thresh,None,iterations=2)
contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
if cv2.contourArea(c) &gt; min_area.get():
motion_detected = True
x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
return frame, motion_detected
def ai_detection(frame):
results = model(frame, verbose=False)
person_found = False
for r in results:
boxes = r.boxes
for b in boxes:
cls = int(b.cls)
if cls == 0:
person_found = True
x1,y1,x2,y2 = map(int,b.xyxy[0])
cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
cv2.putText(frame,&quot;Person&quot;,(x1,y1-10),
cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
return frame, person_found

def update_frame():
global recording, out, last_motion_time
ret, frame = cap.read()
if not ret:
root.after(10,update_frame)
return
frame = cv2.resize(frame,(WIDTH,HEIGHT))
original = frame.copy()
frame, motion = process_frame(frame)
person = False
if ai_enabled.get():
frame, person = ai_detection(frame)
trigger = motion or person or manual_recording
if trigger:
last_motion_time = time.time()
if not recording:
name = datetime.datetime.now().strftime(&quot;%Y%m%d_%H%M%S&quot;)+&quot;.avi&quot;
path = os.path.join(RECORD_DIR,name)
fourcc = cv2.VideoWriter_fourcc(*&#39;XVID&#39;)
out = cv2.VideoWriter(path,fourcc,20,(WIDTH,HEIGHT))
recording = True
if recording:
out.write(original)
if time.time()-last_motion_time &gt; record_duration.get() and not manual_recording:
recording = False
out.release()
frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
img = Image.fromarray(frame_rgb)
imgtk = ImageTk.PhotoImage(image=img)
video_label.imgtk = imgtk
video_label.configure(image=imgtk)
root.after(10,update_frame)

update_frame()
root.mainloop()
cap.release()