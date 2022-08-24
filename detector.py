import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pafy
import sys

class ai():
    def __init__(self, input_cam, youtube_url = None):
        self.youtube_URL = youtube_url
        self.input_cam = input_cam
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.labels = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def video(self):
        if self.youtube_URL is not None:
            vid = pafy.new(self.youtube_URL).streams[-1]
            assert vid is not None
            return cv2.VideoCapture(vid.url)
        else:
            vid = cv2.VideoCapture(0)
            assert vid is not None
            return vid
    
    def classify(self, vid_frame):
        self.model.to(self.device)
        predictions = self.model([vid_frame])
        predicted_labels, bb_coordinates = predictions.xyxyn[0][:, -1].to(self.device).numpy(), predictions.xyxyn[0][:, :-1].to(self.device).numpy()
        return predicted_labels, bb_coordinates

    def bounding_box(self, predictions, vid_frame):
        predicted_labels, bb_coordinates = predictions
        num_labels = len(predicted_labels)
        for i in range(num_labels):
            label = bb_coordinates[i]
            x1, y1, x2, y2 = int(label[0]*vid_frame.shape[1]), int(label[1]*vid_frame.shape[0]), int(label[2]*vid_frame.shape[1]), int(label[3]*vid_frame.shape[0])
            cv2.rectangle(vid_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vid_frame, f"{self.labels[predicted_labels[i]]} {int(label[4] * 100)} %", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return vid_frame
    
    def main_loop(self):
        video = self.video()
        while True:
            ret, frame = video.read()
            predictions = self.classify(frame)
            frame = self.bounding_box(predictions, frame)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        
        video.release()
        cv2.destroyAllWindows()

thing = ai(0)
thing.main_loop()
