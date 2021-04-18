import tkinter as tk
from tkinter import *
import tkinter.font as font


from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time


from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist


class Frontend:
	def __init__(self):
		self.window = tk.Tk()
		self.window.title("Face Mask Detection and Social Distancing Detector")
		self.window.resizable(0, 0)
		window_height = 600
		window_width = 880
		screen_width = self.window.winfo_screenwidth()
		screen_height = self.window.winfo_screenheight()
		x_cordinate = int((screen_width / 2) - (window_width / 2))
		y_cordinate = int((screen_height / 2) - (window_height / 2))
		self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
		self.window.configure(background='#ffffff')
		self.window.grid_rowconfigure(0, weight=1)
		self.window.grid_columnconfigure(0, weight=1)


	



		header = tk.Label(self.window, text="Face mask and Social Distancing Detector", width=80, height=2, fg="white", bg="#363e75",
		                          font=('times', 18, 'bold', 'underline'))
		header.place(x=0, y=0)


		takeImg = tk.Button(self.window, text="Face Mask Detector",command=self.mask_detect, fg="white", bg="#363e75", width=20,
		                            height=2,
		                            activebackground="#118ce1", font=('times', 15, ' bold '))
		takeImg.place(x=320, y=100)

		trainImg = tk.Button(self.window, text="Social Distancing Detector", command=self.SocialDist, fg="white", bg="#363e75", width=25,
		                             height=2,
		                             activebackground="#118ce1", font=('times', 15, ' bold '))
		trainImg.place(x=300, y=250)

		trackimg =tk.Button(self.window, text="People Tracker", command=self.peopletracker, fg="white", bg="#363e75", width=25,height=2,
					activebackground="#118ce1", font=('times',15,'bold'))

		trackimg.place(x=300,y=400)

		quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75", width=10, height=2,
		                               activebackground="#118ce1", font=('times', 15, 'bold'))
		quitWindow.place(x=650, y=510)
		self.window.mainloop()


	def close_window(self):
		self.window.destroy()

	


	def mask_detect(self):


		def detect_and_predict_mask(frame, faceNet, maskNet):
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
					(104.0, 177.0, 123.0))
			faceNet.setInput(blob)
			detections = faceNet.forward()

			faces = []
			locs = []
			preds = []


			for i in range(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > args["confidence"]:
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					(startX, startY) = (max(0, startX), max(0, startY))
					(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

					face = frame[startY:endY, startX:endX]
					face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
					face = cv2.resize(face, (224, 224))
					face = img_to_array(face)
					face = preprocess_input(face)
					face = np.expand_dims(face, axis=0)


					faces.append(face)
					locs.append((startX, startY, endX, endY))

			if len(faces) > 0:
				preds = maskNet.predict(faces)

			return (locs, preds)

		ap = argparse.ArgumentParser()
		ap.add_argument("-f", "--face", type=str,
			default="face_detector",
			help="path to face detector model directory")
		ap.add_argument("-m", "--model", type=str,
			default="mask_detector.model",
			help="path to trained face mask detector model")
		ap.add_argument("-c", "--confidence", type=float, default=0.5,
			help="minimum probability to filter weak detections")
		args = vars(ap.parse_args())

		print("[INFO] loading face detector model...")
		prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
		weightsPath = os.path.sep.join([args["face"],
			"res10_300x300_ssd_iter_140000.caffemodel"])
		faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

		# load the face mask detector model from disk
		print("[INFO] loading face mask detector model...")
		maskNet = load_model(args["model"])

		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)

		while True:
			frame = vs.read()
			frame = imutils.resize(frame, width=400)
			(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
			for (box, pred) in zip(locs, preds):
				(startX, startY, endX, endY) = box

				(mask, withoutMask) = pred

				label = "Mask" if mask > withoutMask else "No Mask"
				color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
				cv2.putText(frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
		cv2.destroyAllWindows()
		vs.stop()

	def SocialDist(self):
		ap = argparse.ArgumentParser()
		ap.add_argument("-i", "--input", type=str, default="test.mp4",
				help="path to (optional) input video file")
		ap.add_argument("-o", "--output", type=str, default="",
			help="path to (optional) output video file")
		ap.add_argument("-d", "--display", type=int, default=1,
			help="whether or not output frame should be displayed")
		args = vars(ap.parse_args())
		labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
		LABELS = open(labelsPath).read().strip().split("\n")

		weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
		configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

		print("[INFO] loading YOLO from disk...")
		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
		if config.USE_GPU:

			print("[INFO] setting preferable backend and target to CUDA...")
			net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		print("[INFO] accessing video stream...")
		vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
		writer = None

		while True:
			(grabbed, frame) = vs.read()
			if not grabbed:
				break
			frame = imutils.resize(frame, width=700)
			results = detect_people(frame, net, ln,
				personIdx=LABELS.index("person"))	
			violate = set()

			if len(results) >= 2:
				centroids = np.array([r[2] for r in results])
				D = dist.cdist(centroids, centroids, metric="euclidean")

				for i in range(0, D.shape[0]):
					for j in range(i + 1, D.shape[1]):
						if D[i, j] < config.MIN_DISTANCE:
							violate.add(i)
							violate.add(j)

			for (i, (prob, bbox, centroid)) in enumerate(results):
				(startX, startY, endX, endY) = bbox
				(cX, cY) = centroid
				color = (0, 255, 0)


				if i in violate:
					color = (0, 0, 255)

				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				cv2.circle(frame, (cX, cY), 5, color, 1)

			text = "Social Distancing Violations: {}".format(len(violate))
			cv2.putText(frame, text, (10, frame.shape[0] - 25),
				cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)	

			if args["display"] > 0:
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF

				if key == ord("q"):
					break

			if args["output"] != "" and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 25,
						(frame.shape[1], frame.shape[0]), True)	

			if writer is not None:
				writer.write(frame)

	def peopletracker(self):
		ap = argparse.ArgumentParser()
		ap.add_argument("-i", "--input", type=str, default="test1.mp4",
			help="path to (optional) input video file")
		ap.add_argument("-o", "--output", type=str, default="",
			help="path to (optional) output video file")
		ap.add_argument("-d", "--display", type=int, default=1,
			help="whether or not output frame should be displayed")
		args = vars(ap.parse_args())

		ct = CentroidTracker()
		(H, W) = (None, None)

		labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
		LABELS = open(labelsPath).read().strip().split("\n")

		weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
		configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

		print("[INFO] loading YOLO from disk...")
		net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

		if config.USE_GPU:
			# set CUDA as the preferable backend and target
			print("[INFO] setting preferable backend and target to CUDA...")
			net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
			net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		print("[INFO] accessing video stream...")
		vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
		writer = None

		count=0

		while True:
			# read the next frame from the file
			(grabbed, frame) = vs.read()

			# if the frame was not grabbed, then we have reached the end
			# of the stream
			if not grabbed:
				break

			
			frame = imutils.resize(frame, width=700)
			results = detect_people(frame, net, ln,
				personIdx=LABELS.index("person"))

			
			
			rects = []

			for (i, (prob, bbox, centroid)) in enumerate(results):
				# extract the bounding box and centroid coordinates, then
				# initialize the color of the annotation
				(startX, startY, endX, endY) = bbox
				#(cX, cY) = centroid
				color = (0, 255, 0)
				rects.append(bbox)
				
				# draw (1) a bounding box around the person and (2) the
				# centroid coordinates of the person,
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			objects = ct.update(rects)

			for (objectID, centroid) in objects.items():
				# draw both the ID of the object and the centroid of the
				# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

			if args["display"] > 0:
				# show the output frame
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF

				# if the `q` key was pressed, break from the loop
				if key == ord("q"):
					break

			if args["output"] != "" and writer is None:
				# initialize our video writer
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 25,
					(frame.shape[1], frame.shape[0]), True)

			if writer is not None:
				writer.write(frame)







Frontend()