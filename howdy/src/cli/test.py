# Show a window with the video stream and testing information

# Import required modules
import configparser
import builtins
import os
import json
import sys
import time
import cv2
import numpy as np
import paths_factory
import recognition

from i18n import _
from recorders.video_capture import VideoCapture

# Read config from disk
config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

if config.get("video", "recording_plugin", fallback="opencv") != "opencv":
	print(_("Howdy has been configured to use a recorder which doesn't support the test command yet, aborting"))
	sys.exit(12)

video_capture = VideoCapture(config)

# Read config values to use in the main loop
exposure = config.getint("video", "exposure", fallback=-1)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)
# Scale factor for the preview window (frames are small, especially IR cams)
preview_scale = max(1, config.getint("debug", "preview_scale", fallback=2))
# Seconds to linger on each frame when slow mode is enabled
slow_mode_delay = 2.0

# Let the user know what's up
print(_("""
Opening a window with a test feed

Press ctrl+C in this terminal to quit
Click on the image to enable or disable slow mode
"""))


def mouse(event, x, y, flags, param):
	"""Handle mouse events"""
	global slow_mode

	# Toggle slowmode on click
	if event == cv2.EVENT_LBUTTONDOWN:
		slow_mode = not slow_mode


def print_text(line_number, text):
	"""Print the status text by line number"""
	cv2.putText(overlay, text, (10, height - 10 - (14 * line_number)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1, cv2.LINE_AA)


# Build the face analyzer (raises if the model pack is missing)
try:
	analyzer = recognition.create_analyzer(config)
except FileNotFoundError:
	sys.exit(1)

print(_("Using execution provider: %s") % (analyzer.howdy_providers[0], ))

# Try to load the enrolled models of the user to show live match scores
similarity_threshold = config.getfloat("video", "similarity_threshold", fallback=0.45)
encodings = None
encoding_owners = None
models = None

try:
	user = builtins.howdy_user
	models = recognition.load_encodings(paths_factory.user_model_path(user))
	encodings, encoding_owners = recognition.flatten_encodings(models)
	print(_("Showing match scores against %d enrolled model(s) of %s") % (len(models), user))
except FileNotFoundError:
	print(_("No enrolled face model found, showing detection only"))
except recognition.LegacyModelError:
	print(_("Enrolled models are in the old incompatible format, showing detection only"))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Open the window and attach a a mouse listener
cv2.namedWindow("Howdy Test")
cv2.setMouseCallback("Howdy Test", mouse)

# Enable a delay in the loop
slow_mode = False
# Console status of the last processed frame
last_face_count = 0
last_match_text = "no face"
# Match state of the previous frame to only log transitions
last_match_state = None
# Count all frames ever
total_frames = 0
# Count all frames per second
sec_frames = 0
# Last secands FPS
fps = 0
# The current second we're counting
sec = int(time.time())
# recognition time
rec_tm = 0

# Wrap everything in an keyboard interrupt handler
try:
	while True:
		frame_tm = time.time()

		# Increment the frames
		total_frames += 1
		sec_frames += 1

		# Id we've entered a new second
		if sec != int(frame_tm):
			# Set the last seconds FPS
			fps = sec_frames

			# Set the new second and reset the counter
			sec = int(frame_tm)
			sec_frames = 0

			# Print a compact status line to the console once per second
			print("fps %2d  rec %3dms  faces %d  %s" % (fps, round(rec_tm * 1000), last_face_count, last_match_text))

		# Grab a single frame of video
		orig_frame, frame = video_capture.read_frame()

		frame = clahe.apply(frame)
		# Make a frame to put overlays in
		overlay = frame.copy()
		overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

		# Fetch the frame height and width
		height, width = frame.shape[:2]

		# Create a histogram of the image with 8 values.
		# Flatten because OpenCV 5 returns a 1-D array where 4.x returned 2-D.
		hist = cv2.calcHist([frame], [0], None, [8], [0, 256]).flatten()
		# All values combined for percentage calculation (guard against black frames)
		hist_total = int(np.sum(hist)) or 1
		# Fill with the overall containing percentage
		hist_perc = []

		# Loop though all values to calculate a percentage and add it to the overlay
		for index, value in enumerate(hist):
			value_perc = float(value) / hist_total * 100
			hist_perc.append(value_perc)

			# Top left point, 10px margins
			p1 = (20 + (10 * index), 10)
			# Bottom right point makes the bar 10px thick, with an height of half the percentage
			p2 = (10 + (10 * index), int(value_perc / 2 + 10))
			# Draw the bar in green
			cv2.rectangle(overlay, p1, p2, (0, 200, 0), thickness=cv2.FILLED)

		# Print the statis in the bottom left
		print_text(0, _("RESOLUTION: %dx%d") % (height, width))
		print_text(1, _("FPS: %d") % (fps, ))
		print_text(2, _("FRAMES: %d") % (total_frames, ))
		print_text(3, _("RECOGNITION: %dms") % (round(rec_tm * 1000), ))
		print_text(4, _("PROVIDER: %s") % (analyzer.howdy_providers[0].replace("ExecutionProvider", ""), ))

		# Show that slow mode is on, if it's on
		if slow_mode:
			cv2.putText(overlay, _("SLOW MODE"), (width - 100, height - 10), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)

		# Ignore dark frames
		if hist_perc[0] > dark_threshold:
			# Show that this is an ignored frame in the top right
			cv2.putText(overlay, _("DARK FRAME"), (width - 102, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)
		else:
			# Show that this is an active frame
			cv2.putText(overlay, _("SCAN FRAME"), (width - 102, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1, cv2.LINE_AA)

			rec_tm = time.time()

			# Detect and encode all faces in the color frame
			faces = recognition.get_faces(analyzer, orig_frame)
			rec_tm = time.time() - rec_tm

			last_face_count = len(faces)
			if not faces:
				last_match_text = "no face"

			# Loop though all faces and paint a circle around them
			for face in faces:
				# By default the circle around the face is red for no match
				color = (0, 0, 230)

				# Unpack the bounding box
				left, top, right, bottom = face.bbox.astype(int)

				# Get the center X and Y from the rectangular points
				x = int((right - left) / 2) + left
				y = int((bottom - top) / 2) + top

				# Get the raduis from the with of the square
				r = (right - left) / 2
				# Add 20% padding
				r = int(r + (r * 0.2))

				# If we have models defined for the current user
				if encodings is not None:
					# Match this found face against the known encodings by cosine similarity
					match_index, similarity = recognition.best_match(face.normed_embedding, encodings)

					matched = similarity >= similarity_threshold
					# If a model matches
					if matched:
						# Turn the circle green
						color = (0, 230, 0)

						# Print the name of the model next to the circle
						circle_text = "{} ({:.3f} >= {:.2f})".format(
							encoding_owners[match_index]["label"], similarity, similarity_threshold)
						cv2.putText(overlay, circle_text, (int(x + r / 3), y - r), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1, cv2.LINE_AA)
					# If no approved matches, show red text
					else:
						circle_text = "no match ({:.3f} < {:.2f})".format(similarity, similarity_threshold)
						cv2.putText(overlay, circle_text, (int(x + r / 3), y - r), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)

					# Track for the console: log match/no-match transitions immediately
					last_match_text = "sim %.3f det %.2f (%s)" % (similarity, face.det_score, encoding_owners[match_index]["label"])
					if matched != last_match_state:
						print(("MATCH    " if matched else "NO MATCH ") + last_match_text)
						last_match_state = matched
				else:
					last_match_text = "det %.2f (no enrolled models)" % (face.det_score, )

				# Show the detector confidence under the circle
				cv2.putText(overlay, "det {:.2f}".format(face.det_score), (int(x + r / 3), y + r + 12), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)

				# Draw the circle around the face
				cv2.circle(overlay, (x, y), r, color, 2)

		# Add the overlay to the frame with some transparency
		alpha = 0.65
		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

		# Scale the preview up for readability
		if preview_scale != 1:
			frame = cv2.resize(frame, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_LINEAR)

		# Show the image in a window
		cv2.imshow("Howdy Test", frame)

		# Quit on any keypress
		if cv2.waitKey(1) != -1:
			raise KeyboardInterrupt()

		frame_time = time.time() - frame_tm

		# Delay the frame if slowmode is on
		if slow_mode:
			time.sleep(max([slow_mode_delay - frame_time, 0.0]))

		if exposure != -1:
			# For a strange reason on some cameras (e.g. Lenoxo X1E)
			# setting manual exposure works only after a couple frames
			# are captured and even after a delay it does not
			# always work. Setting exposure at every frame is
			# reliable though.
			video_capture.internal.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)  # 1 = Manual
			video_capture.internal.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

# On ctrl+C
except KeyboardInterrupt:
	# Let the user know we're stopping
	print(_("\nClosing window"))

	# Release handle to the webcam
	cv2.destroyAllWindows()
