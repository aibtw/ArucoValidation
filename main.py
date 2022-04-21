import time
import imutils
import math
import numpy as np
import cv2
from cv2 import aruco
import csv


def aruco_init():
    # Get coefficients and camera matrix from yaml calibration file
    cv_file = cv2.FileStorage("calibration_chessboard.yaml", cv2.FileStorage_READ)
    camera_matrix = cv_file.getNode('K').mat()
    distortion_coeffs = cv_file.getNode('D').mat()
    cv_file.release()
    # define the type of aruco marker to detect
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()
    return camera_matrix, distortion_coeffs, arucoDict, arucoParams


def main():
    print("[INFO] Setting up Aruco dictionary and camera coefficients ...")
    camera_matrix, distortion_coeffs, arucoDict, arucoParams = aruco_init()

    time.sleep(2.0)  # Necessary !!!

    print("[INFO] Initializing TCP connection ...")
    # Main loop
    val_list = []
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect ArUco markers in the input frame
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray,
                                                           arucoDict,
                                                           parameters=arucoParams,
                                                           cameraMatrix=camera_matrix,
                                                           distCoeff=distortion_coeffs)
        tx, ty, tz, norm_x = np.Inf, np.Inf, np.Inf, np.Inf  # initializing position values
        rx, ry, rz = np.Inf, np.Inf, np.Inf  # initializing rotation values
        # draw borders
        if len(corners) > 0:  # add remote control boolean
            # Get the rotation and translation vectors
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                0.158,  # Change this to your aruco edge size
                camera_matrix,
                distortion_coeffs)

            # Draw the detected marker and its axis
            aruco.drawDetectedMarkers(frame, corners, ids)
            aruco.drawAxis(frame, camera_matrix, distortion_coeffs, rvecs, tvecs, 0.01)

            # Extract information of the position of the detected marker for interpretation
            tvecs = np.squeeze(tvecs)
            tx, ty, tz = tvecs[0] * 100, tvecs[1] * 100, tvecs[2] * 100

            rvecs = np.squeeze(rvecs)
            rx, ry, rz = obtain_angles(rvecs)

        # Resize the frame (This is safe, because we already did the processing)
        frame = imutils.resize(frame, 800)

        # Enable for debugging
        frame = cv2.flip(frame, 1)

        # Validation
        val_points = [[(0, 0), (800, 450)],
                      [(800, 0), (0, 450)],
                      [(0, 225), (800, 225)],
                      [(400, 0), (400, 450)]]
        # Green color in BGR
        color = (255, 0, 0)
        # Line thickness of 9 px
        thickness = 2
        for pt in val_points:
            frame = cv2.line(frame, pt[0], pt[1], color, thickness)

        # Put text on the frame to display it
        cv2.putText(frame, "Position: x:%.2f, y:%.2f, z:%.2f" % (tx, ty, tz),
                    (0, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(0, 150, 255), thickness=2, lineType=cv2.LINE_AA)
        if rz < 0: rz += 360
        cv2.putText(frame, "Rotation: x:%.2f, y:%.2f, z:%.2f" % (rx, ry, rz),
                    (0, 200), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=(0, 150, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # press r to record angles
        if key == ord("r"):
            val_list.append([rz])

        # press r to record distance
        if key == ord("t"):
            val_list.append([tz])

    f = open('output', 'w')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerows(val_list)
    f.close()


def obtain_angles(rvec):
    # https://github.com/tizianofiorenzani/how_do_drones_work/blob/master/opencv/aruco_pose_estimation.py
    # -- Obtain the rotation matrix tag->camera
    R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
    R_tc = R_ct.T
    R_flip = np.zeros((3, 3), dtype=np.float32)
    R_flip[0, 0] = 1.0
    R_flip[1, 1] = -1.0
    R_flip[2, 2] = -1.0

    # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
    roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)

    # -- Print the marker's attitude respect to camera frame
    str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f" % (math.degrees(roll_marker), math.degrees(pitch_marker),
                                                                  math.degrees(yaw_marker))
    return math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker)


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles100big
# The result is the same as MATLAB except the order
# of the euler angles100big ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


if __name__ == '__main__':
    main()
