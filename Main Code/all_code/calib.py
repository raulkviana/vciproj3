import numpy as np
from numpy import load
import cv2 as cv
from Constants import constants_calib
from Constants import constants_app


class Calibration:
    def __init__(self):
        """
        Parameters:
        * mtx:  Intrinsic matrix
        * dist: Distortion coeficients
        * rvecs and tvecs: Extrinsic parameters
        * imgpoints: Coordinates of the corners of chessboard in images
        * objpoints: Coordinates in the real world of the square corners of the chessboard
        """

        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.imgpoints = None
        self.objpoints = None
        self.re_proj = None


    def compute_calib_params(self, opencvCam = None, piCam = None, num_of_pics = constants_calib.NUMBER_OF_PICS,
                             period = constants_calib.SECOND_PER_FRAME, columns = constants_calib.NUMBER_OF_COLUMNS,
                             rows = constants_calib.NUMBER_OF_ROWS):
        """
        @brief Compute the parameters of the camera using a chessboard
        @param [in] opencvCam : Video source
        @param [in] num_of_pics : Number of pictures to use for the calibration
        @param [in] columns: Number of columns corresponding to the chessboard scheme
        @param [in] rows:  Number of rows corresponding to the chessboard scheme
        @param [in] period: Secondes per frame

        @param [out] camare_params: Parameters for the calibration
        """
        camera_params = None

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, constants_calib.CHESSBOARD_SQUARE_SIZE, 0.001)

        # prepare object points coordinates
        self.objp = np.zeros((columns * rows, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

        if opencvCam != None:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, constants_calib.CHESSBOARD_SQUARE_SIZE, 0.001)

            # Start camera
            cam = opencvCam
            counter = 0 # Counter of the number of pictures for the calibration

            while (True):

                # Get frame from camara
                ret, frame = cam.read()

                # Transform image to grayscale
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

                # If found, add object points, image points (after refining them)
                if ret == True:

                    # Adding points found to the vectors
                    self.objpoints.append(self.objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    self.imgpoints.append(corners)

                    # Draw and display the corners
                    cv.drawChessboardCorners(frame, (rows, columns), corners2, ret)

                    # Update
                    counter = counter +1

                cv.waitKey(period)  # Delay
                cv.imshow("Calibration", frame)  # Show image

                if counter == num_of_pics:
                    break

            # Calibration parameters
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints,
                                                                                  gray.shape[::-1], None, None)

            # Present results
            print("Distortion coeficients: ", str(self.dist))
            print("Rotation vector: ", str(self.rvecs))
            print("Translation vector: ", str(self.tvecs))
            print("Intrinsic parameters: ", str(self.mtx))

            cv.destroyAllWindows()

            camera_params = [self.mtx, self.dist, self.rvecs, self.tvecs]

        elif piCam != None:

            # Start camera
            counter = 0  # Counter of the number of pictures for the calibration

            while (True):

                # Get frame from camara
                frame = np.empty((constants_app.PICTURES_DIMENSION[0] * constants_app.PICTURES_DIMENSION[1] * 3,),
                                 dtype=np.uint8)
                camera.capture(frame, 'bgr')
                frame = frame.reshape((constants_app.PICTURES_DIMENSION[1], constants_app.PICTURES_DIMENSION[0], 3))

                # Transform image to grayscale
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    # Adding points found to the vectors
                    self.objpoints.append(self.objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    self.imgpoints.append(corners)

                    # Draw and display the corners
                    cv.drawChessboardCorners(frame, (rows, columns), corners2, ret)

                    # Update
                    counter = counter + 1

                cv.waitKey(period)  # Delay
                cv.imshow("Calibration", frame)  # Show image

                if counter == num_of_pics:
                    break

            # Calibration parameters
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints,
                                                                                  gray.shape[::-1], None, None)

            # Present results
            print("Distortion coeficients: ", str(self.dist))
            print("Rotation vector: ", str(self.rvecs))
            print("Translation vector: ", str(self.tvecs))
            print("Intrinsic parameters: ", str(self.mtx))

            cv.destroyAllWindows()

            camera_params = [self.mtx, self.dist, self.rvecs, self.tvecs]

        else:

            # Start camera
            cam = cv.VideoCapture(0)
            counter = 0  # Counter of the number of pictures for the calibration

            while (True):

                # Get frame from camara
                ret, frame = cam.read()

                # Transform image to grayscale
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    # Adding points found to the vectors
                    self.objpoints.append(self.objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    self.imgpoints.append(corners)

                    # Draw and display the corners
                    cv.drawChessboardCorners(frame, (rows, columns), corners2, ret)

                    # Update
                    counter = counter + 1

                cv.waitKey(period)  # Delay
                cv.imshow("Calibration", frame)  # Show image

                if counter == num_of_pics:
                    break

            # Calibration parameters
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints,
                                                                                  gray.shape[::-1], None, None)

            # Present results
            print("Distortion coeficients: ", str(self.dist))
            print("Rotation vector: ", str(self.rvecs))
            print("Translation vector: ", str(self.tvecs))
            print("Intrinsic parameters: ", str(self.mtx))

            cv.destroyAllWindows()

            camera_params = [self.mtx, self.dist, self.rvecs, self.tvecs]


        return camera_params

    def read_params_file(self, in_path_cam_param = constants_calib.OUTPUT_FILE_NAME1,
                         in_path_points  = constants_calib.OUTPUT_FILE_NAME2, points_and_params = False):
        '''
        @brief Ler os parametros num ficheiro para esta classe

        :param in_path_cam_param: nome do ficheiro com os parametros da câmara
        :param in_path_points: nome do ficheiro com os pontos no mundo real do tabuleiro de xadrez

        '''
        if points_and_params:
            # Get the camera parameters
            data1 = load(in_path_cam_param)
            # Get the points
            data2 = load(in_path_points)

            # Load params to class
            self.mtx = data1["intrinsics"]
            self.dist = data1["distortion"]
            self.rvecs = data1["rotation"]
            self.tvecs = data1["transtion"]

            # Load points
            self.imgpoints = data2["imgpoints"]
            self.objpoints = data2["objpoints"]

        else:
            # Get the camera parameters
            data1 = load(in_path_cam_param)

            # Load params to class
            self.mtx = data1["intrinsics"]
            self.dist = data1["distortion"]
            self.rvecs = data1["rotation"]
            self.tvecs = data1["translation"]

    def re_projection_error(self):
        '''
        @brief Calcular o erro associado aos coeficientes de distorção

        '''
        # Re-projection error
        print("Calculate Re-projection error")
        mean_error = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx,
                                             dist)  # Project points obtained before with the parameters calculated
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2) / len(
                imgpoints2)  # Calculate the norm between the point obtained in the line above and the
            mean_error = mean_error + error  # Add to the variable mean_error, to calculate the mean afterwards

        print("Mean Re-projection error: {}".format(mean_error / len(objpoints)))


        self.re_proj = mean_error / len(self.objpoints)

        return self.re_proj

    def write_param_out(self, out_path_cam_param = constants_calib.OUTPUT_FILE_NAME1, out_path_points= constants_calib.OUTPUT_FILE_NAME2):
        '''
        @brief Guardar os parametros presentes na classe num ficheiro

        :param out_path_cam_param: nome do ficheiro com os parametros da câmara
        :param out_path_points: nome do ficheiro com os pontos no mundo real do tabuleiro de xadrez

        '''
        print("\n\nSaving parameters")
        from tempfile import TemporaryFile

        # Write parameters
        outfile = open(out_path_cam_param, 'w')
        np.savez(out_path_cam_param, intrinsics=self.mtx, distortion=self.dist, rotation=self.rvecs, translation=self.tvecs)

        # Write points
        outfile2 = open(out_path_points, 'w')
        np.savez(out_path_points, imgpoints = self.imgpoints, objpoints = self.objpoints)


    def undistort(self, img):
        '''
        @brief Undistort da imagem

        :param img: Imagem que se pretende fazer o undistort

        '''

        # Get the camera parameters
        h, w = img.shape[:2]  # Get height and width
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1,
                                                         (w, h))  # Calculate new camera matrix, optimized

        # Undistort
        dst = cv.undistort(img, self.mtx, self.dist, None, newcameramtx)

        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        return dst
