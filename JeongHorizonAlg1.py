"""
Important note:
---------------
The case where no horizon is detected can be tricky. The following processes take place:
1) set the flag self.detected_hl_flag to logical True. This flag is used to decide whether next processes will be
performed.
2) assign np.nan to horizon parameters and detection latency. np.nan can be stored in numpy arrays of any np.dtype:
    self.det_position_hl, self.det_tilt_hl, self.theta, self.theta_deg, self.rho, self.latency
3) do not draw the horizon on the color frame/image to draw_and_save. Instead, put a text saying: "NO HORIZON IS
DETECTED"
"""
import cv2 as cv
import numpy as np
from numpy import nan
import os
from scipy.spatial import distance
from math import pi, atan
from time import time


class JeongC1:
    def __init__(self):
        """A class implementing the horizon detection algorithm published in DOI: 10.1177/1550147718790753 by Chi Yoon
        Jeong et al
        """
        self.img_rgb = None
        self.roi_rgb = None
        self.roi_gray = None  # the extracted roi converted to grayscale; equivalent to I (see equation 2)

        # self.kernel_size_s is equivalent to fs (see equation 2)
        self.kernel_size_1 = (10 * 1) + 1  # size of the median kernel of the first scale (10 * s) + 1; s = 1
        self.kernel_size_2 = (10 * 2) + 1  # size of the median kernel of the first scale (10 * s) + 1; s = 2
        self.kernel_size_3 = (10 * 3) + 1  # size of the median kernel of the first scale (10 * s) + 1; s = 3

        self.canny_th_low = 30
        self.canny_th_high = 150

        self.D_rho = 1
        self.D_theta = pi/180
        self.D_rho_j = nan
        self.D_rho_g = nan
        self.rho = nan
        self.theta = nan
        self.x_cte = nan
        self.y_cte = nan
        self.rho_j = nan
        self.x_j = None
        self.y_j = None

        # coordinates
        self.min_y = None  # the starting row of roi extraction
        self.max_y = None  # the ending row of roi extraction
        self.edge_map = None
        self.inlier_edges_x = None  # x coordinates of inlier edges (works for both original sized image and roi image)
        self.inlier_edges_y = None  # y cooridnates of inlier edges in the coordinates of roi image
        self.inlier_edges_y_org = None  # y cooridnates inlier edges in the coordinates of original image. It is equal
        # to self.inlier_edges_y + self.min_y
        self.inlier_edges_xy = None  # a 2d array of two columns: 1st = self.inlier_edges_x, 2nd = self.inlier_edges_y

        # outputs
        self.hl_slope = nan
        self.hl_intercept = nan

        self.xs_hl = nan
        self.xe_hl = nan
        self.ys_hl = nan
        self.ye_hl = nan

        self.det_position_hl = nan
        self.det_tilt_hl = nan
        self.latency = nan

        # flags
        self.detected_hl_flag = True  # True indicates that a line is detected

    def get_horizon(self, img):
        """
        Find the position Y and tilt alpha (in degrees) of the horizon line
        :param img: the input image to process, which must be an RGB (or BGR).
        :return:
        """
        if len(img.shape) != 3:
            raise Exception("The input image img must be an RGB (or BGR) image")
        self.start_time = time()
        self.img_rgb = img
        self.org_height, self.org_width = self.img_rgb.shape[:2]
        self.roi_detection()  # extracts the roi and stores it in self.roi_rgb
        self.multiscale_processing()
        self.edge_map_fusion()
        self.hough_transform()
        self.linear_least_square_fitting()

    def roi_detection(self):
        """
        Extracts the region-of-interest from self.img_rgb and stores it in self.roi_rgb
        """
        # image resize
        resize_ratio = 0.25
        resize_color = cv.resize(self.img_rgb, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv.INTER_AREA)
        rows, cols, chan = np.shape(resize_color)

        # find ROI region
        height_length = 65
        height_step = 30

        prev_mean = []
        max_idx = 0
        max_ed = 0

        for height_idx in range(0, 8):

            end_y = int(height_idx * height_step) + int(height_length) - 1

            if (height_idx * height_step + height_length) > rows:
                end_y = rows - 1

            self.roi_rgb = resize_color[int(height_idx * height_step):end_y, :]

            cur_mean = np.mean(self.roi_rgb, axis=(0, 1))

            if height_idx > 0:
                ed = distance.euclidean(cur_mean, prev_mean)
                if ed > max_ed:
                    max_ed = ed
                    max_idx = height_idx

            prev_mean = cur_mean

        self.min_y = max_idx * height_step * 4
        self.max_y = (max_idx * height_step + height_length) * 4

        if self.min_y < 0:
            self.min_y = 0

        # ROI img extraction
        self.roi_rgb = self.img_rgb[int(self.min_y):int(self.max_y), :]

    def multiscale_processing(self):
        self.roi_gray = cv.cvtColor(self.roi_rgb, cv.COLOR_BGR2GRAY)

        # apply three median scales (see equation 2); self.median_scale_s is equivalent to Is (equation 2)
        self.median_scale_1 = cv.medianBlur(self.roi_gray, self.kernel_size_1)
        self.median_scale_2 = cv.medianBlur(self.roi_gray, self.kernel_size_2)
        self.median_scale_3 = cv.medianBlur(self.roi_gray, self.kernel_size_2)

        # conversion from np.uint8 to np.float32: this avoids bit overflow
        self.canny_edges_scale_1 = np.float32(cv.Canny(self.median_scale_1, self.canny_th_low, self.canny_th_high))
        self.canny_edges_scale_2 = np.float32(cv.Canny(self.median_scale_2, self.canny_th_low, self.canny_th_high))
        self.canny_edges_scale_3 = np.float32(cv.Canny(self.median_scale_3, self.canny_th_low, self.canny_th_high))

    def edge_map_fusion(self):
        self.weighted_canny_edges = np.add(np.add(self.canny_edges_scale_1, self.canny_edges_scale_2),
                                           self.canny_edges_scale_3)  # equivalent to W(x,y) (see equation 3); np.uint8
        self.weighted_canny_edges = np.uint8(np.divide(self.weighted_canny_edges, 3))  # normalize to [0, 255] np.uint8
        _, self.edge_map = cv.threshold(src=self.weighted_canny_edges, thresh=153, maxval=255, type=cv.THRESH_BINARY)

    def hough_transform(self):
        self.hough_lines = cv.HoughLines(image=self.edge_map, threshold=2, rho=self.D_rho, theta=self.D_theta)
        if self.hough_lines is not None:  # executes if Hough detects a line
            self.detected_hl_flag = True
            self.rho, self.theta = self.hough_lines[0][0]  # self.theta in radians
        else:
            self.detected_hl_flag = False
            self.rho = nan
            self.theta = nan
            self.det_position_hl = nan
            self.det_tilt_hl = nan
            self.latency = nan

    def linear_least_square_fitting(self):
        if self.detected_hl_flag:
            self.get_inlier_edges()
            self.inlier_edges_xy = np.zeros((self.inlier_edges_x.size, 2), dtype=np.int32)
            self.inlier_edges_xy[:, 0], self.inlier_edges_xy[:, 1] = self.inlier_edges_x, self.inlier_edges_y_org
            [vx, vy, x, y] = cv.fitLine(points=self.inlier_edges_xy, distType=cv.DIST_L2,
                                        param=0, reps=self.D_rho, aeps=self.D_theta)
            self.hl_slope = float(vy / vx)  # float to convert from (1,) float numpy array to python float
            self.hl_intercept = float(y - self.hl_slope * x)

            self.xs_hl = int(0)
            self.xe_hl = int(self.org_width - 1)
            self.ys_hl = int(self.hl_intercept)  # = int((self.hl_slope * self.xs_hl) + self.hl_intercept)
            self.ye_hl = int((self.xe_hl * self.hl_slope) + self.hl_intercept)

            self.det_tilt_hl = (-atan(self.hl_slope))*(180/pi)  # - because the y axis of images goes down
            self.det_position_hl = ((self.org_width - 1)/2) * self.hl_slope + self.hl_intercept

            self.end_time = time()
            self.latency = round((self.end_time - self.start_time), 4)

    def get_inlier_edges(self):
        """
        Process is described in inlier_edges.pdf file attached with this code project.
        """
        self.y_j, self.x_j = np.where(self.edge_map == 255)

        theta_p = self.theta + self.D_theta
        theta_n = self.theta - self.D_theta
        self.x_cte = 0.5 * (np.cos(theta_p) - np.cos(theta_n))
        self.y_cte = 0.5 * (np.sin(theta_p) - np.sin(theta_n))

        self.D_rho_j = np.abs(np.add(np.multiply(self.x_j, self.x_cte), np.multiply(self.y_j, self.y_cte)))
        self.D_rho_g = np.add(self.D_rho_j, self.D_rho)

        self.rho_j = np.add(np.multiply(self.x_j, np.cos(self.theta)), np.multiply(self.y_j, np.sin(self.theta)))
        inlier_condition = np.logical_and(self.rho_j <= (self.rho + self.D_rho_g/2),
                                          self.rho_j >= (self.rho - self.D_rho_g/2))

        self.inlier_edges_indexes = np.where(inlier_condition)
        self.inlier_edges_x = self.x_j[self.inlier_edges_indexes]
        self.inlier_edges_y = self.y_j[self.inlier_edges_indexes]
        self.inlier_edges_y_org = np.add(self.inlier_edges_y, self.min_y)
        self.inlier_edges_map = np.zeros(shape=self.edge_map.shape, dtype=np.uint8)
        self.inlier_edges_map[self.inlier_edges_y, self.inlier_edges_x] = 255

    def draw_hl(self):
        """
        Draws the horizon line on attribute 'self.img_with_hl' if it is detected. Otherwise, the text 'NO HORIZON IS
        DETECTED' is put on the image.
        """
        self.img_with_hl = np.copy(self.img_rgb)
        if self.detected_hl_flag:
            cv.line(self.img_with_hl, (self.xs_hl, self.ys_hl), (self.xe_hl, self.ye_hl), (0, 0, 255), 5)
        else:
            put_text = "NO HORIZON IS DETECTED"
            org = (int(self.org_height / 2), int(self.org_height / 2))
            color = (0, 0, 255)
            cv.putText(img=self.img_with_hl, text=put_text, org=org, fontFace=0, fontScale=2, color=color, thickness=3)

    def evaluate(self, src_video_folder, src_gt_folder, dst_video_folder=r"", dst_quantitative_results_folder=r"",
                 draw_and_save=True):
        """
        Produces a .npy file containing quantitative results of the Horizon Edge Filter algorithm. The .npy file
        contains the following information for each image: |Y_gt - Y_det|, |alpha_gt - alpha_det|, and latency in
        seconds between 0 and 1) specifying the ratio of the diameter of the resized image being processed. For
        instance, if the attributre self.dsize = (640, 480), the threshold that will be used in the hough transform
        is sqrt(640^2 + 480^2) * hough_threshold_ratio, rounded to the nearest integer. :param src_gt_folder:
        absolute path to the ground truth horizons corresponding to source video files. :param src_video_folder:
        absolute path to folder containing source video files to process :param dst_video_folder: absolute path where
        video files with drawn horizon will be saved. :param dst_quantitative_results_folder: destination folder
        where quantitative results will be saved. :param draw_and_save: if True, all detected horizons will be drawn
        on their corresponding frames and saved as video files in the folder specified by 'dst_video_folder'.
        """
        src_video_names = sorted(os.listdir(src_video_folder))
        srt_gt_names = sorted(os.listdir(src_gt_folder))
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):
            print("{} will correspond to {}".format(src_video_name, src_gt_name))

        # Allowing the user to verify that each gt .npy file corresponds to the correct video file # # # # # # # # # # #
        while True:
            yn = input("Above are the video files and their corresponding gt files. If they are correct, click on 'y'"
                       " to proceed, otherwise, click on 'n'.\n"
                       "If one or more video file has incorrect gt file correspondence, we recommend to rename the"
                       "files with similar names.")
            if yn == 'y':
                break
            elif yn == 'n':
                print("\nTHE QUANTITATIVE EVALUATION IS ABORTED AS ONE OR MORE LOADED GT FILES DOES NOT CORRESPOND TO "
                      "THE CORRECT VIDEO FILE")
                return
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.det_horizons_all_files = np.empty(shape=[0, 5])
        nbr_of_vids = len(src_video_names)
        vid_indx = 0
        for src_video_name, src_gt_name in zip(src_video_names, srt_gt_names):  # each iteration processes one video
            # file
            vid_indx += 1
            print("loaded video/loaded gt: {}/{}".format(src_video_name, src_gt_name))  # printing which video file
            # correspond to which gt file

            src_video_path = os.path.join(src_video_folder, src_video_name)
            src_gt_path = os.path.join(src_gt_folder, src_gt_name)

            cap = cv.VideoCapture(src_video_path)  # create a video reader object
            # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            fps = cap.get(propId=cv.CAP_PROP_FPS)
            self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
            self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
            if draw_and_save:
                dst_vid_path = os.path.join(dst_video_folder, "C.Jeo_" + src_video_name)
                video_writer = cv.VideoWriter(dst_vid_path, fourcc, fps, (self.org_width, self.org_height),
                                              True)  # video writer object
            self.gt_horizons = np.load(src_gt_path)
            #
            nbr_of_annotations = self.gt_horizons.shape[0]
            nbr_of_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
            if nbr_of_frames != nbr_of_annotations:
                warning_text_1 = "The number of annotations (={}) does not equal to the number of frames (={})". \
                    format(nbr_of_annotations, nbr_of_frames)
                print("----------WARNING---------")
                print(warning_text_1)
                print("--------------------------")

            self.det_horizons_per_file = np.zeros((nbr_of_annotations, 5))
            for idx, gt_horizon in enumerate(self.gt_horizons):
                no_error_flag, frame = cap.read()
                if not no_error_flag:
                    break
                self.get_horizon(img=frame)  # gets the horizon position and
                # tilt
                self.gt_position_hl, self.gt_tilt_hl = gt_horizon[0], gt_horizon[1]
                # print("detected position/gt position {}/{};\n detected tilt/gt tilt {}/{}". format(round(
                # self.det_position_hl, 2), round(self.gt_position_hl, 2), round(self.det_tilt_hl, 2),
                # round(self.gt_tilt_hl, 2))) print("with latency = {} seconds".format(round(self.latency, 4)))
                print("Frame {}/{}. Video {}/{}".format(idx, nbr_of_frames, vid_indx, nbr_of_vids))
                self.det_horizons_per_file[idx] = [self.det_position_hl,
                                                   self.det_tilt_hl,
                                                   round(abs(self.det_position_hl - self.gt_position_hl), 4),
                                                   round(abs(self.det_tilt_hl - self.gt_tilt_hl), 4),
                                                   self.latency]
                self.draw_hl()  # draws the horizon on self.img_with_hl
                video_writer.write(self.img_with_hl)
            cap.release()
            video_writer.release()
            print("The video file {} has been processed.".format(src_video_name))

            # saving the .npy file of quantitative results of current video file # # # # # # # # # # # # # # # # # # # #
            src_video_name_no_ext = os.path.splitext(src_video_name)[0]
            det_horizons_per_file_dst_path = os.path.join(dst_quantitative_results_folder,
                                                          src_video_name_no_ext + ".npy")
            np.save(det_horizons_per_file_dst_path, self.det_horizons_per_file)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.det_horizons_all_files = np.append(self.det_horizons_all_files,
                                                    self.det_horizons_per_file,
                                                    axis=0)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # after processing all video files, save quantitative results as .npy file
        src_video_folder_name = os.path.basename(src_video_folder)
        dst_detected_path = os.path.join(dst_quantitative_results_folder,
                                         "all_det_hl_" + src_video_folder_name + ".npy")
        np.save(dst_detected_path, self.det_horizons_all_files)

    def evaluate_fine_tuning(self, src_vid_path, src_gt_path=r"", dst_path=r"", out_option=""):
        """
        A temporary method evaluating changes corresponding to fine-tuning of this algorithm.
        """
        cap = cv.VideoCapture(src_vid_path)  # create a video reader object
        # Creating the video writer # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        fps = cap.get(propId=cv.CAP_PROP_FPS)
        self.total_frames = int(cap.get(propId=cv.CAP_PROP_FRAME_COUNT))
        self.org_width = int(cap.get(propId=cv.CAP_PROP_FRAME_WIDTH))
        self.org_height = int(cap.get(propId=cv.CAP_PROP_FRAME_HEIGHT))
        # create video writer
        fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # codec used to compress the video.
        video_basename = os.path.basename(src_vid_path).split(sep=".")[0]
        print(video_basename)
        dst_vid_path_2 = os.path.join(dst_path, video_basename + "_Detected_horizon.avi")

        self.down_size_factor = 1
        self.res_width = int(self.org_width * self.down_size_factor)
        self.res_height = int(self.org_height * self.down_size_factor)
        video_writer_2 = cv.VideoWriter(dst_vid_path_2, fourcc, fps, (self.res_width, self.res_height), True)
        no_error_flag, frame = cap.read()
        if not no_error_flag:
            return
        self.frame_index = 0
        while no_error_flag:
            # self.input_img = np.copy(frame)
            self.input_img = cv.resize(np.copy(frame), dsize=(self.res_width, self.res_height))
            self.get_horizon(img=self.input_img)

            self.draw_hl()
            put_text = "frame {}/{}".format(self.frame_index, self.total_frames)
            cv.putText(img=self.img_with_hl,
                       text=put_text,
                       org=(0, 30),
                       fontFace=0,
                       fontScale=1,
                       color=(0, 0, 255),
                       thickness=2)
            video_writer_2.write(self.img_with_hl)
            print(put_text)
            no_error_flag, frame = cap.read()
            self.frame_index += 1
        cap.release()
        # video_writer_1.release()
