import cv2 as cv
import threading
import numpy as np
import random
import glm
import multiprocessing

campath = 'data/cam%d/intrinsics.xml'
boardpath = 'data/checkerboard.xml'
videopath = 'data/%s/%s.avi'
samplesize = 30

manual_position = np.zeros((4, 2), np.float32)
axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)


# axis *= self.cBSquareSize


# mouse click event
def click_event(event, x, y, flags, params):
    global click
    global manual_position
    if event == cv.EVENT_LBUTTONDOWN:
        if click < 4:
            manual_position[click] = (x, y)
            # print(manual_position)
            # cv.circle(img, (x, y), 6, (0, 0, 255), -1)
            # cv.imshow('img', img)
            click += 1


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())

    # imgpoints
    imgpts0 = tuple(imgpts[0].ravel())
    imgpts1 = tuple(imgpts[1].ravel())
    imgpts2 = tuple(imgpts[2].ravel())

    # y axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts0[0]), int(imgpts0[1])), (255, 0, 0), 1)
    # x axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts1[0]), int(imgpts1[1])), (0, 255, 0), 1)
    # z axis
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(imgpts2[0]), int(imgpts2[1])), (0, 0, 255), 1)

    return img


class CameraConfig:
    _instance_lock = threading.Lock()

    mtx: dict = {}
    dist: dict = {}
    video: dict = {}
    cBWidth: int
    cBHeight: int
    cBSquareSize: int
    mask: dict = {}
    _rvecs: dict = {}
    _tvecs: dict = {}
    _cameraposition: dict = {}
    _rotation: dict = {}

    def __new__(cls, *args, **kwargs):
        if not hasattr(CameraConfig, "_instance"):
            with CameraConfig._instance_lock:
                if not hasattr(CameraConfig, "_instance"):
                    CameraConfig._instance = object.__new__(cls)
                    cls.load_parameter(cls)
        return CameraConfig._instance

    # read the checkerboard data from checkerboard.xml
    def load_parameter(self):
        fs = cv.FileStorage(boardpath, cv.FILE_STORAGE_READ)
        self.cBWidth = fs.getNode("CheckerBoardWidth").real()
        self.cBHeight = fs.getNode("CheckerBoardHeight").real()
        self.cBSquareSize = fs.getNode("CheckerBoardSquareSize").real()
        fs.release()
        for i in range(1, 5):
            fs = cv.FileStorage(campath % i, cv.FILE_STORAGE_READ)
            mtx = np.mat(fs.getNode("CameraMatrix").mat())
            self.mtx['cam%d' % i] = mtx
            dist = np.mat(fs.getNode("DistortionCoeffs").mat())
            self.dist['cam%d' % i] = dist
            fs.release()
        print('parameters loaded')

    # update the 'cname' file, if no input, update all
    def __update(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.__update('cam%d' % i)
            return
        # print('update %s' % cname)
        fs = cv.FileStorage('data/%s/intrinsics.xml' % cname, cv.FILE_STORAGE_WRITE)
        fs.write("CameraMatrix", np.matrix(self.mtx[cname]))
        fs.write("DistortionCoeffs", np.array(self.dist[cname]))
        fs.release()

    def save_xml(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.save_xml('cam%d' % i)
            return
        fs = cv.FileStorage('data/%s/config.xml' % cname, cv.FILE_STORAGE_WRITE)
        fs.write("CameraMatrix", np.matrix(self.mtx[cname]))
        fs.write("DistortionCoeffs", np.array(self.dist[cname]))
        fs.write("RVector", np.matrix(self._rvecs[cname]))
        fs.write("RMatrix", np.matrix(cv.Rodrigues(self._rvecs[cname])[0]))
        fs.write("TVector", np.matrix(self._tvecs[cname]))
        fs.release()

    def load_xml(self):
        for i in range(1, 5):
            cname = 'cam%d' % i
            fs = cv.FileStorage('data/%s/config.xml' % cname, cv.FILE_STORAGE_READ)
            mtx = np.mat(fs.getNode("CameraMatrix").mat())
            self.mtx[cname] = mtx
            dist = np.mat(fs.getNode("DistortionCoeffs").mat())
            self.dist[cname] = dist
            rvecs = np.mat(fs.getNode("RVector").mat())
            self._rvecs[cname] = rvecs
            tvecs = np.mat(fs.getNode("TVector").mat())
            self._tvecs[cname] = tvecs
            fs.release()

    def mtx_dist_compute(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.mtx_dist_compute(cname='cam%d' % i)
            return
        criteria1 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria0 = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

        size = (int(self.cBHeight), int(self.cBWidth))

        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = (self.cBSquareSize * np.mgrid[0:size[0], 0:size[1]]).T.reshape(-1, 2)

        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        cap = cv.VideoCapture(videopath % (cname, 'intrinsics'))
        if not cap.isOpened():
            return
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            h, w = img.shape[:2]
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, size, None, criteria0)
            if ret:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria1)
                imgpoints.append(corners2)
        cap.release()
        if not objpoints:
            return
        print('start calibrating')

        random.seed()
        while True:
            objpoints2 = []
            imgpoints2 = []

            lp = np.arange(len(imgpoints)).tolist()
            lp = random.sample(lp, samplesize)
            for j in range(len(lp)):
                objpoints2.append(objpoints[lp[j]])
                imgpoints2.append(imgpoints[lp[j]])
            lp.clear()

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints2, imgpoints2, (h, w), None, None)
            print(ret)
            if ret < 0.35:
                break
        self.mtx[cname] = mtx
        self.dist[cname] = dist
        self.__update(cname)

    def rt_compute(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.rt_compute(cname='cam%d' % i)
            return
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        axis *= self.cBSquareSize

        size = (int(self.cBHeight), int(self.cBWidth))

        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = (self.cBSquareSize * np.mgrid[0:size[1], 0:size[0]]).T.reshape(-1, 2)

        cap = cv.VideoCapture(videopath % (cname, 'checkerboard'))
        # cap = cv.VideoCapture(videopath % (cname, 'intrinsics'))
        ret, img = cap.read()
        # while not ret:
        # ret, img = cap.read()
        # cap.release()

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        cv.imshow('img', img)
        cv.setMouseCallback('img', click_event)
        cv.waitKey(0)

        # print(click)
        if click == 4:
            # print(manual_position)
            dst_pts = np.float32(manual_position)
            scr_pts = np.float32([[0, 0], [7, 0], [7, 5], [0, 5]])
            scr_pts *= self.cBSquareSize
            M = cv.getPerspectiveTransform(scr_pts, dst_pts)

            img_pts = np.array([[[j, i]] for i in range(6) for j in range(8)], dtype=np.float32)
            img_pts *= self.cBSquareSize
            obj_pts = cv.perspectiveTransform(img_pts, M)
            corners2 = obj_pts

            # corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            retval, self._rvecs[cname], self._tvecs[cname] = \
                cv.solvePnP(objp, corners2, self.mtx[cname], self.dist[cname])
            imgpts, jac = cv.projectPoints(axis, self._rvecs[cname], self._tvecs[cname], self.mtx[cname],
                                           self.dist[cname])
            self.save_xml(cname)
            img = draw(img, corners2, imgpts)
            cv.imshow('img', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        '''
        ret, corners = cv.findChessboardCorners(gray, size, None, criteria0)
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria1)
            _, self._rvecs[cname], self._tvecs[cname], _ = \
                cv.solvePnPRansac(objp, corners2, self.mtx[cname], self.dist[cname])
        '''

    def subtract_background(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.subtract_background(cname='cam%d' % i)
            return

        capbg = cv.VideoCapture(videopath % (cname, 'background'))
        capfg = cv.VideoCapture(videopath % (cname, 'video'))
        ret, bg = capbg.read()
        while not ret:
            ret, bg = capbg.read()
        ret, fg = capfg.read()
        while not ret:
            ret, fg = capfg.read()
        capbg.release()
        capfg.release()

        # bg = cv.GaussianBlur(bg, (3, 3), 0)
        # fg = cv.GaussianBlur(fg, (3, 3), 0)
        kernel = np.ones((5, 5), np.uint8)

        backSub = cv.createBackgroundSubtractorMOG2(detectShadows=True)
        for i in range(10):
            backSub.apply(bg)
        mask = backSub.apply(fg)
        mask[mask == 127] = 0

        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # using superpixel for dividing background accurately
        ths = 0.4  # the threshold of picking superpixels as foreground
        spp = cv.ximgproc.createSuperpixelLSC(fg)
        spp.iterate(10)
        label = np.array(spp.getLabels()) + 1
        mask[mask == 255] = 1
        mask_t = mask * label
        for i in range(label.min(), label.max() + 1):
            if np.mean(mask_t[label == i]) / i < ths:
                mask[label == i] = 0
            else:
                mask[label == 1] = 1

        mask[mask == 1] = 255
        # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # cv.imshow('mask', mask)
        cv.imwrite(('%s_mask.jpg' % cname), mask)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        mask[mask == 255] = 1
        self.mask[cname] = mask

    '''
    def voxel_pos(self, block_size):
        data = []
        imgp = []
        step = 5
        xrange = range(-15 * step, 15 * step)
        yrange = range(-15 * step, 15 * step)
        zrange = range(-15 * step, 15 * step)

        objp = np.zeros((len(xrange) * len(yrange) * len(zrange), 3), np.float32)
        objp[:, :3] = np.mgrid[xrange, yrange, zrange].T.reshape(-1, 3)
        objp /= step

        testrange = range(1, 5)

        for i in range(1, 5):
            imgpts, _ = cv.projectPoints(objp * self.cBSquareSize, self._rvecs['cam%d' % i], self._tvecs['cam%d' % i],
                                         self.mtx['cam%d' % i], self.dist['cam%d' % i])
            imgp.append(imgpts.astype(int))
        for i in range(objp.shape[0]):
            objpt = objp[i]
            score = 0
            for j in testrange:
                imgpts2 = imgp[j - 1][i][0]
                mask = self.mask['cam%d' % j]
                h, w = mask.shape[:2]
                if 0 <= imgpts2[0] < w and 0 <= imgpts2[1] < h:
                    score += mask[imgpts2[1]][imgpts2[0]]
            if score > 3:
                data.append([-objpt[0]*step, -objpt[2]*step, objpt[1]*step])
        
        return data
    '''
    def project_points(self, objp, cam_idx):
        return cv.projectPoints(objp * self.cBSquareSize, self._rvecs['cam%d' % cam_idx], self._tvecs['cam%d' % cam_idx],
                                self.mtx['cam%d' % cam_idx], self.dist['cam%d' % cam_idx])[0]
    
    def voxel_pos_mp(self, block_size):
        data = []
        imgp = []
        step = 5
        xrange = range(-15 * step, 15 * step)
        yrange = range(-15 * step, 15 * step)
        zrange = range(-15 * step, 15 * step)

        objp = np.zeros((len(xrange) * len(yrange) * len(zrange), 3), np.float32)
        objp[:, :3] = np.mgrid[xrange, yrange, zrange].T.reshape(-1, 3)
        objp /= step

        testrange = range(1, 5)

        pool = multiprocessing.Pool()
        results = []
        for i in range(1, 5):
            results.append(pool.apply_async(self.project_points, args=(objp, i)))

        for res in results:
            imgpts = res.get()
            imgp.append(imgpts.astype(int))

        for i in range(objp.shape[0]):
            objpt = objp[i]
            score = 0
            for j in testrange:
                imgpts2 = imgp[j - 1][i][0]
                mask = self.mask['cam%d' % j]
                h, w = mask.shape[:2]
                if 0 <= imgpts2[0] < w and 0 <= imgpts2[1] < h:
                    score += mask[imgpts2[1]][imgpts2[0]]
            if score > 3:
                data.append([-objpt[0]*step, -objpt[2]*step, objpt[1]*step])

        return data
    
    def camera_position(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.camera_position(cname='cam%d' % i)
            return None
        if cname in self._cameraposition:
            return self._cameraposition[cname]
        else:
            R_mat = np.mat(cv.Rodrigues(self._rvecs[cname])[0])

            # print(R_mat)
            # print(cam_angle)
            R_mat = R_mat.T
            cpos = -R_mat * self._tvecs[cname] / (self.cBSquareSize/5)

            cposgl = []
            cposgl.append(cpos[0])  # x
            cposgl.append(-cpos[2])  # z
            cposgl.append(cpos[1])  # y
            self._cameraposition[cname] = cposgl
            # print(cposgl)
            # print('-----------------------')
            return cposgl

    def roi(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.roi(cname='cam%d' % i)
            return

        capv = cv.VideoCapture(videopath % (cname, 'video'))
        ret, img = capv.read()
        while not ret:
            ret = img.read()
        capv.release()
        h, w = img.shape[:2]
        _, roi = cv.getOptimalNewCameraMatrix(self.mtx[cname], self.dist[cname], (w, h), 1, (w, h))
        # print(roi)
        # print('-------------')

    def rot(self, cname=[]):
        if not cname:
            for i in range(1, 5):
                self.rot(cname='cam%d' % i)
            return None
        if cname in self._rotation:
            return self._rotation[cname]
        else:
            R_mat = np.mat(cv.Rodrigues(self._rvecs[cname])[0])
            R_mat_gl = R_mat[:, [0, 2, 1]]
            R_mat_gl[1, :] *= -1

            gl_rot_mat = np.eye(4)
            gl_rot_mat[:3, :3] = R_mat_gl
            gl_rot_mat[3, :3] = [0, 0, 0]
            gl_rot_mat[:3, 3] = [0, 0, 0]

            gl_rot_mat = glm.mat4(*gl_rot_mat.T.ravel())
            rotation_matrix_y = glm.rotate(glm.mat4(1), glm.radians(-90), glm.vec3(0, 1, 0))
            cam_rotations = gl_rot_mat * rotation_matrix_y

            return cam_rotations


# for testing
cc = CameraConfig()
cc.load_xml()
cc.save_xml()
# cc.mtx_dist_compute()
# cc.rt_compute()
# cc.subtract_background()
# cc.camera_position()
# data = cc.voxel_pos(1.0)
