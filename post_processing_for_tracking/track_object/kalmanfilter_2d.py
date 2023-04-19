from post_processing_for_tracking.track_object.utils.kalmanfilter_init import KalmanFilter
import numpy as np
import time


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, results=None):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = bbox[5]
        self.class_id = bbox[4]
        self.center_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        self.check_direction = []
        self.motion = 2
        self.direction = 2
        self.class_head_p2 = None
        self.count_head_p2 = 0
        self.class_head_p1 = None
        self.count_head_p1 = 0

        # attribute
        humans = results.humans
        heads = results.heads

        self.human_bboxes = []
        self.bbox_head = []
        self.head_h = []
        for human in humans:
            box = human.get_box_info()
            heads1 = human.heads
            if len(heads1) == 1:
                self.head_h.append(1)
            else:
                self.head_h.append(0)
            self.human_bboxes.append([box[0], box[1], box[2], box[3], box[4], box[5], float(box[6])])
        for head in heads:
            box = head.get_box_info()
            self.bbox_head.append([box[0], box[1], box[2], box[3], box[4], box[5], float(box[6])])

    def IOU(self, bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return (o)

    def direct_detection(self, center_point_current):
        """Detect the motor direction 1 if IN detection and 0 if OUT detection

        Args:
            center_point_current (list): motor center coordinates
        """
        if self.check_direction is not None:
            if center_point_current[1] - self.center_point[1] > 0:
                self.check_direction.append(1)
            else:
                self.check_direction.append(0)
            if len(self.check_direction) >= 3:
                rs_in = self.check_direction.count(1)
                rs_out = self.check_direction.count(0)
                if rs_in >= rs_out:
                    self.direction = 1  # in
                else:
                    self.direction = 0  # out
                self.check_direction = None
        self.center_point = center_point_current

    def P1_P2_checking(self):
        """Checking if P1 and P2 is on the motorbike
        """
        if self.class_head_p2 is None:
            if len(self.bbox_head) == 3:  # > 2
                self.count_head_p2 += 1
            if self.count_head_p2 > 3:  # 1
                self.class_head_p2 = 'P2'
        if self.class_head_p1 is None:
            if len(self.bbox_head) == 2:
                self.count_head_p1 += 1
            if self.count_head_p1 > 3:
                self.class_head_p1 = "P1"

    def attachment_process(self):
        """Reassigning new class for humans on the motorbike
        """

        self.human_bboxes = np.array(self.human_bboxes)
        ###### Passenger 2 (P2) is on the motor ######
        if self.class_head_p2 == "P2":
            if len(self.human_bboxes) == 1:
                if self.direction == 0:
                    if self.human_bboxes[0][4] not in [5, 6]:
                        self.human_bboxes[:, 4] = 6 if self.human_bboxes[0][4] in [2, 4] else 5
            elif len(self.human_bboxes) == 2:
                if self.direction == 1:
                    ids = np.argsort(self.human_bboxes[:, 1])
                    if self.human_bboxes[ids[0]][4] not in [5, 6]:
                        self.human_bboxes[ids[0]][4] = 6 if self.human_bboxes[ids[0]][4] in [2, 4] else 5
                else:
                    ids = np.argsort(self.human_bboxes[:, 1])
                    if self.human_bboxes[ids[1]][4] not in [5, 6]:
                        self.human_bboxes[ids[1]][4] = 6 if self.human_bboxes[ids[1]][4] in [2, 4] else 5

            elif len(self.human_bboxes) == 3:
                if self.direction == 1:
                    ids = np.argsort(self.human_bboxes[:, 1])
                    if self.human_bboxes[ids[0]][4] not in [5, 6]:
                        self.human_bboxes[ids[0]][4] = 6 if self.human_bboxes[ids[0]][4] in [2, 4] else 5
                    if self.human_bboxes[ids[1]][4] not in [3, 4]:
                        self.human_bboxes[ids[1]][4] = 4 if self.human_bboxes[ids[1]][4] in [2, 6] else 3
                    if self.human_bboxes[ids[2]][4] not in [1, 2]:
                        self.human_bboxes[ids[2]][4] = 2 if self.human_bboxes[ids[2]][4] in [4, 6] else 1
                else:
                    ids = np.argsort(self.human_bboxes[:, 1])
                    if self.human_bboxes[ids[0]][4] not in [1, 2]:
                        self.human_bboxes[ids[0]][4] = 2 if self.human_bboxes[ids[0]][4] in [4, 6] else 1
                    if self.human_bboxes[ids[1]][4] not in [3, 4]:
                        self.human_bboxes[ids[1]][4] = 4 if self.human_bboxes[ids[1]][4] in [2, 6] else 3
                    if self.human_bboxes[ids[2]][4] not in [5, 6]:
                        self.human_bboxes[ids[2]][4] = 6 if self.human_bboxes[ids[2]][4] in [2, 4] else 5

        ###### Passenger 2 (P2) is not on the motor and Passenger 1 (P1) is on the motor ######
        if self.class_head_p2 is None and self.class_head_p1 == "P1":
            if len(self.human_bboxes) == 1:
                if self.direction == 0:
                    if self.human_bboxes[0][4] not in [4, 3]:
                        self.human_bboxes[:, 4] = 4 if self.human_bboxes[0][4] in [2, 6] else 3
                else:
                    if self.human_bboxes[0][4] not in [2, 1]:
                        self.human_bboxes[:, 4] = 2 if self.human_bboxes[0][4] in [4, 6] else 1

            elif len(self.human_bboxes) == 2:
                if self.direction == 1:
                    ids = np.argsort(self.human_bboxes[:, 1])
                    if self.human_bboxes[ids[0]][4] not in [4, 3]:
                        self.human_bboxes[ids[0]][4] = 4 if self.human_bboxes[ids[0]][4] in [2, 6] else 3
                    if self.human_bboxes[ids[1]][4] not in [1, 2]:
                        self.human_bboxes[ids[1]][4] = 2 if self.human_bboxes[ids[1]][4] in [4, 6] else 1
                else:
                    ids = np.argsort(self.human_bboxes[:, 1])
                    if self.human_bboxes[ids[0]][4] not in [1, 2]:
                        self.human_bboxes[ids[0]][4] = 2 if self.human_bboxes[ids[0]][4] in [4, 6] else 1
                    if self.human_bboxes[ids[1]][4] not in [4, 3]:
                        self.human_bboxes[ids[1]][4] = 4 if self.human_bboxes[ids[1]][4] in [2, 6] else 3

    def update(self, bbox, results):
        """
        Updates the state vector with observed bbox.
        """
        humans = results.humans
        heads = results.heads
        self.human_bboxes = []
        self.bbox_head = []
        for head in heads:
            box = head.get_box_info()
            self.bbox_head.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if len(bbox) > 0:
            self.kf.update(convert_bbox_to_z(bbox))
        self.score = bbox[5]
        self.class_id = bbox[4]
        self.head_h = []
        for human in humans:
            box = human.get_box_info()
            heads1 = human.heads
            if len(heads1) == 1:
                self.head_h.append(1)
            else:
                self.head_h.append(0)
            self.human_bboxes.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])
        # check direction
        current_center_point = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        self.direct_detection(current_center_point)
        ######## find p1,p2
        self.P1_P2_checking()
        ### update p1p2
        self.attachment_process()

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)
