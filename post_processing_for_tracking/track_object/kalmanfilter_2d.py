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

    def __init__(self, bbox, enable_voting=False, results=None):
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
        self.bbox_old = bbox
        self.check_motion = []
        # self.uuid = bbox[-1]
        # self.uuid_match = np.where((dets[:, -1] == self.uuid) & (dets[:, 0] != bbox[0]))[0]
        # self.list_bbox_match = dets[self.uuid_match]
        self.class_id_r2 = None  # 0 - driver , 1 - p1, 2 - p2
        self.enable_motion = False

        # voting
        self.frame_id = 0
        self.class_ = None
        self.dir_class = [{"id": 0, "prob": 0.0}, {"id": 1, "prob": 0.0},
                          {"id": 2, "prob": 0.0}, {"id": 3, "prob": 0.0},
                          {"id": 4, "prob": 0.0}, {"id": 5, "prob": 0.0},
                          {"id": 6, "prob": 0.0}]
        self.class_voting1 = None
        self.count__ = 0
        self.class_voting2 = None
        self.enable_voting = enable_voting

        self.class_head_p2 = None
        self.count_head_p2 = 0
        self.class_head_p1 = None
        self.count_head_p1 = 0

        # attribute
        humans = results.humans
        heads = results.heads

        self.bbox_human = []
        self.bbox_head = []
        self.head_h = []
        for human in humans:
            box = human.get_box_info()
            heads1 = human.heads
            if len(heads1) == 1:
                self.head_h.append(1)
            else:
                self.head_h.append(0)
            self.bbox_human.append([box[0], box[1], box[2], box[3], box[4], box[5], float(box[6])])
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

    def update(self, bbox, results):
        """
        Updates the state vector with observed bbox.
        """
        humans = results.humans
        heads = results.heads
        self.bbox_human = []
        self.bbox_head = []
        for head in heads:
            box = head.get_box_info()
            self.bbox_head.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if len(bbox) > 0:
            self.kf.update(convert_bbox_to_z(bbox))  # ton cpu
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
            self.bbox_human.append([box[0], box[1], box[2], box[3], int(box[4]), box[5]])
        # check direction
        center_point_current = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
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
        self.bbox_human = np.array(self.bbox_human)
        if self.class_head_p2 == "P2":
            if len(self.bbox_human) == 1:
                if self.direction == 0:
                    if self.bbox_human[0][4] not in [5, 6]:
                        self.bbox_human[:, 4] = 6 if self.bbox_human[0][4] in [2, 4] else 5
                        # print("case 0")
            elif len(self.bbox_human) == 2:
                # if len(self.head_h) == 3:
                    if self.direction == 1:
                        ids = np.argsort(self.bbox_human[:, 1])
                        if self.bbox_human[ids[0]][4] not in [5, 6]:
                            self.bbox_human[ids[0]][4] = 6 if self.bbox_human[ids[0]][4] in [2, 4] else 5
                            # print("case 1")
                    else:
                        ids = np.argsort(self.bbox_human[:, 1])
                        if self.bbox_human[ids[1]][4] not in [5, 6]:
                            self.bbox_human[ids[1]][4] = 6 if self.bbox_human[ids[1]][4] in [2, 4] else 5
                            # print("case 11")

            elif len(self.bbox_human) == 3:
                if self.direction == 1:
                    ids = np.argsort(self.bbox_human[:, 1])
                    if self.bbox_human[ids[0]][4] not in [5, 6]:
                        self.bbox_human[ids[0]][4] = 6 if self.bbox_human[ids[0]][4] in [2, 4] else 5
                    if self.bbox_human[ids[1]][4] not in [3, 4]:
                        self.bbox_human[ids[1]][4] = 4 if self.bbox_human[ids[1]][4] in [2, 6] else 3
                    if self.bbox_human[ids[2]][4] not in [1, 2]:
                        self.bbox_human[ids[2]][4] = 2 if self.bbox_human[ids[2]][4] in [4, 6] else 1
                else:
                    ids = np.argsort(self.bbox_human[:, 1])
                    if self.bbox_human[ids[0]][4] not in [1, 2]:
                        self.bbox_human[ids[0]][4] = 2 if self.bbox_human[ids[0]][4] in [4, 6] else 1
                    if self.bbox_human[ids[1]][4] not in [3, 4]:
                        self.bbox_human[ids[1]][4] = 4 if self.bbox_human[ids[1]][4] in [2, 6] else 3
                    if self.bbox_human[ids[2]][4] not in [5, 6]:
                        self.bbox_human[ids[2]][4] = 6 if self.bbox_human[ids[2]][4] in [2, 4] else 5
                    # print("case 2")
        if self.class_head_p2 is None and self.class_head_p1 == "P1":
            if len(self.bbox_human) == 1:
                if self.direction == 0:
                    if self.bbox_human[0][4] not in [4, 3]:
                        self.bbox_human[:, 4] = 4 if self.bbox_human[0][4] in [2, 6] else 3
                        # print("case 3")
                else:
                    if self.bbox_human[0][4] not in [2, 1]:
                        self.bbox_human[:, 4] = 2 if self.bbox_human[0][4] in [4, 6] else 1
                        # print("case 31")

            elif len(self.bbox_human) == 2:
                if self.direction == 1:
                    ids = np.argsort(self.bbox_human[:, 1])
                    if self.bbox_human[ids[0]][4] not in [4, 3]:
                        self.bbox_human[ids[0]][4] = 4 if self.bbox_human[ids[0]][4] in [2, 6] else 3
                    if self.bbox_human[ids[1]][4] not in [1, 2]:
                        self.bbox_human[ids[1]][4] = 2 if self.bbox_human[ids[1]][4] in [4, 6] else 1
                else:
                    ids = np.argsort(self.bbox_human[:, 1])
                    if self.bbox_human[ids[0]][4] not in [1, 2]:
                        self.bbox_human[ids[0]][4] = 2 if self.bbox_human[ids[0]][4] in [4, 6] else 1
                    if self.bbox_human[ids[1]][4] not in [4, 3]:
                        self.bbox_human[ids[1]][4] = 4 if self.bbox_human[ids[1]][4] in [2, 6] else 3
                # print("case 4")

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
