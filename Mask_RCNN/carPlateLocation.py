import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import Mask_RCNN.mrcnn.model as modelib
from matplotlib import patches


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class CarplateConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "carplate"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + carplate

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class LicenseLocator:
    def __init__(self, image, modelPath="Mask_RCNN/mask_rcnn_carplate_0030.h5"):
        self.image = image
        self.height, self.width = self.image.shape[:2]
        # Root directory of the project
        self.ROOT_DIR = os.path.abspath("./")

        # Import Mask RCNN
        sys.path.append(os.path.join(self.ROOT_DIR, "Mask_RCNN"))  # To find local version of the library

        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")

        # Path to carPalateLocation trained weights
        self.CARPLATE_WEIGHTS_PATH = modelPath

        # 车牌定位
        self.config = CarplateConfig()

        # Override the training configurations with a few
        # changes for inferencing.
        class InferenceConfig(self.config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.config = InferenceConfig()

        # Device to load the neural network on.
        # Useful if you're training a model on the same
        # machine, in which case use CPU and leave the
        # GPU for training.
        self.DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        self.TEST_MODE = "inference"
        self.flag = True
        self.fit()

    def fit(self):
        with tf.device(self.DEVICE):
            model = modelib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)
        # Load weights
        self.weights_path = self.CARPLATE_WEIGHTS_PATH
        print("Loading weights", self.weights_path)
        model.load_weights(self.weights_path, by_name=True)

        # Run Detection 这里为模型标注数据
        # print(self.image.shape[:2])
        self.image = modelib.load_image_gt(self.image, self.config)
        # print(self.image.shape[:2])
        # 模型预测功能
        self.results = model.detect([self.image], verbose=0)
        self.r = self.results[0]
        # 1.提取车牌区域，只取第一个车牌
        mask = self.r['masks'][:, :, 0].astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        try:
            cnt = contours[0]
        except Exception:
            self.flag = False
            print("Model Location Error")
            return None
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = approx.squeeze()
        if approx.shape == (4, 2):
            self.box = np.zeros_like(approx)
            self.box[:, 0] = approx[:, 1]
            self.box[:, 1] = approx[:, 0]
        else:
            rect = cv2.minAreaRect(np.array(np.nonzero(mask)).T)
            self.box = cv2.boxPoints(rect).astype(np.int)

    def getRectImage(self):
        def get_ax():
            """Return a Matplotlib Axes array to be used in
            all visualizations in the notebook. Provide a
            central point to control graph sizes.

            Adjust the size attribute to control how big to render images
            """
            fig, ax = plt.subplots()
            return fig, ax

        if self.flag == True:
            N = self.r['rois'].shape[0]
            if not N:
                print("\n*** No instances to display *** \n")
            else:
                assert self.r['rois'].shape[0] == self.r['masks'].shape[-1]
            fig, ax = get_ax()
            for i in range(N):
                # Bounding box
                if not np.any(self.r['rois'][i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = self.r['rois'][i]
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                      alpha=0.7, linestyle="dashed",
                                      edgecolor="red", facecolor='none')
                ax.add_patch(p)
            height, width = self.image.shape[:2]
            ax.imshow(self.image, aspect='equal')
            plt.axis('off')
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.axis('off')
            plt.savefig("temp.jpg", dpi=300)
            img = cv2.imread("temp.jpg")
            a = round((height - self.height) / 2)
            b = round((width - self.width) / 2)
            img = img[a:a + self.height, b:b + self.width]
            os.remove("temp.jpg")
            # height, width = self.image.shape[:2]
            #
            # ax.imshow(image, aspect='equal')
            # plt.axis('off')
            # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # plt.axis('off')
            # plt.savefig("temp.jpg", dpi=300)

            # img = cv2.imread("temp.jpg")
            # a = round((height - self.height) / 2)
            # b = round((width - self.width) / 2)
            # image = self.image[a:a + self.height, b:b + self.width]
            # os.remove("temp.jpg")
            return img
        else:
            return None

    def getLicenseImage(self):
        if self.flag == True:
            y0, x0 = self.box.min(axis=0)
            y1, x1 = self.box.max(axis=0)
            img = self.image[y0:y1, x0:x1]
            res = self.correction(img, self.box, x0, y0)
            return res
        else:
            return None

    def correction(self, img, box, x0, y0):
        """
        :param img:输入待纠正的图片
        :param box:待纠正的图片的最大外界矩阵
        :param x0:旋转中心的横坐标
        :param y0:旋转中心纵坐标
        :return:纠正后图片
        """
        box[:, 0] -= y0
        box[:, 1] -= x0
        # 调整box顺序，从左上角开始，逆时针转动
        i0 = (box[:, 0] + box[:, 1]).argmin()
        box = box[[i0, (i0 + 1) % 4, (i0 + 2) % 4, (i0 + 3) % 4]]

        h = np.max([box[1][0] - box[0][0], box[2][0] - box[3][0]])
        w = np.max([box[2][1] - box[1][1], box[3][1] - box[0][1]])
        box2 = np.array([(0, 0), (h, 0), (h, w), (0, w)])
        M = cv2.getPerspectiveTransform(box[:, ::-1].astype(np.float32), box2[:, ::-1].astype(np.float32))
        img = cv2.warpPerspective(img, M, (w, h))
        img = cv2.resize(img, (220, 70))
        return img


if __name__ == '__main__':
    # 输入图片 :"0204.jpg", 输出结果: "test.jpg"
    image = cv2.imdecode(np.fromfile("../dataset/036.jpg", dtype=np.uint8), -1)
    model = LicenseLocator(image, 'mask_rcnn_carplate_0030.h5')
    rectImage = model.getRectImage()
    licenseImage = model.getLicenseImage()
    if rectImage is not None:
        cv2.imshow('1', rectImage)
    if licenseImage is not None:
        cv2.imshow('2', licenseImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
