# import numpy as np
# import cv2
# import albumentations as A
# from albumentations.core.bbox_utils import BboxParams
#
#
# pipeline = A.Compose([
#     A.Rotate(limit=(-30,30), p=0.55),
#     A.PlasmaBrightnessContrast(p=0.3),
#     A.GaussianBlur(p=0.3),
#     A.GaussianBlur(p=0.2),
#     A.AtLeastOneBBoxRandomCrop(p=0.35),
#     A.ChannelDropout(p=0.2),
#     A.Perspective(p=0.3),
# ], bbox_params=BboxParams())
#
#
#
