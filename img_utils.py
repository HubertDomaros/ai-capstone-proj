import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\examples\image_0000005.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('img', img)