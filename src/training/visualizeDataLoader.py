import matplotlib.pyplot as plt
import cv2

from src.preprocessing import build_augmentation_pipeline

sample_img_path = r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\kaggle\input\codebrim-original\original_dataset\images\image_0000003.jpg'
sample_img = cv2.imread(sample_img_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

sample_bboxes = [[50, 60, 150, 200]]  # Example bbox (xmin, ymin, xmax, ymax)
sample_labels = [tuple([1, 0, 1, 0, 0, 0])]  # Convert list to tuple to avoid TypeError

# Apply augmentation
augmented = build_augmentation_pipeline()(image=sample_img, bboxes=sample_bboxes, category_ids=sample_labels)

plt.imshow(augmented['image'])
for bbox in augmented['bboxes']:
    cv2.rectangle(augmented['image'], (int(bbox[0]), int(bbox[1])),
                  (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
plt.title(f"Labels: {augmented['category_ids']}")
plt.show()