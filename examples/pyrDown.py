import cv2
import torch
import fastcv


img = cv2.imread("../artifacts/test.jpg", cv2.IMREAD_COLOR)
img_tensor = torch.from_numpy(img).cuda()
pyrDown_tensor = fastcv.pyrDown(img_tensor)
pyrDown_image = pyrDown_tensor.cpu().numpy()

cv2.imwrite("output_pyrdown.jpg", pyrDown_image)

print(f"Original shape: {img.shape}")
print(f"Gauss Pyramide shape:   {pyrDown_image.shape}")
print("Saved image.")