import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image

ort_session = ort.InferenceSession('./models/super_resolution.onnx')
print(ort_session.get_inputs()[0])
img = Image.open("./images/cat12.jpg")
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()
input = np.expand_dims(np.expand_dims(np.array(img_y, dtype=np.float32),axis=0),axis=0)
ort_inputs = {
    "input": input
}
print(input.shape)
ort_output = ort_session.run(None, ort_inputs)[0]
print(ort_output)

mg_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")