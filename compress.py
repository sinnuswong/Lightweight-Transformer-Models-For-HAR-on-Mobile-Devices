from PIL import Image
import os

downloads = '/Users/sinnus/Downloads/image'
save_path = '/Users/sinnus/Downloads/bad'

for filename in os.listdir(downloads):
    print(filename)
    # 打开图像
    image = Image.open(downloads+"/"+filename)
    #
    # # 选择输出的文件名
    output_image_path = save_path+"/"+filename
    #
    # # 压缩图像
    # # 使用 `quality` 参数来控制压缩质量，范围为 1-95，值越低，图像越小，质量越低
    image.save(output_image_path, "JPEG", quality=50)
