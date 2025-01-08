import os
from PIL import Image
import random
import numpy as np
from torchvision import transforms


# 定义二值化函数
def binary_func(x):
    # 这里假设阈值为5，小于等于5返回0（黑色），大于5返回255（白色）
    return 0 if x < 5 else 255


def get_image_list(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            # path = os.path.join(root, fname)
            path = root+"/" +fname
            images.append(path)
    return images


def tobinImage(mask):
    # 分别获取R、G、B通道的数据
    r, g, b = mask.split()

    # 对每个通道应用二值化函数
    r_binary = r.point(binary_func)
    g_binary = g.point(binary_func)
    b_binary = b.point(binary_func)

    # 合并三个二值化后的通道
    binary_img = Image.merge('RGB', (r_binary, g_binary, b_binary))
    mask_np=np.array(binary_img)
    # print(mask_np[10,14])
    return binary_img

def irregular_hole_synthesize(img, mask, rotate=False):
    if mask.mode != "RGB":
        mask = mask.convert("RGB")
    if img.mode != "RGB":
        img = img.convert("RGB")
    # 将mask与原图的尺寸一致
    if mask.size != img.size:
        width, height = mask.size
        crop_width, crop_height = img.size

        width, height = mask.size
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        # crop_area=(0,0,width,height)
        crop_area = (left, top, right, bottom)
        mask = mask.crop(crop_area)

    # 烟雾mask不进行旋转
    if rotate:
        mask=random_rotate_image(mask, degrees_range=(0, 360))  # 0到360度之间随机旋转

    img_np = np.array(img).astype('uint8')
    mask_np = np.array(mask).astype('uint8')
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype('uint8')).convert("RGB")

    return hole_img, mask


def random_rotate_image(img, degrees_range=(0, 360)):
    # 生成随机旋转角度
    rotation_angle = random.randint(degrees_range[0], degrees_range[1])
    # 对图片进行旋转
    rotated_img = img.rotate(rotation_angle)

    return rotated_img


if __name__ == "__main__":
    for_genflog_path = r"dataset/clean"
    mask_flog_path = r"dataset/mask/mask_flog"
    maskwhite_path = r"dataset/mask/black_and_white"

    for_genflog_list = get_image_list(for_genflog_path)
    mask_flog_list = get_image_list(mask_flog_path)
    mask_white_list = get_image_list(maskwhite_path)

    # crop = transforms.RandomCrop((256,256))

    output_path=r"dataset/cond_images"          # 合成图片输出的地址
    gt_path=os.path.join(output_path,"gt")
    cond_image_path=os.path.join(output_path,"cond_images")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(gt_path)
        os.makedirs(cond_image_path)

    for i in range(len(for_genflog_list)):
        filename = for_genflog_list[i].split("/")[-1]
        gt_save_path = os.path.join(gt_path, filename)
        cond_image_save_path = os.path.join(cond_image_path, filename)

        # print(mask_white0_save_path)
        # print(mask_white_save_path)
        print("[{}/{}] previous images: {}".format(i+1,len(for_genflog_list),gt_save_path))
        print("[{}/{}] generated images: {}".format(i+1,len(for_genflog_list),cond_image_save_path))

        for_genflog = Image.open(for_genflog_list[i]).convert("RGB")
        if for_genflog.size != (256, 256):
            for_genflog = transforms.Resize((256, 256))(for_genflog)

        j = random.randint(0, len(mask_flog_list) - 1)
        k = random.randint(0, len(mask_white_list) - 1)
        mask_flog = Image.open(mask_flog_list[j]).convert("RGB")
        image_flog, _ = irregular_hole_synthesize(for_genflog, mask_flog)
        mask_white = Image.open(mask_white_list[k]).convert("RGB")
        image_white, _ = irregular_hole_synthesize(image_flog, mask_white,rotate=True)

        for_genflog.save(gt_save_path)
        image_white.save(cond_image_save_path)

    print("Number of generated images: ",len(for_genflog_list))