import os
import time
import argparse
import numpy as np
from models.gradcam import YOLOV7GradCAM
from models.yolov7oddete import YOLOV7ODDete
import cv2
from deep_utils import Box, split_extension

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="yolov7-tiny.pt", help='Path to the model')
parser.add_argument('--img-path', type=str, default='inference/images', help='input image path')
parser.add_argument('--output-dir', type=str, default='images_heatmap', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")

# 这里修改层数
parser.add_argument('--target-layer', type=str, default='76',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')

args = parser.parse_args()

def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat

def put_text_box(bbox, cls_name, res_img):
    x1, y1, x2, y2 = bbox
    cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    res_img = cv2.imread('temp.jpg')
    return res_img

def concat_images(images):
    w, h = images[0].shape[:2]
    width = w
    height = h * len(images)
    base_img = np.zeros((width, height, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        base_img[:, h * i:h * (i + 1), ...] = img
    return base_img

def main(img_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    img = cv2.imread(img_path)
    print('[INFO] Loading the model')
    model = YOLOV7ODDete(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))
    torch_img = model.preprocessing(img[..., ::-1])
    if args.method == 'gradcam':
        saliency_method = YOLOV7GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)
    tic = time.time()
    masks, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
    print("total time:", round(time.time() - tic, 4))
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]
    images = [result]
    for i, mask in enumerate(masks):
        res_img = result.copy()
        bbox, cls_name = boxes[0][i], class_names[0][i]
        res_img, heat_map = get_res_img(bbox, mask, res_img)
        res_img = put_text_box(bbox, cls_name, res_img)
        images.append(res_img)
    final_image = concat_images(images)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
    output_path = f'{args.output_dir}/{img_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Saving the final image at {output_path}')
    cv2.imwrite(output_path, final_image)

def folder_main(folder_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    print('[INFO] Loading the model')
    model = YOLOV7ODDete(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))
    for item in os.listdir(folder_path):
        img_path = os.path.join(folder_path, item)
        img = cv2.imread(img_path)
        torch_img = model.preprocessing(img[..., ::-1])
        saliency_method = YOLOV7GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)
        tic = time.time()
        masks, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
        print("total time:", round(time.time() - tic, 4))
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        images = [result]
        for i, mask in enumerate(masks):
            res_img = result.copy()
            bbox, cls_name = boxes[0][i], class_names[0][i]
            res_img, heat_map = get_res_img(bbox, mask, res_img)
            res_img = put_text_box(bbox, cls_name, res_img)
            images.append(res_img)
            
            img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
            
            img_name_str = f'{i}_{img_name}'
           
            img_root = os.path.split(img_path)[1].split('.')[0]
            
            
            
            output_path = f'{args.output_dir}/{img_root}/{img_name_str}'
            os.makedirs(f'{args.output_dir}/{img_root}', exist_ok=True)
            print(f'[INFO] Saving the final image at {output_path}')
            cv2.imwrite(output_path, res_img)
        
        final_image = concat_images(images)
        img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
        output_path = f'{args.output_dir}/{img_name}'
        os.makedirs(args.output_dir, exist_ok=True)
        print(f'[INFO] Saving the final image at {output_path}')
        cv2.imwrite(output_path, final_image)

if __name__ == '__main__':
    if os.path.isdir(args.img_path):
        folder_main(args.img_path)
    else:
        main(args.img_path)
