# @title style_clip_draw() Definition (Fast Version)
import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from utils import show_img,get_image_augmentation,initialize_curves,save_svg,init_image,pil_resize_long_edge_to,pil_to_np,pil_loader_internet,np_to_tensor,Vgg16_Extractor,render_drawing,calculate_loss,sample_indices,render_scaled
import cv2
from PIL import Image
import pydiffvg
import clip
import tqdm
import loss_utils
import numpy as np
import config
import torch.nn.functional as F
import ttools.modules
# device = torch.device('cuda')
device = torch.device("cuda:0")
model, preprocess = clip.load('ViT-B/32', device, jit=False)
pydiffvg.set_print_timing(False)
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)
def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    image = image.to(device)
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image-mean)/std
    return image
def main(args):
    loss_func = loss_utils.Loss(args)
    min_delta = 1e-5
    print('layer:', 1)
    svg_name = os.path.basename(args.svg)
    canvas_width, canvas_height, shapes_svg, shape_groups_svg = \
        pydiffvg.svg_to_scene(args.svg)
    # canvas_width, canvas_height = 256, 256
    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        canvas_width, canvas_height, shapes_svg, shape_groups_svg)
    img = render(canvas_width,  # width
                 canvas_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,  # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1],
                                                      3, device=pydiffvg.get_device()) * (
                  1 - img[:, :, 3:4])
    pydiffvg.imwrite(img.cpu(), './content/svg_img/init.png', gamma=1.0)
    img = img[:, :, :3]
    # Convert img from HWC to NCHW
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2).to(device)  # NHWC -> NCHW
    print("img=========",img.size())
    source_features = model.encode_image(clip_normalize(img, device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    # Image Augmentation Transformation
    augment_trans = get_image_augmentation(args.use_normalized_clip)
    image_name = os.path.basename(args.image_path)

    # points_vars = []
    # stroke_width_vars = []
    # color_vars = []
    # for path in shapes:
    #     path.points.requires_grad = True
    #     points_vars.append(path.points)
    #     path.stroke_width.requires_grad = True
    #     stroke_width_vars.append(path.stroke_width)
    # for group in shape_groups:
    #     group.stroke_color.requires_grad = True
    #     color_vars.append(group.stroke_color)
    # Initialize Random Curves
    shapes, shape_groups = initialize_curves(args.num_paths, canvas_width, canvas_height)
    # shapes = shapes + shapes_svg
    # shape_groups = shape_groups + shape_groups_svg
    # points_vars = []
    # for path in shapes:
    #     path.points.requires_grad = True
    #     points_vars.append(path.points)
    # color_vars = []
    # for group in shape_groups:
    #     group.fill_color.requires_grad = True
    #     color_vars.append(group.fill_color)
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)
    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # points_vars = [l.data.requires_grad_() for l in points_vars]
    texture = Image.open(args.image_path)
    counter = 0
    target = torch.Tensor(np.array(texture))
    target = target[:, :, :3]
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2)
    target = target / 255
    target = target.to(device)
    print("====target.size=====", target.size())
    target_features = model.encode_image(clip_normalize(target, device))
    target_features /= (target_features.clone().norm(dim=-1, keepdim=True))
    print("====target_features=====", target_features.size())
    print("====source_features=====", source_features.size())
    # Run the main optimization loop
    for t in range(args.num_iter) if args.debug else tqdm.tqdm(range(args.num_iter)):
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            canvas_width, canvas_height, shapes, shape_groups)
        # canvas_width, canvas_height = 256, 256
        render = pydiffvg.RenderFunction.apply
        img2 = render(canvas_width,  # width
                      canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      0,  # seed
                      None,  # bg
                      *scene_args)
        if t % 5 == 0:
            pydiffvg.imwrite(img2.cpu(), './content/{}_num{}/iter_{}.png'.format(image_name, args.num_paths, int(t)),
                             gamma=1.0)
            save_svg(f"./svg_logs", f"{image_name}_num{args.num_paths}_{int(t)}", canvas_width, canvas_height, shapes,
                     shape_groups)

        img2 = img2[:, :, :3]
        img2 = img2.unsqueeze(0)
        img2 = img2.permute(0, 3, 1, 2)  # NHWC -> NCHW
        img2 = img2.to(device)

        # shapes_new = shapes + shapes_svg
        # shape_groups_new = shape_groups + shape_groups_svg
        # scene_args_new = pydiffvg.RenderFunction.serialize_scene( \
        #     canvas_width, canvas_height, shapes_new, shape_groups_new)
        # # canvas_width, canvas_height = 256, 256
        # render = pydiffvg.RenderFunction.apply
        # img_new = render(canvas_width,  # width
        #               canvas_height,  # height
        #               2,  # num_samples_x
        #               2,  # num_samples_y
        #               0,  # seed
        #               None,  # bg
        #               *scene_args_new)
        # if t % 5 == 0:
        #     pydiffvg.imwrite(img_new.cpu(), './content/{}_num{}/iter_{}.png'.format(image_name, args.num_paths, int(t)),
        #                      gamma=1.0)
        #     save_svg(f"./svg_logs", f"{image_name}_new_num{args.num_paths}_{int(t)}", canvas_width, canvas_height, shapes_new,
        #              shape_groups_new)
        #
        # img_new = img_new[:, :, :3]
        # img_new = img_new.unsqueeze(0)
        # img_new = img_new.permute(0, 3, 1, 2)  # NHWC -> NCHW
        # img_new = img_new.to(device)
        # print("====img_new=====", img_new.size())
        print("====img=====", img.size())
        print("====img2=====", img2.size())
        # image_new_features = model.encode_image(clip_normalize(img_new, device))
        # image_new_features /= (image_new_features.clone().norm(dim=-1, keepdim=True))

        image_features = model.encode_image(clip_normalize(img, device))
        # image_features /= (image_new_features.clone().norm(dim=-1, keepdim=True))

        image2_features = model.encode_image(clip_normalize(img2, device))
        image2_features /= (image2_features.clone().norm(dim=-1, keepdim=True))


        loss = 0
        # for n in range(args.num_augs):
        losses_dict = loss_func(img2, target, counter)
        loss = sum(list(losses_dict.values()))
        loss_lpips = (img2 - target).pow(2).mean()
        loss += loss_lpips
        # loss_f = 1 - torch.cosine_similarity(image_new_features, image_features, dim=1)
        # loss += loss_f
        # print("loss_lpips=======", loss_lpips)
        # loss_dict = 1 - torch.cosine_similarity(img2, img, dim=1)
        # loss += loss_dict.mean()
        # loss = loss_lpips
        #!!!!!
        # losses_dict = loss_func(img, target, counter)
        # loss = sum(list(losses_dict.values()))
        print("final loss=======", loss)
        loss.backward()
        points_optim.step()
        color_optim.step()
        #
        for path in shapes:
            path.stroke_width.data.clamp_(1.0, 50)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)
        # for group in shape_groups:
        #     group.fill_color.data.clamp_(0.0, 1.0)
    # return render_scaled(image_name, args.num_paths, shapes, shape_groups, canvas_width, canvas_height,
    #                      scale_factor=2, t=t).detach().cpu().numpy()[0]
    final_img = render(canvas_width, canvas_height, shapes, shape_groups)
    pydiffvg.imwrite(final_img.cpu(), './content/{}_num{}/final_img.png'.format(image_name, int(t)),gamma=1.0)


# img = style_clip_draw_slow('The bear is playing basketball.', './inputs/start.png')
# # img = style_clip_draw_slow('./inputs/banhua/banhua3.png', './style/6.jpg', 256)
# show_img(img)
if __name__ == "__main__":
    args = config.parse_arguments()
    args.device = device
    main(args)
