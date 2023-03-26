# Original Xanthius (https://xanthius.itch.io/multi-frame-rendering-for-stablediffusion)
# 二次修改 
# 三次修改 szxysdt [水煮咸鱼三段突] (https://space.bilibili.com/80438516)

import numpy as np
from tqdm import trange
from PIL import Image, ImageSequence, ImageDraw, ImageFilter, PngImagePlugin

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from modules import deepbooru
from modules.script_callbacks import ImageSaveParams, before_image_saved_callback
from modules.shared import opts, cmd_opts, state
from modules.sd_hijack import model_hijack

import pandas as pd

import piexif
import piexif.helper

import os, re
import time

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def gr_show_value_none(visible=True):
    return {"value": None, "visible": visible, "__type__": "update"}

def gr_show_and_load(value=None, visible=True):
    if value:
        if value.orig_name.endswith('.csv'):
            value = pd.read_csv(value.name)
        else:
            value = pd.read_excel(value.name)
    else:
        visible = False
    return {"value": value, "visible": visible, "__type__": "update"}

class Script(scripts.Script):
    def title(self):
        return "(Beta) Multi-frame Video inpainting Pro-No org"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):  
        with gr.Row():
            input_dir = gr.Textbox(label='Input directory', lines=1)
            output_dir = gr.Textbox(label='Output directory', lines=1)
        # reference_imgs = gr.UploadButton(label="Upload Guide Frames", file_types = ['.png','.jpg','.jpeg'], live=True, file_count = "multiple")   
        first_denoise = gr.Slider(minimum=0, maximum=1, step=0.05, label='第一张的去噪强度', value=1, elem_id=self.elem_id("first_denoise"))
        other_denoise = gr.Slider(minimum=0, maximum=1, step=0.05, label='后续帧的去噪强度', value=0.7, elem_id=self.elem_id("other_denoise"))
        
        append_interrogation = gr.Dropdown(label="Append interrogated prompt at each iteration", choices=["None", "CLIP", "DeepBooru"], value="None")
        # third_frame_image = gr.Dropdown(label="Third Frame Image", choices=["None", "FirstGen", "OriginalImg", "Historical"], value="FirstGen")
        
        # multiframe processing pro
        use_multiframe_pro = gr.Dropdown(label="开启多帧重绘魔改版！（选Multiframe开启）", choices=["Multiframe", "None"], value="Multiframe")
        # num_of_inputs = gr.Slider(minimum=3, maximum=6, step=1, label='Num of frames to process()', value=3, elem_id=self.elem_id("num_of_inputs"))
        num_of_front_frame = gr.Slider(minimum=1, maximum=5, step=1, label='Num of frames have been processed', value=1, elem_id=self.elem_id("num_of_front_frame"))
        num_of_first_frame = gr.Slider(minimum=1, maximum=5, step=1, label='Num of first frame(whitch will be repeated)', value=1, elem_id=self.elem_id("num_of_first_frame"))
        org_alpha = gr.Slider(minimum=0, maximum=1, step=0.01, label='原图的透明度，降低为0，则为从空白图生成', value=0.1, elem_id=self.elem_id("org_alpha"))
        max_frames = gr.Number(label='max_frames', precision=0, value=30)
        
        color_correction_enabled = gr.Checkbox(label="Enable Color Correction", value=False, elem_id=self.elem_id("color_correction_enabled"))
        unfreeze_seed = gr.Checkbox(label="Unfreeze Seed", value=False, elem_id=self.elem_id("unfreeze_seed"))
        loopback_source = gr.Dropdown(label="Loopback Source", choices=["PreviousFrame", "InputFrame","FirstGen"], value="InputFrame")

        with gr.Row():
            use_txt = gr.Checkbox(label='Read tags from text files')

        with gr.Row():
            txt_path = gr.Textbox(label='Text files directory (Optional, will load from input dir if not specified)', lines=1)

        with gr.Row():
            use_csv = gr.Checkbox(label='Read tabular commands')
            csv_path = gr.File(label='.csv or .xlsx', file_types=['file'], visible=False)

        with gr.Row():
            with gr.Column():
                table_content = gr.Dataframe(visible=False, wrap=True)

        use_csv.change(
            fn=lambda x: [gr_show_value_none(x), gr_show_value_none(False)],
            inputs=[use_csv],
            outputs=[csv_path, table_content],
        )
        csv_path.change(
            fn=lambda x: gr_show_and_load(x),
            inputs=[csv_path],
            outputs=[table_content],
        )

        return [append_interrogation,
                input_dir,
                output_dir,
                first_denoise,
                other_denoise,
                use_multiframe_pro,
                num_of_front_frame,
                num_of_first_frame,
                org_alpha,
                max_frames,
                color_correction_enabled,
                unfreeze_seed,
                loopback_source,
                use_csv,
                table_content,
                use_txt,
                txt_path
               ]

    def run(self, p, 
            append_interrogation,
            input_dir,
            output_dir,
            first_denoise, 
            other_denoise,
            use_multiframe_pro,
            num_of_front_frame,
            num_of_first_frame,
            org_alpha,
            max_frames,
            color_correction_enabled,
            unfreeze_seed,
            loopback_source,
            use_csv,
            table_content,
            use_txt,
            txt_path
           ):
        if output_dir is "":
            outpath ="output-" + time.strftime("%Y%m%d-%H%M-%S", time.localtime())

            output_dir = os.path.join(input_dir, outpath)
            
        freeze_seed = not unfreeze_seed

        num_of_all_frame = num_of_front_frame + 1 + num_of_first_frame

        if use_csv:
            prompt_list = [i[0] for i in table_content.values.tolist()]
            prompt_list.insert(0, prompt_list.pop())

        # 输入
        reference_imgs = [os.path.join(input_dir, f) for f in (os.listdir(input_dir)) if re.match(r'.+\.(jpg|png)$', f)]
        
        reference_imgs = sorted(reference_imgs, key=lambda x:int(x.split('/')[-1].split('\\')[-1].split('.')[0]))
        
        # print(f'Will process following files: {", ".join(reference_imgs)}')

        if use_txt:
            if txt_path == "":
                files = [re.sub(r'\.(jpg|png)$', '.txt', path) for path in reference_imgs]
            else:
                files = [os.path.join(txt_path, os.path.basename(re.sub(r'\.(jpg|png)$', '.txt', path))) for path in reference_imgs]
            prompt_list = [open(file, 'r').read().rstrip('\n') for file in files]

        loops = len(reference_imgs) if len(reference_imgs) > max_frames else max_frames
        print(f'Will process following files: {", ".join(reference_imgs[:loops])}')
        
        
              
        processing.fix_seed(p)
        batch_count = p.n_iter

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        initial_width = p.width
        initial_img = reference_imgs[0] # p.init_images[0]

        grids = []
        all_images = []
        # 传入图像
        original_init_image = p.init_images
        # 传入prompt
        original_prompt = p.prompt
        if original_prompt != "":
            original_prompt = original_prompt.rstrip(', ') + ', ' if not original_prompt.rstrip().endswith(',') else original_prompt.rstrip() + ' '
        # original_denoise = p.denoising_strength
        # original_denoise = 0.3
        
        state.job_count = loops * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        # for n in range(batch_count):
        history = None
        # 多帧缓存，保存前面跑完的各个帧
        multiframepro_frames = []
        # frames = []
        last_image = None
        last_image_index = 0
        frame_color_correction = None

        # Reset to original init image at the start of each batch
        p.init_images = original_init_image
        p.width = initial_width

        for i in range(loops):
            if state.interrupted:
                break
            filename = os.path.basename(reference_imgs[i])
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True
            p.control_net_input_image = Image.open(reference_imgs[i]).convert("RGBA").resize((initial_width, p.height), Image.ANTIALIAS)

            if(i > 0):
                # 初始化loopback图像源，并根据需求进行修改
                loopback_image = p.init_images[0]
                # 选择输入源为输入帧
                if loopback_source == "InputFrame":
                    loopback_image = p.control_net_input_image
                # 当循环源为第一张图的时候，输入为第一张图（history就是第一张图）（这是要干嘛？？？？）
                elif loopback_source == "FirstGen":
                    loopback_image = history

                if use_multiframe_pro != "None":
                    ###############################################################################
                    p.width = initial_width * num_of_all_frame
                    img = Image.new("RGB", (initial_width*num_of_all_frame, p.height), "white")

                    # # 把历史帧放进去
                    # for mpf_count in range(1,num_of_inputs-1):

                    #     if mpf_count < len(multiframepro_frames):
                    #         img.paste(multiframepro_frames[-mpf_count], (initial_width*(num_of_inputs-2-mpf_count), 0))
                    #     else:
                    #         img.paste(multiframepro_frames[0], (initial_width*(num_of_inputs-2-mpf_count), 0))
                    # 把历史帧放进去
                    for mpf_count in range(num_of_front_frame):
                        if mpf_count < len(multiframepro_frames):
                            img.paste(multiframepro_frames[-mpf_count-1], (initial_width*(num_of_front_frame-1-mpf_count), 0))
                        else:
                            img.paste(multiframepro_frames[0], (initial_width*(num_of_front_frame-1-mpf_count), 0))                      

                    # img.paste(p.init_images[0], (0, 0))  # 把刚刚画完的上一帧贴在左边

                    # 要画的图贴在中间
                    img2 = Image.new("RGBA", (initial_width, p.height), "black")
                    img2 = Image.blend(img2, loopback_image, org_alpha)
                    img.paste(img2, (initial_width*(num_of_front_frame), 0))  

                    # 第一步时，存下第一张图
                    if i == 1:
                        last_image = p.init_images[0]
                    # 把输入的第一个跑完的图，丢在右边
                    for count_last in range(num_of_first_frame):
                        img.paste(last_image, (initial_width*(num_of_front_frame+1+count_last), 0))

                    p.init_images = [img]  # 把img放进去处理，待会送出去开跑
                    ###############################################################################

                    if color_correction_enabled:
                        p.color_corrections = [processing.setup_color_correction(img)]
                    ###############################################################################

                    # CTRLnet的输入控制
                    # 创建CTRL输入（默认为黑图）
                    ctrl_input_image = Image.new("RGB", (initial_width*num_of_all_frame, p.height))
                    # # 往里面塞CTRL图层
                    # for ctrl_count in range(num_of_inputs-1):
                    #     if i + 2 - num_of_inputs + ctrl_count < 0:
                    #         ctrl_input_image.paste(Image.open(reference_imgs[0]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (initial_width*(ctrl_count), 0))
                    #     else:
                    #         ctrl_input_image.paste(Image.open(reference_imgs[i+2-num_of_inputs+ctrl_count]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (initial_width*(ctrl_count), 0))
                    # 往里面塞CTRL图层
                    for ctrl_count in range(num_of_front_frame+1):
                        if i - num_of_front_frame + ctrl_count < 0:
                            ctrl_input_image.paste(Image.open(reference_imgs[0]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (initial_width*(ctrl_count), 0))
                        else:
                            ctrl_input_image.paste(Image.open(reference_imgs[i-num_of_front_frame + ctrl_count]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (initial_width*(ctrl_count), 0))

                    # # 左边位置放上一张的图
                    # ctrl_input_image.paste(Image.open(reference_imgs[i-1]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (0, 0))
                    # # 要画的地方放当前输入
                    # ctrl_input_image.paste(p.control_net_input_image, (initial_width*(num_of_inputs-2), 0))
                    # 最后一张，放最右边的（可放第一张，也可放按序号来的图）
                    for count_ctrl_last in range(num_of_first_frame):
                        ctrl_input_image.paste(Image.open(reference_imgs[last_image_index]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (initial_width*(num_of_front_frame+1+count_ctrl_last), 0))
                    p.control_net_input_image = ctrl_input_image
                    ###############################################################################

                    # 潜空间遮罩
                    latent_mask = Image.new("RGB", (initial_width*num_of_all_frame, p.height), "black")
                    latent_draw = ImageDraw.Draw(latent_mask)
                    latent_draw.rectangle((initial_width*(num_of_front_frame),0,initial_width*(num_of_front_frame+1),p.height), fill="white")
                    p.image_mask = latent_mask
                    p.denoising_strength = other_denoise
                    ###############################################################################


#                 elif third_frame_image != "None":
#                     p.width = initial_width * 3
#                     img = Image.new("RGB", (initial_width*3, p.height))  # 创建一个新的img

#                     img.paste(p.init_images[0], (0, 0))  # 把刚刚画完的上一帧贴在左边
#                     # img.paste(p.init_images[0], (initial_width, 0))

#                     img.paste(loopback_image, (initial_width, 0))  # 要跑的图贴在中间
#                     if i == 1:  # 第一步的时候，把第三张图拿进去，后面这东西会一直保持
#                         last_image = p.init_images[0]
#                     img.paste(last_image, (initial_width*2, 0))
#                     p.init_images = [img]
#                     if color_correction_enabled:
#                         p.color_corrections = [processing.setup_color_correction(img)]

#                     ctrl_input_image = Image.new("RGB", (initial_width*3, p.height))
#                     ctrl_input_image.paste(Image.open(reference_imgs[i-1]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (0, 0))
#                     ctrl_input_image.paste(p.control_net_input_image, (initial_width, 0))
#                     ctrl_input_image.paste(Image.open(reference_imgs[last_image_index]).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (initial_width*2, 0))
#                     p.control_net_input_image = ctrl_input_image

#                     latent_mask = Image.new("RGB", (initial_width*3, p.height), "black")
#                     latent_draw = ImageDraw.Draw(latent_mask)
#                     latent_draw.rectangle((initial_width,0,initial_width*2,p.height), fill="white")
#                     p.image_mask = latent_mask
#                     p.denoising_strength = original_denoise


            else:
                # 第0步
                latent_mask = Image.new("RGB", (initial_width, p.height), "white")
                # p.latent_mask = latent_mask
                p.image_mask = latent_mask
                p.denoising_strength = first_denoise
                p.control_net_input_image = p.control_net_input_image.resize((initial_width, p.height))
                p.init_images = [Image.open(reference_imgs[0]).convert("RGB").resize((p.width, p.height), Image.ANTIALIAS)]
                # p.init_images = [Image.new("RGB", (initial_width, p.height), "black").resize((p.width, p.height), Image.ANTIALIAS)]
                # frames.append(p.control_net_input_image)

            # if opts.img2img_color_correction:
            #     p.color_corrections = initial_color_corrections
            # 这东西卵用没有（
            if append_interrogation != "None":
                p.prompt = original_prompt
                if append_interrogation == "CLIP":
                    p.prompt += shared.interrogator.interrogate(p.init_images[0])
                elif append_interrogation == "DeepBooru":
                    p.prompt += deepbooru.model.tag(p.init_images[0])

            if use_csv or use_txt:
                # p.prompt = original_prompt + prompt_list[i]
                p.prompt = prompt_list[i]

            # state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"
            # 开跑#########################################################################
            # 得到返回值
            processed = processing.process_images(p)

            # 固定种子
            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            # 拿出处理完的图片
            init_img = processed.images[0]
            if(i > 0):  # 如果不是第一步，则需要裁剪图片（把中间那个图片抠出来，作为下一次循环的初始图片）
                init_img = init_img.crop((initial_width*(num_of_front_frame), 0, initial_width*(num_of_front_frame+1), p.height))

            # 劫持原模型
            comments = {}
            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            # 获取图片信息
            info = processing.create_infotext(
                p,
                p.all_prompts,
                p.all_seeds,
                p.all_subseeds,
                comments,
                0,
                0)
            pnginfo = {}
            if info is not None:
                pnginfo['parameters'] = info

            params = ImageSaveParams(init_img, p, filename, pnginfo)
            before_image_saved_callback(params)
            fullfn_without_extension, extension = os.path.splitext(
                filename)

            info = params.pnginfo.get('parameters', None)

            def exif_bytes():
                return piexif.dump({
                    'Exif': {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(info or '', encoding='unicode')
                    },
                })

            # 保存图片
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if extension.lower() == '.png':
                pnginfo_data = PngImagePlugin.PngInfo()
                for k, v in params.pnginfo.items():
                    pnginfo_data.add_text(k, str(v))

                init_img.save(
                    os.path.join(
                        output_dir,
                        filename),
                    pnginfo=pnginfo_data)

            elif extension.lower() in ('.jpg', '.jpeg', '.webp'):
                init_img.save(os.path.join(output_dir, filename))

                if opts.enable_pnginfo and info is not None:
                    piexif.insert(
                        exif_bytes(), os.path.join(
                            output_dir, filename))
            else:
                init_img.save(os.path.join(output_dir, filename))


            # if third_frame_image != "None":  # 跑图时，根据需求修改这货，往last_image里面喂东西
            if True:  # 跑图时，根据需求修改这货，往last_image里面喂东西
                # 喂第一张图（只有在第0步才会喂）
                if True and i == 0:
                # if third_frame_image == "FirstGen" and i == 0:
                    last_image = init_img
                    last_image_index = 0
                # # 喂原图（只有在第0步才会喂）
                # elif third_frame_image == "OriginalImg" and i == 0:
                #     last_image = original_init_image[0]
                #     last_image_index = 0
                # 喂上一张（把处理完的喂进去）
                # elif third_frame_image == "Historical":
                #     last_image = processed.images[0].crop((0, 0, initial_width, p.height))
                #     last_image_index = (i-1)  # 这东西唯一的用处（输入历史图片，然后没了）

            # 把跑完的图，放进要跑的东西里面
            p.init_images = [init_img]
            # 删除缓存区第一张，把另一张塞进结尾
            if len(multiframepro_frames) >= (num_of_front_frame): 
                del (multiframepro_frames[0])
            multiframepro_frames += [init_img]
            
            if(freeze_seed):
                p.seed = processed.seed
            else:
                p.seed = processed.seed + 1
            # p.seed = processed.seed
            # 第一步时，则直接把刚跑完的图片放进跑图历史中
            if i == 0:
                history = init_img
            # history.append(processed.images[0])
            # frames.append(processed.images[0])

        # for循环一直跑到这里#######################################################

        # grid = images.image_grid(history, rows=1)
        # if opts.grid_save:
        #     images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

        # grids.append(grid)
        # # all_images += history + frames
        # all_images += history

        # p.seed = p.seed+1

        # if opts.return_grid:
        #     all_images = grids + all_images

        processed = Processed(p, [], initial_seed, initial_info)

        return processed
