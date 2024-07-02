import argparse

import torch
import clip
import numpy as np

from models import create_model
from utils.options import dict_to_nonedict, parse
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

def load_image(file_path):
    downsample_factor = 2
    with open(file_path, 'rb') as f:
        image = Image.open(f)
        width, height = image.size
        width = width // downsample_factor
        height = height // downsample_factor
        image = image.resize(
            size=(width, height), resample=Image.NEAREST)
        image = np.array(image).transpose(2, 0, 1)
    return image.astype(np.float32)

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./configs/region_gen.yml', help='Path to option YAML file.')
    parser.add_argument('--img_path', type=str, help='Path to the fashion image.', required=True)
    parser.add_argument('--output_path', type=str, help='Saving path to the edited image.', required=True)
    parser.add_argument('--text_prompt', type=str, help='The editing text prompt.', required=True)
    parser.add_argument('--erlm_model_path', type=str, help='Path to ERLM model.', required=True)
    parser.add_argument('--texfit_model_path', type=str, help='Path to TexFit model.', required=True)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)
    opt['pretrained_model_path'] = args.erlm_model_path

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)
    model = create_model(opt)
    model.load_network()
    model.encoder.eval()
    model.decoder.eval()

    img_path = args.img_path
    text = args.text_prompt

    img = load_image(img_path)
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0)

    img = img.to(model.device)
    text_inputs = torch.cat([clip.tokenize(text)]).to(model.device)

    with torch.no_grad():
        text_embedding = model.clip.encode_text(text_inputs)
        text_enc = model.encoder(img, text_embedding)
        seg_logits = model.decoder(text_enc)
    seg_pred = seg_logits.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()[0]
    seg_img = Image.fromarray(np.uint8(seg_pred * 255))

    img = Image.open(img_path).convert("RGB").resize((256, 512))

    # Load pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.texfit_model_path, revision="fp16",
                                                          torch_dtype=torch.float16,
                                                          safety_checker=None,
                                                          requires_safety_checker=False).to("cuda")
    prompt = [text]
    generator = torch.Generator("cuda").manual_seed(2023)
    images = pipe(
        height=512,
        width=256,
        prompt=prompt,
        image=img,
        mask_image=seg_img,
        num_inference_steps=50,
        generator=generator,
        ).images

    final_img = Image.composite(images[0], img, seg_img)
    final_img.save(f'{args.output_path}')
    print('Saved edited result to', args.output_path)

if __name__ == '__main__':
    main()
