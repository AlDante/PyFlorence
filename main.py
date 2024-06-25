'''
Patched version of the Florence example from HuggingFace to not use flash_attn. (Which
requires CUDA and doesn't really work on Apple).
Patch discussion: https://huggingface.co/microsoft/Florence-2-large-ft/discussions/4

Model example
https://huggingface.co/microsoft/Florence-2-large-ft
https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
'''

import copy
import os
import random
from unittest.mock import patch

import numpy as np

import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
MODEL = "microsoft/Florence-2-large-ft"

EBUDIR="/Users/david/Downloads/Bridge/EBU/"
EBUNAME="1946-09"

EBUPNGSUFFIX= ".png"
EBUPNGPATH= EBUDIR + EBUNAME + EBUPNGSUFFIX

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",
                    fill=color)
    # display(image)
    return image

def run_example(prompt: str, image: Image):

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

    return parsed_answer

'''

url = "http://ecx.images-amazon.com/images/I/51UUzBDAMsL.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
prompt = "<OCR>"
run_example(prompt)
(Outputs {'<OCR>': 'CUDAFOR ENGINEERSAn Introduction to High-PerformanceParallel ComputingDUANE STORTIMETE YURTOGLU'})

task_prompt = '<OCR_WITH_REGION>'
results = run_example(task_prompt)
print(results)
# ocr results format
# {'OCR_WITH_REGION': {'quad_boxes': [[x1, y1, x2, y2, x3, y3, x4, y4], ...], 'labels': ['text1', ...]}}
# {'<OCR_WITH_REGION>': {'quad_boxes': [[167.0435028076172, 50.25, 375.7974853515625, 50.25, 375.7974853515625, 114.75, 167.0435028076172, 114.75], [144.8784942626953, 120.75, 375.7974853515625, 120.75, 375.7974853515625, 149.25, 144.8784942626953, 149.25], [115.86249542236328, 165.25, 376.6034851074219, 166.25, 376.6034851074219, 184.25, 115.86249542236328, 183.25], [239.9864959716797, 184.25, 376.6034851074219, 186.25, 376.6034851074219, 204.25, 239.9864959716797, 202.25], [266.1814880371094, 441.25, 376.6034851074219, 441.25, 376.6034851074219, 456.25, 266.1814880371094, 456.25], [252.0764923095703, 460.25, 376.6034851074219, 460.25, 376.6034851074219, 475.25, 252.0764923095703, 475.25]], 'labels': ['</s>CUDA', 'FOR ENGINEERS', 'An Introduction to High-Performance', 'Parallel Computing', 'DUANE STORTI', 'METE YURTOGLU']}}

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",      
                    fill=color)
    display(image)
output_image = copy.deepcopy(image)
draw_ocr_bboxes(output_image, results['<OCR_WITH_REGION>'])  

'''

if __name__ == "__main__":
    # Disable parallelism to avoid this error
    # "huggingface/tokenizers: The current process just got forked, after parallelism has already been used."
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    # image = Image.open(requests.get(URL, stream=True).raw)
    image = Image.open(EBUPNGPATH)
    #image.show()


    '''
    Huggingface caches models. Default directory is here: ~/.cache/huggingface/hub
    https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory
    '''
    prompt = "<OCR_WITH_REGION>"
    results = run_example(prompt, image)

    output_image = copy.deepcopy(image)
    labelled_image = draw_ocr_bboxes(output_image, results['<OCR_WITH_REGION>'])
    image.show()

