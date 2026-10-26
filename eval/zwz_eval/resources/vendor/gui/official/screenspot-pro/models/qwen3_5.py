import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.generation import GenerationConfig
import json
import base64
import re
import os
import hashlib
from io import BytesIO
from PIL import Image
import sys
from pathlib import Path
# Also support a source checkout launched by absolute script path, without -e.
sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from zwz_eval.gui_grounding import legacy_point, parse_point

from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_qwen3_5_prompt_msg(image, instruction, screen_width, screen_height):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": 
                    """You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.\n\n# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>""".replace(r"{screen_width}", str(screen_width)).replace(r"{screen_height}", str(screen_height))
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # "min_pixels": 3136,
                    # "max_pixels": 12845056,
                    "image_url": {
                        "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                    }
                },
                {
                    "type": "text",
                    "text": instruction
                }
            ]
        }
    ]



GUIDED_PROMPT = """<|im_start|>You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{{instruction}}<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ["""


class Qwen3_5Model():
    def load_model(self, model_name_or_path="Qwen/Qwen3.5-4B"):
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_impl = os.environ.get("QWEN3_5_ATTN_IMPL", "sdpa")
        self.max_pixels = int(os.environ.get("QWEN3_5_MAX_PIXELS", "8294400"))
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, 
            device_map=None,
            torch_dtype=model_dtype,
            attn_implementation=attn_impl,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = self.model.generation_config.to_dict()
        # self.set_generation_config(
        #     max_length=2048,
        #     do_sample=False,
        #     temperature=0.0
        # )
        self.set_generation_config(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=4096,
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."
        print("Original image size: {}x{}".format(image.width, image.height))

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=8,
            min_pixels=32 * 32,
            # max_pixels=self.processor.image_processor.max_pixels,
            max_pixels=self.max_pixels,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        messages = get_qwen3_5_prompt_msg(image, instruction, resized_width, resized_height)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if os.environ.get("QWEN3_5_ENABLE_THINKING") is not None:
            template_kwargs["enable_thinking"] = os.environ["QWEN3_5_ENABLE_THINKING"].lower() not in ("0", "false", "no")

        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, **template_kwargs
        )
        
        # guided_text = """<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ["""
        # text_input += guided_text
        
        inputs = self.processor(
            text=[text_input],
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        print("Len: ", len(inputs.input_ids[0]))
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        # response = guided_text + response
        print(response)

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        try:
            action = json.loads(response.split('<tool_call>\n')[-1].split('\n</tool_call>')[-2])
            coordinates = action['arguments']['coordinate']
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError("Wrong output format")
            print(point_x, point_y)
            result_dict["point"] = [point_x / 1000, point_y / 1000]  # Normalize predicted coordinates. Qwen3-VL uses relative coordinates in range [0, 1000]
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()


class Qwen3_5_VLLM_Model():
    def __init__(self):
        # Check if the current process is daemonic.
        from multiprocessing import current_process
        process = current_process()
        if process.daemon:
            print("Latest vllm versions spawns children processes, therefore can not be started in a daemon process. Are you using multiprocess.Pool? Try multiprocess.Process instead.")

    def load_model(self, model_name_or_path="Qwen/Qwen3.5-4B"):
        from vllm import LLM
        self.max_pixels = int(os.environ.get("QWEN3_5_MAX_PIXELS", "8294400"))
        self.model = LLM(
            model_name_or_path,
            gpu_memory_utilization=0.99,
            tensor_parallel_size=int(os.environ.get("QWEN3_5_TP", "1")),
            max_num_seqs=16,
            # 16 independent replicas otherwise race on a shared Inductor cache.
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 32 * 32,
                "max_pixels": self.max_pixels,
            },
        )

    def set_generation_config(self, **kwargs):
        pass

    def close(self):
        """Release this replica's engine before the next GUI split starts."""
        model = getattr(self, 'model', None)
        self.model = None
        if model is None:
            return
        shutdown = getattr(model, 'shutdown', None)
        if callable(shutdown):
            shutdown()
            return
        engine = getattr(model, 'llm_engine', None)
        core = getattr(engine, 'engine_core', None)
        shutdown = getattr(core, 'shutdown', None)
        if not callable(shutdown):
            raise RuntimeError('This vLLM version has no supported engine shutdown method')
        shutdown()

    def ground_only_positive(self, instruction, image):
        from vllm import SamplingParams
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=32,
            min_pixels=32 * 32,
            # max_pixels=self.processor.image_processor.max_pixels,
            max_pixels=self.max_pixels,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        # Build the official tool-use task prompt through the checkpoint's own
        # chat template (the same template used by the native profile), rather than forcing
        # the legacy hard-coded Qwen2.5 special-token template.
        messages = get_qwen3_5_prompt_msg(resized_image, instruction, resized_width, resized_height)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if os.environ.get("QWEN3_5_ENABLE_THINKING") is not None:
            template_kwargs["enable_thinking"] = os.environ["QWEN3_5_ENABLE_THINKING"].lower() not in ("0", "false", "no")
        prompt = self.model.get_tokenizer().apply_chat_template(
            messages, **template_kwargs
        )
        inputs = {"prompt": prompt, "multi_modal_data": {"image": resized_image}}

        generated = self.model.generate(inputs, sampling_params=SamplingParams(temperature=0.0, max_tokens=int(os.environ.get("QWEN3_5_MAX_NEW_TOKENS", "8192"))))

        completion = generated[0].outputs[0]
        response = completion.text.strip()

        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Preserve the native XML score exactly, but make formatting-only failures
        # visible. Never substitute the diagnostic point into the primary one.
        result_dict["point"] = legacy_point(response)
        result_dict["audit"] = {
            "parser": "native_xml-exact-newlines", "coordinate_space": "normalized_1000",
            "original_size": list(image.size), "sent_size": list(resized_image.size),
            "max_pixels": self.max_pixels,
            "max_tokens": int(os.environ.get("QWEN3_5_MAX_NEW_TOKENS", "8192")),
            "enable_thinking": template_kwargs.get("enable_thinking"),
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
            "output_tokens": len(getattr(completion, "token_ids", [])),
            "rendered_prompt": prompt,
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "system_prompt": messages[0]["content"][0]["text"],
            "diagnostic_tool_call_point": parse_point(response, parser="tool_call",
                coordinate_space="normalized_1000", sent_size=resized_image.size),
        }
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()
