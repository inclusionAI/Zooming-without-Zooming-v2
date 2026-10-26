import re, os, threading
from .model_utils import LLM, call_api, truncate_images
from .model_utils import encode_image_base64
from functools import partial
import openai
from PIL import Image
import time
import io
import requests
import base64


class VLLMModel(LLM):
    def __init__(
            self,
            model_name,
            temperature=0.9,
            top_p=0.9,
            max_length=32768,
            generation_max_length=2048,
            generation_min_length=0,
            do_sample=True,
            stop_newline=False,
            use_chat_template=True,
            **kwargs,
    ):
        super().__init__(
            model_name,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            generation_max_length=generation_max_length,
            generation_min_length=generation_min_length,
            do_sample=do_sample,
            stop_newline=stop_newline,
            use_chat_template=use_chat_template,
        )
        self.api_model = True
        self.max_image_num = kwargs.get("max_image_num", None)
        self.api_sleep = kwargs.get("api_sleep", 0.0)

        raw_urls = os.environ.get("VLLM_BASE_URLS") or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.base_urls = [url.strip().rstrip("/") for url in raw_urls.split(",") if url.strip()]
        if not self.base_urls:
            raise ValueError("VLLM_BASE_URLS must contain at least one local endpoint")
        print(f"Using vllm endpoint(s) {self.base_urls} for evaluation")
        # Portable authentication; retain generation and parser behavior.
        self.models = [openai.OpenAI(base_url=url, api_key=os.environ.get("MAIN_API_KEY", "EMPTY")) for url in self.base_urls]
        self.model = self.models[0]  # backward-compatible handle
        self._client_lock = threading.Lock()
        self._client_index = 0

        self.model_name = model_name
        self.processor = None

    def _next_client(self):
        with self._client_lock:
            client = self.models[self._client_index % len(self.models)]
            self._client_index += 1
        return client

    def format_chat(self, text, image_list, system_prompt, is_url_image=False):
        content = re.split(r'(<image>)', text)
        image_idx, new_content = 0, []
        for c in content:
            if c == "<image>":
                current_image = image_list[image_idx]
                if isinstance(current_image, bytes):
                    img_file = io.BytesIO(current_image)

                    with Image.open(img_file) as img:
                        image_format = img.format.lower()

                    base64_str = base64.b64encode(current_image).decode('utf-8')
                    url_str = f"data:image/{image_format};base64,{base64_str}"
                else:
                    url_str = f"file://{current_image}"
                new_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url_str
                    },
                })
                image_idx += 1
            else:
                new_content.append({
                    "type": "text",
                    "text": c
                })
        assert image_idx == len(image_list)
        messages = [{"role": "user", "content": new_content}]
        
        if system_prompt:
            messages.append({"role": "assistant", "content": system_prompt})

        return messages

    def prepare_inputs(self, test_item, data):
        text = data["user_template"].format(**test_item)
        image_list = test_item["image_list"] # use local image path
        
        messages = self.format_chat(text, image_list, data["system_template"])

        return {"messages": messages}

    def generate(self, inputs=None, prompt=None, **kwargs):
        save_prompt = [c["text"] if c["type"] == "text" else "<image>" for c in inputs["messages"][0]["content"]]
        save_prompt = "".join(save_prompt)
        if len(save_prompt) > 6000:
            save_prompt = save_prompt[:2000] + " <skip> " + save_prompt[-2000:]

        func = partial(
            self._next_client().chat.completions.create,
            model=self.model_name,
            messages=inputs["messages"],
            max_tokens=self.generation_max_length,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p,
            # Explicit for both Base and trained chat templates.  Some templates
            # enable reasoning when this key is absent.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **kwargs,
        )

        start_time = time.time()
        try:
            output = call_api(func, limit=5, pause=5)
        except Exception:
            # Transport failure is not an empty model answer. Let the native
            # writer retain its completed prefix and resume the failed item.
            raise RuntimeError("MMLongBench API request failed; no result cached") from None

        end_time = time.time()

        print(f"example finished, used {end_time - start_time} secs, sleep {max(self.api_sleep - (end_time - start_time), 0.0)} secs")
        time.sleep(max(self.api_sleep - (end_time - start_time), 0.0))

        try:
            choice = output.choices[0]
            content = choice.message.content
            input_len = output.usage.prompt_tokens
            output_len = output.usage.completion_tokens
            if content is not None and not isinstance(content, str):
                raise ValueError('Invalid response content')
            if type(input_len) is not int or type(output_len) is not int or min(input_len, output_len) < 0:
                raise ValueError('Missing/invalid native token accounting')
        except (AttributeError, IndexError, TypeError, ValueError):
            raise RuntimeError("Malformed MMLongBench API response; no result cached") from None
        return {
            "output": content or "",
            "input_len": input_len,
            "output_len": output_len,
            "input_text": save_prompt,
            "finish_reason": getattr(choice, 'finish_reason', None),
            "api_status": "ok",
        }
