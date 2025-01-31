import os
from openai import OpenAI
from http import HTTPStatus


class DeepSeekCaller:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    ["deepseek-chat", "deepseek-reasoner"],
                    {"default": "deepseek-chat"},
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter the system prompt here.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter the user prompt here.",
                    },
                ),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
            }
        }

    FUNCTION = "call_model"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)

    CATEGORY = "deepseek"

    def call_model(self, model, system_prompt, user_prompt, max_tokens, temperature):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        if response.status_code == HTTPStatus.OK:
            return response.choices[0].message.content
        else:
            if response.status_code == HTTPStatus.BAD_REQUEST:
                raise Exception("请求体格式错误。请根据错误信息提示修改请求体")
            elif response.status_code == HTTPStatus.UNAUTHORIZED:
                raise Exception(
                    "API key 错误，认证失败。请检查您的 API key 是否正确，如没有 API key，请先创建 API key"
                )
            elif response.status_code == HTTPStatus.PAYMENT_REQUIRED:
                raise Exception(
                    "账号余额不足。请确认账户余额，并前往 充值 页面进行充值"
                )
            elif response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                raise Exception("请求体参数错误。请根据错误信息提示修改相关参数")
            elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                raise Exception(
                    "请求速率（TPM 或 RPM）达到上限。请合理规划您的请求速率。"
                )
            elif response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
                raise Exception(
                    "服务器内部故障。请等待后重试。若问题一直存在，请联系我们解决"
                )
            elif response.status_code == HTTPStatus.SERVICE_UNAVAILABLE:
                raise Exception("服务器负载过高。请稍后重试您的请求")
