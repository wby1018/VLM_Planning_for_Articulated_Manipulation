import os
import sys
import base64
from openai import OpenAI


def image_to_data_url(image_path: str) -> str:
    """
    把本地图片转成 base64 data URL，供 Responses API 的 input_image 使用。
    """
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(ext, "application/octet-stream")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64}"


def ask_text(client: OpenAI, prompt: str) -> str:
    """
    发送纯文本
    """
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )
    return response.output_text


def ask_image_and_text(client: OpenAI, prompt: str, image_path: str) -> str:
    """
    发送 图片 + 文本
    """
    image_data_url = image_to_data_url(image_path)

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )
    return response.output_text


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误：未检测到 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    while True:
        print("\n==== ChatGPT API Demo ====")
        print("1. 发送纯文本")
        print("2. 发送图片 + 文本")
        print("3. 退出")
        choice = input("请选择: ").strip()

        if choice == "1":
            prompt = input("请输入文本问题: ").strip()
            if not prompt:
                print("输入不能为空")
                continue
            try:
                result = ask_text(client, prompt)
                print("\n--- 模型回复 ---")
                print(result)
            except Exception as e:
                print(f"请求失败: {e}")

        elif choice == "2":
            image_path = input("请输入本地图片路径: ").strip()
            prompt = input("请输入关于这张图的问题: ").strip()

            if not os.path.exists(image_path):
                print("图片路径不存在")
                continue
            if not prompt:
                print("问题不能为空")
                continue

            try:
                result = ask_image_and_text(client, prompt, image_path)
                print("\n--- 模型回复 ---")
                print(result)
            except Exception as e:
                print(f"请求失败: {e}")

        elif choice == "3":
            print("退出")
            break
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()