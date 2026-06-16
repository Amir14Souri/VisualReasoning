import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenClient:

    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            padding_side="left",
        )

    def ask(self, images, prompts):
        single = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            prompts = [prompts]
            single = True

        for image, prompt in zip(images, prompts):
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                for image, prompt in zip(images, prompts)
            ]

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]

        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=8)

        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        answers = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return answers[0].strip() if single else [ans.strip() for ans in answers]
