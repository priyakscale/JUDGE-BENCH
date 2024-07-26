import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import torch
from tqdm import tqdm
import time

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class HFModel:
    def __init__(self, name, new_tokens, max_length=None, max_gen=None, use_max_gen=True, rank=0) -> None:
        device = torch.device(f'cuda:{rank}')

        if os.path.exists(name):
            # Load model and tokenizer from local path
            self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to("cuda")
            self.local_model = True
        else:
            self.model = pipeline(
                "text-generation",
                model=name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True if "OLMo" in name else False,
            )
            self.tokenizer = self.model.tokenizer
            self.local_model = False
            if not self.model.tokenizer.pad_token_id:
                self.model.tokenizer.pad_token_id = (
                    self.model.model.config.eos_token_id
                )
        self.n_tokens = new_tokens
        self.max_length = max_length
        self.max_gen = max_gen
        self.use_max_gen = use_max_gen
        self.device = device

    def process(self, example, system_prompt=None):
        user_message = {"role": "user", "content": example}
        if system_prompt is None:
            messages = [user_message]
        else:
            system_message = {"role": "system", "content": system_prompt}
            messages = [system_message, user_message]
        return self.model.tokenizer.apply_chat_template(
            messages, tokenize=False
        )

    def generate_responses(self, dataset, batch_size, system_prompt=None):
        # dataset = list(map(lambda x: self.process(x, system_prompt), dataset))
        responses = []
        if self.local_model: #llama3 models
            # Generate responses for local model
            for batch_start in tqdm(range(0, len(dataset), batch_size), total=len(dataset)//batch_size):
                batch = dataset[batch_start:batch_start + batch_size]
                for prompt in batch:
                    tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                    if len(tokenized_prompt) > self.max_length:
                        half = int(self.max_length / 2)
                        prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                                 self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    user_message = {"role": "user", "content": prompt}
                    if system_prompt is None:
                        messages = [user_message]
                    else:
                        system_message = {"role": "system", "content": system_prompt}
                        messages = [system_message, user_message]
                    # messages = [
                    #     {"role": "user", "content": prompt},
                    # ]
                    input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
                    context_length = input.shape[-1]
                    
                    if self.use_max_gen:
                        output = self.model.generate(
                            input,
                            max_new_tokens=self.max_gen,
                            num_beams=1,
                            do_sample=False,
                            temperature=1.0,
                        )[0]
                    else:
                        output = self.model.generate(
                            input,
                            num_beams=1,
                            do_sample=False,
                            temperature=1.0,
                        )[0]
                    generated_text = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
                    responses.append(generated_text)
        else:
            dataset = list(map(lambda x: self.process(x, system_prompt), dataset))
            for response in tqdm(
                self.model(
                    dataset,
                    batch_size=batch_size,
                    max_new_tokens=self.n_tokens,
                    do_sample=False,
                    num_beams=1,
                    return_full_text=False,
                ),
                total=len(dataset),
            ):
                responses.append(response[0]["generated_text"])
        return responses


class APIModel:
    name: str
    new_tokens: int
    family: str

    def __init__(self, name: str, new_tokens: int) -> None:
        self.name = name
        self.new_tokens = new_tokens
        if "gpt" in name:
            self.family = "OpenAI"
            self.client = OpenAI()
        elif "claude" in name:
            self.family = "Anthropic"
            self.client = Anthropic()
        elif "gemini" in name:
            self.family = "Google"
            self.client = genai.GenerativeModel(name)
        else:
            raise ValueError(
                f'Model "{name} is not a valid ClosedModel (gpt, claude, gemini)'
            )

    def generate_responses(self, dataset, batch_size, system_prompt=None):
        responses = []
        for query in tqdm(dataset):
            user_message = {"role": "user", "content": query}

            if system_prompt is None:
                messages = [user_message]
            else:
                system_message = {"role": "system", "content": system_prompt}
                messages = [system_message, user_message]

            if self.family == "OpenAI":
                response = (
                    self.client.chat.completions.create(
                        model=self.name,
                        messages=messages,
                        max_tokens=self.new_tokens,
                        temperature=0,
                    )
                    .choices[0]
                    .message.content
                )
                responses.append(response)

            elif self.family == "Anthropic":
                response = (
                    self.client.messages.create(
                        model=self.name,
                        messages=messages,
                        max_tokens=self.new_tokens,
                        temperature=0,
                    )
                    .content[0]
                    .text
                )
                responses.append(response)
                # time.sleep(3) # in case of rate limiting resort to 15 RPM

            elif self.family == "Google":
                response = self.client.generate_content(
                    query,
                    generation_config={
                        "max_output_tokens": self.new_tokens,
                        "temperature": 0,
                    },
                    safety_settings={  # https://ai.google.dev/gemini-api/docs/safety-settings (e.g., usecase: QAGS dataset)
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    },
                )
                try:
                    responses.append(response.text)
                except ValueError:
                    print(f"FAILED REQUEST: {messages}\nRESPONSE: {response}")
                    responses.append("")  # to keep all valid responses

            else:
                raise ValueError(
                    f'Family "{self.family} is not a valid family'
                )

        return responses
