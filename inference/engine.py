"""
MMSearch-R1 inference engine.

Loads lmms-lab/MMSearch-R1-7B and runs multi-turn tool-augmented generation.
Supports two tools:
  - <search><img>URL</img>URL</search>  → call_image_search
  - <text_search>query</text_search>   → call_text_search

Tool results are injected as <information>…</information> and generation continues.
Max turns controlled by image_search_limit and text_search_limit.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from search_tools import call_image_search, call_text_search

MODEL_ID = os.environ.get("MMSEARCH_MODEL", "lmms-lab/MMSearch-R1-7B")

# XML tag patterns
_PAT_IMAGE_SEARCH = re.compile(
    r"<search>(?:<img>(.*?)</img>)?(.*?)</search>", re.DOTALL
)
_PAT_TEXT_SEARCH = re.compile(r"<text_search>(.*?)</text_search>", re.DOTALL)
_PAT_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    image_search_limit: int = 1
    text_search_limit: int = 2
    max_turns: int = 3


@dataclass
class TurnState:
    image_searches_used: int = 0
    text_searches_used: int = 0
    turns: int = 0
    tool_calls: list = field(default_factory=list)


class MMSearchEngine:
    def __init__(self, model_id: str = MODEL_ID, device: str = "auto"):
        print(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded.")

    def _build_messages(
        self,
        question: str,
        image: Optional[Image.Image],
        conversation_history: list[dict],
    ) -> list[dict]:
        if not conversation_history:
            content = []
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": question})
            return [{"role": "user", "content": content}]
        return conversation_history

    def _generate(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> tuple[str, list]:
        """Run one forward pass, return (text, images_in_context)."""
        images = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        images.append(part["image"])

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=images if images else None,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                do_sample=config.do_sample,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True), images

    def _execute_tool(
        self,
        response_text: str,
        state: TurnState,
        config: GenerationConfig,
    ) -> tuple[Optional[str], list[Image.Image]]:
        """
        Parse tool call from response text, execute it, return (info_text, new_images).
        Returns (None, []) if no tool call found or limits exhausted.
        """
        # Try image search first
        img_match = _PAT_IMAGE_SEARCH.search(response_text)
        if img_match and state.image_searches_used < config.image_search_limit:
            img_url = (img_match.group(1) or img_match.group(2) or "").strip()
            if img_url:
                state.image_searches_used += 1
                state.tool_calls.append({"type": "image_search", "url": img_url})
                info_text, thumbnails, meta = call_image_search(img_url)
                return info_text, thumbnails

        # Try text search
        txt_match = _PAT_TEXT_SEARCH.search(response_text)
        if txt_match and state.text_searches_used < config.text_search_limit:
            query = txt_match.group(1).strip()
            state.text_searches_used += 1
            state.tool_calls.append({"type": "text_search", "query": query})
            info_text, meta = call_text_search(query)
            return info_text, []

        return None, []

    def run(
        self,
        question: str,
        image: Optional[Image.Image] = None,
        config: Optional[GenerationConfig] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Multi-turn inference with tool use.

        Returns:
            {
                "answer": str,
                "full_response": str,
                "tool_calls": list,
                "turns": int,
            }
        """
        if config is None:
            config = GenerationConfig()

        state = TurnState()
        messages = self._build_messages(question, image, [])
        full_response_parts = []

        while state.turns < config.max_turns:
            response_text, _ = self._generate(messages, config)
            full_response_parts.append(response_text)
            state.turns += 1

            if verbose:
                print(f"\n--- Turn {state.turns} ---\n{response_text}\n")

            # Check for final answer
            answer_match = _PAT_ANSWER.search(response_text)
            if answer_match:
                return {
                    "answer": answer_match.group(1).strip(),
                    "full_response": "\n".join(full_response_parts),
                    "tool_calls": state.tool_calls,
                    "turns": state.turns,
                }

            # Try to execute a tool
            info_text, new_images = self._execute_tool(response_text, state, config)
            if info_text is None:
                # No tool call and no answer — treat last generation as answer
                break

            # Append assistant turn + tool result as user turn
            messages.append({"role": "assistant", "content": response_text})
            info_content: list[dict] = []
            for img in new_images:
                info_content.append({"type": "image", "image": img})
            info_content.append(
                {"type": "text", "text": f"<information>\n{info_text}\n</information>"}
            )
            messages.append({"role": "user", "content": info_content})

        # Exhausted turns — extract best answer from last response
        last = full_response_parts[-1] if full_response_parts else ""
        answer_match = _PAT_ANSWER.search(last)
        return {
            "answer": answer_match.group(1).strip() if answer_match else last.strip(),
            "full_response": "\n".join(full_response_parts),
            "tool_calls": state.tool_calls,
            "turns": state.turns,
        }
