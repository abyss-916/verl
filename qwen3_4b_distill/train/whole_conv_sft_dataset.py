"""整段渲染的单轮 SFT 数据集，修正 Qwen3 `<think>` 在 verl 逐轮分词中被剥除的问题。

对整段对话 `apply_chat_template`（保留 `<think>`）得到 input_ids；
再对 prompt 部分（`messages[:-1]` + 生成提示）单独渲染，确定 loss 边界，仅对 assistant 回复计 loss。
prompt 须是 full 的前缀，否则 fail-loud。

- 仅支持以 assistant 结尾的对话。
- 不改 verl 核心；通过 `data.custom_cls.path=<本文件> data.custom_cls.name=WholeConvSFTDataset` 挂载。
- padding/truncation 分支与父类 `__getitem__` 保持一致，返回相同结构。
"""

import torch
import torch.nn.functional as F

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class WholeConvSFTDataset(MultiTurnSFTDataset):
    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None

        assert len(messages) >= 2 and messages[-1]["role"] == "assistant", (
            f"WholeConvSFTDataset 仅支持以 assistant 结尾的对话，得到 roles="
            f"{[m.get('role') for m in messages]}"
        )

        # enable_thinking：仅当明确为 bool 才透传，否则用模板默认
        et = self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        kw = {**self.apply_chat_template_kwargs}
        if isinstance(et, bool):
            kw["enable_thinking"] = et

        # 整段渲染（保留 <think>…</think>）
        full = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=False, tokenize=True, return_tensors="pt", **kw
        )[0]
        # prompt = 到 assistant 生成提示为止（messages[:-1] + 生成提示），确定 loss 边界
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1], tools=tools, add_generation_prompt=True, tokenize=True, return_tensors="pt", **kw
        )[0]
        plen = int(prompt.shape[0])

        # 校验：prompt 须是 full 的前缀，否则 loss 边界对不齐，fail-loud
        if plen >= int(full.shape[0]) or not torch.equal(full[:plen], prompt):
            raise ValueError(
                f"[WholeConvSFTDataset] prompt 不是 full 的前缀，loss 边界无法对齐："
                f"plen={plen}, full_len={int(full.shape[0])}；检查 chat 模板一致性。"
            )

        input_ids = full
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[plen:] = 1  # 只训 assistant 回复（含 <think>…</think>解答）
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        # ---- padding / truncation：与 MultiTurnSFTDataset.__getitem__ 保持一致 ----
        sequence_length = input_ids.shape[0]
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                pad = self.max_length - sequence_length
                input_ids = torch.cat((input_ids, torch.full((pad,), pad_token_id, dtype=input_ids.dtype)))
                attention_mask = torch.cat((attention_mask, torch.zeros((pad,), dtype=attention_mask.dtype)))
                loss_mask = torch.cat((loss_mask, torch.zeros((pad,), dtype=loss_mask.dtype)))
                position_ids = F.pad(position_ids, (0, pad), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids, attention_mask, loss_mask, position_ids = (
                        input_ids[-self.max_length:], attention_mask[-self.max_length:],
                        loss_mask[-self.max_length:], position_ids[..., -self.max_length:])
                elif self.truncation == "right":
                    input_ids, attention_mask, loss_mask, position_ids = (
                        input_ids[:self.max_length], attention_mask[:self.max_length],
                        loss_mask[:self.max_length], position_ids[..., :self.max_length])
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")
            return {"input_ids": input_ids, "attention_mask": attention_mask,
                    "position_ids": position_ids, "loss_mask": loss_mask}
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if sequence_length > self.max_length and self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
                position_ids = position_ids[..., :self.max_length]
            return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
