# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
# pathlib.Path用于处理文件路径，
# typing库中的类用于类型注解。

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

# 这三个类都是Python的`TypedDict`类型，它们用于定义字典的键和对应值的类型，以提供更好的类型检查和代码补全。
# 这行代码定义了一个类型别名Role，表示角色的类型，可以是"system"、"user"或"assistant"。
Role = Literal["system", "user", "assistant"]


# 这个类定义了消息的数据类型，包括角色（role）和内容（content）。
class Message(TypedDict):
    role: Role
    content: str

# 这个类定义了文本生成的预测结果的数据类型，
# 包括生成的文本（generation）、生成的token（tokens）和每个token的log概率（logprobs）。tokens和logprobs是可选的。
class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


# 这个类定义了聊天对话生成的预测结果的数据类型，
# 包括生成的消息（generation）、生成的token（tokens）和每个token的log概率（logprobs）。tokens和logprobs是可选的。
class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required

"""
`TypedDict`是Python的类型注解系统的一部分，它允许你为字典的键值对定义具体的类型。这在Python的静态类型检查中非常有用，
因为它可以帮助你确保你的代码在处理字典时，键和值都是你期望的类型。在没有`TypedDict`的情况下，你可以使用`Dict`类型注解
来指定字典的键和值的类型，但是这只能指定键和值的通用类型，不能为特定的键指定特定的类型。例如，你可以指定一个字典的键是
字符串，值是整数，但是你不能指定字典的"age"键的值是整数，"name"键的值是字符串。`TypedDict`就是为了解决这个问题而设计
的。通过`TypedDict`，你可以为字典的每个键指定一个具体的类型。例如，你可以定义一个`Person`类型，它有一个"age"键，值是
整数，有一个"name"键，值是字符串。这对于提高代码的可读性和可维护性非常有帮助，因为它可以让你清楚地知道每个字典应该有哪
些键，以及这些键的值应该是什么类型。此外，如果你使用了静态类型检查工具，如mypy，那么它还可以帮助你在运行代码之前就发现
类型错误。
"""

# 这行代码定义了一个类型别名Dialog，表示对话的类型，是一个消息的列表。
Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

"""
B_INST 和 E_INST：这两个常量被用作指令的开始和结束标记。在这个上下文中，"指令"可能是指用户给模型的指令，或者是模型给用户的回复。
这些标记可以帮助模型识别出文本中的指令部分。
B_SYS 和 E_SYS：这两个常量被用作系统提示的开始和结束标记。
系统提示是一种特殊的指令，通常用于设置对话的上下文，或者给模型提供一些特定的指示。这些标记可以帮助模型识别出文本中的系统提示部分。
"""

class Llama:
    # 这是一个静态方法，用于构建Llama类的实例。它接收一些参数，包括模型的checkpoint目录、tokenizer的路径、最大序列长度、最大
    # 批量大小和模型并行化的大小。
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        # 这部分代码用于初始化分布式环境和模型并行化。如果分布式环境还没有初始化，就使用torch.distributed.init_process_group
        # 方法初始化。如果模型并行化还没有初始化，就使用initialize_model_parallel方法初始化。
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        # 这部分代码获取当前进程的本地排名（local rank），然后设置CUDA设备。
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # 这行代码设置了随机数生成器的种子，以确保所有进程生成的随机数是一致的。
        # seed must be the same in all processes
        torch.manual_seed(seed)

        # 这部分代码将标准输出重定向到/dev/null，以避免在多进程环境中产生冗余的输出。
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        # 这部分代码加载模型的checkpoint。首先，它获取checkpoint目录中的所有.pth文件，然后检查是否有checkpoint文件，并且
        # checkpoint的数量是否与模型并行化的大小一致。然后，它根据当前进程的模型并行化排名选择一个checkpoint文件，然后加载
        # 这个checkpoint。最后，它读取并解析params.json文件，获取模型的参数。
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # 这部分代码创建模型和tokenizer。首先，它创建一个ModelArgs实例，然后创建一个Tokenizer实例。然后，它设置模型参数的词汇
        # 表大小为tokenizer的词汇表大小，然后设置默认的tensor类型为半精度浮点数（HalfTensor）。然后，它创建一个Transformer模型
        # ，并加载checkpoint。最后，它打印出加载模型所花费的时间。
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.
        根据给定的提示生成文本

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            prompt_tokens：一个二维列表，包含了一批提示的token。每个提示都是一个整数列表，表示提示的token。
            max_gen_len：一个整数，表示生成的文本的最大长度。
            temperature：一个浮点数，用于控制生成的随机性。值越大，生成的文本越随机；值越小，生成的文本越确定性。
            top_p：一个浮点数，用于控制生成的多样性。只有排名在前top_p的token会被考虑生成。
            logprobs：一个布尔值，如果为True，则返回每个token的对数概率。
            echo：一个布尔值，如果为True，则在生成的文本中包含提示。

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        # 这部分代码获取了模型的参数，并检查了批量大小是否超过了最大批量大小
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # 这部分代码计算了提示的最小和最大长度，并确保最大长度不超过最大序列长度。然后，它计算了生成的文本的总长度，这个长度是
        # 最大序列长度和最大生成长度加上最大提示长度之间的较小值
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # 这部分代码创建了一个全是填充token的tensor，然后将提示的token复制到这个tensor的前面部分。如果logprobs为True，那么
        # 它还会创建一个全是0的tensor，用于存储token的对数概率。
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        # 接下来的部分是生成文本的主循环。在每次迭代中，它会生成一个新的token，并将其添加到已生成的token的末尾。
        # 这部分代码初始化了一些变量。prev_pos是前一个位置的索引，初始化为0。eos_reached是一个布尔型tensor，表
        # 示每个提示是否已经生成了结束token（eos_id），初始化为全False。input_text_mask是一个布尔型mask，表示
        # tokens中哪些位置是输入文本的token，哪些位置是填充token。
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id

        # 当最小的prompt长度和总长度相等，直接推理下一个，也是最后一个。
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        # 这是生成文本的主循环。在每次迭代中，它首先调用模型的forward方法，计算下一个token的对数概率分布（logits）。
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # 这是生成文本的主循环。在每次迭代中，它首先调用模型的forward方法，计算下一个token的对数概率分布（logits）。
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            # 根据 input_text_mask[:, cur_pos] 从tokens[:, cur_pos]和next_token从中选择
            # 将新生成的代码加入到tokens中
            # 由于prompt长短不一，又是从最短prompt开始的，所以早些推理出来的token不一定是所有的prompt都需要。
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            # 如果logprobs为True，那么它会计算每个token的对数概率，并将其存储在token_logprobs中。
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            # 更新结束标志
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.
            prompts：一个字符串列表，包含了一批提示。每个提示都是一个字符串。
            temperature：一个浮点数，用于控制生成的随机性。值越大，生成的文本越随机；值越小，生成的文本越确定性。
            top_p：一个浮点数，用于控制生成的多样性。只有排名在前top_p的token会被考虑生成。
            max_gen_len：一个可选的整数，表示生成的文本的最大长度。如果为None，那么最大长度为模型的最大序列长度减1。
            logprobs：一个布尔值，如果为True，则返回每个token的对数概率。
            echo：一个布尔值，如果为True，则在生成的文本中包含提示。

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            dialogs：一个对话列表，每个对话是一个Message列表，表示一个完整的对话历史。
            temperature：一个浮点数，用于控制生成的随机性。值越大，生成的文本越随机；值越小，生成的文本越确定性。
            top_p：一个浮点数，用于控制生成的多样性。只有排名在前top_p的token会被考虑生成。
            max_gen_len：一个可选的整数，表示生成的文本的最大长度。如果为None，那么最大长度为模型的最大序列长度减1。
            logprobs：一个布尔值，如果为True，则返回每个token的对数概率。

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """

        # 这部分代码首先检查了max_gen_len是否为None，如果是，那么它会设置为模型的最大序列长度减1。然后，它会遍历每个对话，将对话
        # 的每个消息编码为token，结果保存在prompt_tokens中。注意，对话的第一条消息必须是系统消息，对话的最后一条消息必须是用户消息，
        # 对话的其他消息必须按照用户、助手、用户、助手的顺序交替。
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            # 如果包含了SPECIAL_TAGS，则被认定为unsafe_requests
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            # 角色的类型，可以是"system"、"user"或"assistant"。
            # system 指定系统的回复格式
            # user 真正的用户
            # 模型返回的答复
            # 如果第一个是system角色，就把这个加入到用户的第一个问题中去。
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            # 判断对话中是不是都是role和assistant交互出现
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            # 将每轮prompt和answer整合到一起
            # 将list列表与一个空列表相加，就能把嵌套列表合并成一个
            # a=[[1],[2],[3],[4],[5]]
            # merge=sum(a,[])
            # print('sum result:',merge)
            # sum result: [1, 2, 3, 4, 5]
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    torch.cumsum()函数返回一个新的张量，其中每个元素都是原张量中对应位置及之前所有元素的累加和。
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x)
    y = torch.cumsum(x, dim=1)
    print(y)
    tensor([[1, 2, 3],
        [4, 5, 6]])
    tensor([[ 1,  3,  6],
            [ 4,  9, 15]])
    
    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    # 排序，递减
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算出累加概率值
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 根据阈值去掉一部分概率预测
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    # 计算剩下的相对的概率
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 从中抽取样本
    # tensor.multinomial(1)是一个PyTorch中的函数，用于从多项式分布中抽取样本。
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # 将抽取的token收集到tensor
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token



