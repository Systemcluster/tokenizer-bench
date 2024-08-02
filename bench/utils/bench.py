import gc

from collections import OrderedDict
from collections.abc import Callable
from typing import Any


def set_high_priority() -> None:
    import sys

    try:
        sys.getwindowsversion()  # type: ignore
    except AttributeError:
        windows = False
    else:
        windows = True

    if windows:
        import win32api  # type: ignore
        import win32con  # type: ignore
        import win32process  # type: ignore

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
    else:
        import os

        os.nice(-10)  # type: ignore


BenchFunc = Callable[..., Any]


def bench() -> Callable[..., BenchFunc]:
    def decorator(func: BenchFunc) -> BenchFunc:
        def wrapper(*args: Any, **kwargs: Any) -> None:
            gc.disable()
            gc.freeze()
            try:
                set_high_priority()
            except Exception as e:
                print(f'Could not set process priority: {e}')
            try:
                func(*args, **kwargs)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                raise e

        return wrapper

    return decorator


def run_bench_kitoken(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..benches.kitoken import run

    text = open(text, encoding='utf-8', newline='\n').read()
    run(name, model, text, iters, warmup)


def run_bench_tiktoken(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..benches.tiktoken import run

    text = open(text, encoding='utf-8', newline='\n').read()
    run(name, model, text, iters, warmup)


def run_bench_sentencepiece(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..benches.sentencepiece import run

    text = open(text, encoding='utf-8', newline='\n').read()
    run(name, model, text, iters, warmup)


def run_bench_tokenizers(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..benches.tokenizers import run

    text = open(text, encoding='utf-8', newline='\n').read()
    run(name, model, text, iters, warmup)


def run_bench_tekken(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..benches.tekken import run

    text = open(text, encoding='utf-8', newline='\n').read()
    run(name, model, text, iters, warmup)


def run_bench_gptbpe(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..benches.gptbpe import run

    text = open(text, encoding='utf-8', newline='\n').read()
    run(name, model, text, iters, warmup)


def run_bench_llamacpp(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..benches.llamacpp import run

    text = open(text, encoding='utf-8', newline='\n').read()
    run(name, model, text, iters, warmup)


datasets = OrderedDict([
    ('pride and prejudice', 'data/pride_and_prejudice.txt'),
    ('utf8 sequence', 'data/utf8_sequence_0x10ffff.txt'),
    ('wagahai', 'data/wagahai.txt'),
])  # fmt: skip

tokenizers = OrderedDict([
    ('kitoken', run_bench_kitoken),
    ('tiktoken', run_bench_tiktoken),
    ('sentencepiece', run_bench_sentencepiece),
    ('tokenizers', run_bench_tokenizers),
    ('tekken', run_bench_tekken),
    ('gpt_bpe', run_bench_gptbpe),
    ('llamacpp', run_bench_llamacpp),
])  # fmt: skip

benchmarks = OrderedDict([
    ('gpt2', OrderedDict([
        ('kitoken', {
            'model': 'models/gpt2.json',
            'slow': [],
            'inf': [],
        }),
        ('tiktoken', {
            'model': 'gpt2',
            'slow': ['utf8 sequence'],
            'inf': [],
        }),
        ('tokenizers', {
            'model': 'models/gpt2.json',
            'slow': [],
            'inf': [],
        }),
        ('gpt_bpe', {
            'model': 'gpt2',
            'slow': [],
            'inf': ['utf8 sequence'],
        }),
    ])),
    ('llama2', OrderedDict([
        ('kitoken', {
            'model': 'models/llama2.model',
            'slow': [],
            'inf': [],
        }),
        ('sentencepiece', {
            'model': 'models/llama2.model',
            'slow': [],
            'inf': [],
        }),
        ('tokenizers', {
            'model': 'models/llama2.json',
            'slow': [],
            'inf': [],
        }),
        ('gpt_bpe', {
            'model': 'llama-tokenizer',
            'slow': [],
            'inf': [],
        }),
        ('llamacpp', {
            'model': 'models/llama-2-7b.Q2_K.gguf',
            'slow': ['pride and prejudice'],
            'inf': ['utf8 sequence'],
        }),
    ])),
    ('cl100k', OrderedDict([
        ('kitoken', {
            'model': 'models/cl100k_base.tiktoken',
            'slow': [],
            'inf': [],
        }),
        ('tiktoken', {
            'model': 'cl100k_base',
            'slow': ['utf8 sequence'],
            'inf': [],
        }),
    ])),
    ('xlnet', OrderedDict([
        ('kitoken', {
            'model': 'models/xlnet_base_cased.model',
            'slow': [],
            'inf': [],
        }),
        ('sentencepiece', {
            'model': 'models/xlnet_base_cased.model',
            'slow': ['utf8 sequence'],
            'inf': [],
        }),
        ('tokenizers', {
            'model': 'models/xlnet_base_cased.json',
            'slow': [],
            'inf': [],
        }),
    ])),
    ('nemo', OrderedDict([
        ('kitoken', {
            'model': 'models/tekken_nemo.json',
            'slow': [],
            'inf': [],
        }),
        ('tekken', {
            'model': 'models/tekken_nemo.json',
            'slow': ['utf8 sequence'],
            'inf': [],
        }),
    ])),
])  # fmt: skip
