from ..utils.bench import bench

from pathlib import Path


@bench()
def run(timings: str, compare: str | None, name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..utils.timer import BenchmarkTimer

    from gigatoken import Tokenizer

    model_path = Path(model)
    if model_path.suffix == '.model':
        encoder = Tokenizer.from_sentencepiece(model_path)
    elif model_path.suffix == '.tiktoken':
        encoder = Tokenizer.from_tiktoken(model_path)
    else:
        encoder = Tokenizer(model_path)

    with BenchmarkTimer(name=name, output_dir=timings, compare_dir=compare) as tm:
        for timing_iteration in tm.iterations(n=iters, warmup=warmup):
            with timing_iteration:
                for _ in range(10):
                    encoder.encode(text)
