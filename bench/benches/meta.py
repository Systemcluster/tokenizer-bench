from ..utils.bench import bench

from pathlib import Path


@bench()
def run(timings: str, compare: str | None, name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..utils.timer import BenchmarkTimer

    from llama_models.llama4.tokenizer import Tokenizer

    with BenchmarkTimer(name=name, output_dir=timings, compare_dir=compare) as tm:
        encoder = Tokenizer(Path(model))
        for timing_iteration in tm.iterations(n=iters, warmup=warmup):
            with timing_iteration:
                for _ in range(10):
                    encoder.encode(text, eos=False, bos=False, allowed_special='all')
