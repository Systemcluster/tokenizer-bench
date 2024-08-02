from ..utils.bench import bench


@bench()
def run(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..utils.timer import BenchmarkTimer

    from tokenizers import Tokenizer

    with BenchmarkTimer(name=name) as tm:
        encoder: Tokenizer = Tokenizer.from_file(model)
        encoder.encode_special_tokens = False  # type: ignore
        for timing_iteration in tm.iterations(n=iters, warmup=warmup):
            with timing_iteration:
                for _ in range(10):
                    encoder.encode(text, add_special_tokens=False)
