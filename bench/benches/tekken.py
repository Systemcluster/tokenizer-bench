from ..utils.bench import bench


@bench()
def run(timings: str, compare: str | None, name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..utils.timer import BenchmarkTimer

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    with BenchmarkTimer(name=name, output_dir=timings, compare_dir=compare) as tm:
        encoder = MistralTokenizer.from_file(model)
        for timing_iteration in tm.iterations(n=iters, warmup=warmup):
            with timing_iteration:
                for _ in range(10):
                    encoder.instruct_tokenizer.tokenizer.encode(
                        text, True, True)
