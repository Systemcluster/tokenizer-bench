from ..utils.bench import bench


@bench()
def run(timings: str, compare: str | None, name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..utils.timer import BenchmarkTimer

    from llama_cpp import Llama, LlamaTokenizer

    with BenchmarkTimer(name=name, output_dir=timings, compare_dir=compare) as tm:
        llama = Llama(model, vocab_only=True)
        encoder: LlamaTokenizer = LlamaTokenizer(llama)
        for timing_iteration in tm.iterations(n=iters, warmup=warmup):
            with timing_iteration:
                for _ in range(10):
                    encoder.encode(text)
