from ..utils.bench import bench


@bench()
def run(timings: str, compare: str | None, name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..utils.timer import BenchmarkTimer

    from kitoken import Kitoken

    with BenchmarkTimer(name=name, output_dir=timings, compare_dir=compare) as tm:
        encoder: Kitoken = (
            Kitoken.from_sentencepiece_file(model)
            if model.endswith('.model')
            else Kitoken.from_tokenizers_file(model)
            if model.endswith('.json') and 'tekken' not in model
            else Kitoken.from_tekken_file(model)
            if model.endswith('.json')
            else Kitoken.from_tiktoken_file(model)
        )
        for timing_iteration in tm.iterations(n=iters, warmup=warmup):
            with timing_iteration:
                for _ in range(10):
                    encoder.encode(text, True)
