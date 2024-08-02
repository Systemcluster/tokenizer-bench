from ..utils.bench import bench


@bench()
def run(name: str, model: str, text: str, iters: int, warmup: int) -> None:
    from ..utils.timer import BenchmarkTimer

    from sentencepiece import SentencePieceProcessor

    with BenchmarkTimer(name=name) as tm:
        encoder: SentencePieceProcessor = SentencePieceProcessor()
        encoder.Load(model)
        for timing_iteration in tm.iterations(n=iters, warmup=warmup):
            with timing_iteration:
                for _ in range(10):
                    encoder.EncodeAsIds(text)
