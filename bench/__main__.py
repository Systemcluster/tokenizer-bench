from . import __version__
from .utils.bench import benchmarks, datasets, tokenizers

import argparse
import functools
import gc
import logging
import operator

from multiprocessing import Process  # type: ignore
from time import sleep

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich_argparse import RichHelpFormatter  # type: ignore


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    console = Console(theme=Theme(inherit=False))
    console.print(f'[bold]Tokenizer Bench[/] [dim]({__version__})[/]')

    RichHelpFormatter.styles['argparse.groups'] = 'bold underline'
    RichHelpFormatter.styles['argparse.metavar'] = 'green dim italic'
    argparser = argparse.ArgumentParser(
        prog='tokenizer-bench',
        description='Benchmark for different tokenizers.',
        add_help=False,
        formatter_class=RichHelpFormatter
    )

    argparser_general = argparser.add_argument_group('General options')
    argparser_general.add_argument(
        '--tokenizers',
        '-t',
        action='extend',
        nargs='+',
        help='Select tokenizers. Will run all if not specified.',
        type=str,
    )
    argparser_general.add_argument(
        '--models',
        '-m',
        action='extend',
        nargs='+',
        help='Select models. Will run all if not specified.',
        type=str,
    )
    argparser_general.add_argument(
        '--datasets',
        '-d',
        action='extend',
        nargs='+',
        help='Select datasets. Will run all if not specified.',
        type=str,
    )
    argparser_general.add_argument(
        '--list-models',
        action='store_true',
        help='Show available models and exit. (default: False)',
        default=False,
    )
    argparser_general.add_argument(
        '--list-tokenizers',
        action='store_true',
        help='Show available tokenizers and exit. (default: False)',
        default=False,
    )
    argparser_general.add_argument(
        '--list-datasets',
        action='store_true',
        help='Show available datasets and exit. (default: False)',
        default=False,
    )
    argparser_general.add_argument(
        '--show-results',
        action='store_true',
        help='Show results for previous benchmark runs and exit. (default: False)',
        default=False,
    )
    argparser.add_argument('--log-level', type=str,
                           help='Log level. (default: INFO)', default='INFO')
    argparser.add_argument('--help', '-h', action='help',
                           help='Show this help message and exit.')

    argparser_bench = argparser.add_argument_group('Benchmark options')
    argparser_bench.add_argument(
        '--skip-slow',
        action=argparse.BooleanOptionalAction,
        help='Skip known very slow benchmarks. (default: False)',
        default=False,
    )
    argparser_bench.add_argument(
        '--only-slow',
        action=argparse.BooleanOptionalAction,
        help='Only run known very slow benchmarks. (default: False)',
        default=False,
    )
    argparser_bench.add_argument(
        '--allow-inf',
        action=argparse.BooleanOptionalAction,
        help='Allow known infinite benchmarks. (default: False)',
        default=False,
    )
    argparser_bench.add_argument(
        '--verify-imports',
        action=argparse.BooleanOptionalAction,
        help='Verify tokenizer imports. (default: False)',
        default=False,
    )
    argparser_bench.add_argument(
        '--exit-on-error', action=argparse.BooleanOptionalAction, help='Exit on error. (default: False)', default=False
    )
    argparser_bench.add_argument(
        '--multiprocessing',
        action=argparse.BooleanOptionalAction,
        help='Run benchmarks in their own process. (default: True)',
        default=True,
        type=bool,
    )
    argparser_bench.add_argument(
        '--error-trace',
        action=argparse.BooleanOptionalAction,
        help='Show tracebacks on errors. (default: False)',
        default=False,
    )
    argparser_bench.add_argument(
        '--timeout', type=int, help='Timeout for each benchmark in seconds. (default: 1200)', default=1200
    )

    args = argparser.parse_args()

    console.print('[dim]Arguments:[/dim]', vars(args))

    # verify log level
    if args.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        console.print(
            f'[red]Invalid log level, must be one of[/] {"[dim], [/]".join(
                [f"[bold]{x}[/]" for x in logging.getLevelNamesMapping()
                 if x != "NOTSET"]
            )}'
        )
        exit(1)

    logging.basicConfig(
        level=args.log_level.upper(),
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    # list models
    if args.list_models:
        console.print('[dim]Available models:[/]')
        for model in benchmarks:
            console.print(f'[blue]{model}[/]')
        exit(0)

    # list tokenizers
    if args.list_tokenizers:
        console.print('[dim]Available tokenizers:[/]')
        for tokenizer in tokenizers:
            console.print(f'[blue]{tokenizer}[/]')
        exit(0)

    # list datasets
    if args.list_datasets:
        console.print('[dim]Available datasets:[/]')
        for dataset in datasets:
            console.print(f'[blue]{dataset}[/]')
        exit(0)

    # verify tokenizer args
    if args.tokenizers:
        args.tokenizers = functools.reduce(
            operator.add, ([s.strip() for s in t.split(',')] for t in args.tokenizers), [])
        for tok in args.tokenizers:
            if tok not in tokenizers:
                console.print(f'[red]Unknown tokenizer: [bold]{tok}[/][/]')
                console.print(
                    f'[dim]Available tokenizers:[/dim] {", ".join(f"[blue]{x}[/]" for x in tokenizers)}')
                exit(1)

    # verify model args
    if args.models:
        args.models = functools.reduce(
            operator.add, ([s.strip() for s in t.split(',')] for t in args.models), [])
        for model in args.models:
            if model not in benchmarks:
                console.print(f'[red]Unknown model: [bold]{model}[/][/]')
                console.print(
                    f'[dim]Available models:[/dim] {", ".join(f"[blue]{x}[/]" for x in benchmarks)}')
                exit(1)

    # verify dataset args
    if args.datasets:
        args.datasets = functools.reduce(
            operator.add, ([s.strip() for s in t.split(',')] for t in args.datasets), [])
        for dataset in args.datasets:
            if dataset not in datasets:
                console.print(f'[red]Unknown dataset: [bold]{dataset}[/][/]')
                console.print(
                    f'[dim]Available datasets:[/dim] {", ".join(f"[blue]{x}[/]" for x in datasets)}')
                exit(1)

    def do_verify_imports() -> None:
        from vendor.gpt_bpe import BPETokenizer  # type: ignore # noqa: F401

        from kitoken import Kitoken  # type: ignore # noqa: F401
        from llama_cpp import Llama, LlamaTokenizer  # type: ignore # noqa: F401
        from sentencepiece import SentencePieceProcessor  # type: ignore # noqa: F401
        from tiktoken.core import Encoding  # type: ignore # noqa: F401
        from tokenizers import Tokenizer  # type: ignore # noqa: F401

    if args.verify_imports:
        do_verify_imports()

    gc.collect()
    gc.freeze()
    gc.disable()

    if args.show_results:
        console.print('[bold]Showing results...[/]')
        from .utils.timer import Timings

        for model, tokenizer in benchmarks.items():
            if args.models and model not in args.models:
                continue
            for tok, _ in tokenizer.items():
                if args.tokenizers and tok not in args.tokenizers:
                    continue
                for name, _ in datasets.items():
                    if args.datasets and name not in args.datasets:
                        continue
                    console.print(f'[blue bold]{tok} - {model} - {name}[/]')
                    timings = Timings.from_dir(
                        f'{tok} - {model} - {name}', 'timings')
                    timings.print_timings()
        exit(0)

    console.print('[bold]Running benchmarks...[/]')
    try:
        for model, tokenizer in benchmarks.items():
            if args.models and model not in args.models:
                logger.info(f'Skipping unselected model: {model}')
                continue
            for tok, params in tokenizer.items():
                if args.tokenizers and tok not in args.tokenizers:
                    logger.info(f'Skipping unselected tokenizer: {
                                model} - {tok}')
                    continue
                for name, file in datasets.items():
                    if args.datasets and name not in args.datasets:
                        logger.info(f'Skipping unselected dataset: {
                                    model} - {tok} - {name}')
                        continue
                    if args.skip_slow and name in params['slow']:
                        logger.info(f'Skipping slow benchmark: {
                                    model} - {tok} - {name}')
                        continue
                    if args.only_slow and name not in params['slow']:
                        logger.info(f'Skipping non-slow benchmark: {
                                    model} - {tok} - {name}')
                        continue
                    if not args.allow_inf and name in params['inf']:
                        logger.info(f'Skipping infinite benchmark: {
                                    model} - {tok} - {name}')
                        continue
                    p: Process | None = None
                    try:
                        if tok not in tokenizers:
                            logger.error(f'Unknown tokenizer: {tok}')
                            continue
                        fn = tokenizers[tok]
                        if not args.multiprocessing:
                            fn(f'{tok} - {model} - {name}',
                               str(params['model']), file, 100, 15)
                        else:
                            p = Process(
                                target=fn, args=(
                                    f'{tok} - {model} - {name}', str(params['model']), file, 100, 15)
                            )
                            p.start()
                            result = p.join(timeout=args.timeout)
                            if result is None and p.exitcode is None:
                                print()
                                logger.error(f'Timeout for {
                                    model} - {tok} - {name} after {args.timeout}s')
                                p.terminate()
                                while p.is_alive():
                                    sleep(0.1)
                            elif result is None and p.exitcode is not None and p.exitcode != 0:
                                logger.error(
                                    f'{model} - {tok} - {name} exited with code {p.exitcode}')
                                p.close()
                    except KeyboardInterrupt as e:
                        print()
                        if p:
                            p.join(timeout=2)
                            p.terminate()
                            p.close()
                        raise e
                    except Exception as e:
                        if args.exit_on_error:
                            raise e
                        else:
                            if args.error_trace:
                                logger.exception(e)
                            else:
                                logger.error(e)
                            continue
                    finally:
                        gc.collect()
                        sleep(0.1)
    except KeyboardInterrupt:
        logger.info('Interrupted')
        exit(0)
    except Exception as e:
        logger.exception(e)
        exit(1)
