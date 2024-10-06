from . import __version__

import argparse
import glob
import logging
import os

from collections.abc import Callable
from typing import cast

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich_argparse import RichHelpFormatter  # type: ignore


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    console = Console(theme=Theme(inherit=False))
    console.print(f'[bold]Tokenizer Test Generator[/] [dim]({__version__})[/]')

    RichHelpFormatter.styles['argparse.groups'] = 'bold underline'
    RichHelpFormatter.styles['argparse.metavar'] = 'red dim italic'
    argparser = argparse.ArgumentParser(
        prog='tokenizer-generator',
        description='Generator for tokenization test data.',
        add_help=False,
        formatter_class=RichHelpFormatter,
    )
    argparser.add_argument('--log-level', type=str, help='Log level. (default: INFO)', default='INFO')
    argparser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')
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
        format='[%(module)s] %(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    os.environ['TOKENIZERS_LOG'] = str.replace(args.log_level.lower(), 'critical', 'off').removesuffix('ing')

    def gen_lines(
        model: str, name: str, encode: Callable[[str], list[int]], decode: Callable[[list[int]], str]
    ) -> None:
        for data in ['small', 'mixed']:
            text = open(f'data/{data}_input.txt', encoding='utf-8').read()
            lines = text.strip('\n').split('\n')
            lines_tokens = []
            lines_decoded = []
            for line_ in lines:
                line = line_.rstrip()
                if len(line) == 0:
                    continue
                line = line.replace('\\n', '\n').replace('\\s', ' ')
                output = encode(line)
                lines_tokens.append(', '.join([str(token) for token in output]))
                lines_tokens.append('\n')
                decoded = decode(output)
                decoded = decoded.replace('\n', '\\n')
                trailing_spaces = len(decoded) - len(decoded.rstrip())
                decoded = decoded.rstrip() + ('\\s' * trailing_spaces)
                if len(decoded) > 0:
                    lines_decoded.append(decoded)
                else:
                    lines_decoded.append('\\x')
            text_decoded = '\n'.join(lines_decoded) + '\n'
            os.makedirs(f'outputs/{model}', exist_ok=True)
            with open(f'outputs/{model}/{data}_tokens_{name}.txt', 'w', encoding='utf-8', newline='\n') as f:
                f.writelines(lines_tokens)
            if len(lines) != len(lines_decoded) or text != text_decoded:
                logger.info(f'{model}: {name}: {data}_input.txt: {len(lines)} != {
                    len(lines_decoded)} or lines not equal')
                with open(f'outputs/{model}/{data}_output_{name}.txt', 'w', encoding='utf-8', newline='\n') as f:
                    f.write(text_decoded)

    def gen_full(model: str, name: str, encode: Callable[[str], list[int]], decode: Callable[[list[int]], str]) -> None:
        for data in ['utf8']:
            text = open(f'data/{data}_input.txt', encoding='utf-8').read().replace('\\n', '\n').replace('\\s', ' ')
            output = encode(text)
            decoded = decode(output)
            os.makedirs(f'outputs/{model}', exist_ok=True)
            with open(f'outputs/{model}/{data}_tokens_{name}.txt', 'w', encoding='utf-8', newline='\n') as f:
                f.write(', '.join([str(token) for token in output]))
            if len(decoded) != len(text) or text != decoded:
                logger.info(f'{model}: {name}: {data}_input.txt: {len(text)} != {
                    len(decoded)} or text not equal')
                with open(f'outputs/{model}/{data}_output_{name}.txt', 'w', encoding='utf-8', newline='\n') as f:
                    f.write(decoded)

    try:
        # generate outputs for all models

        def sentencepiece() -> None:
            from sentencepiece import SentencePieceProcessor

            models = glob.glob('models/tests/*.model')
            for model in models:
                name = os.path.basename(model).split('.')[0]
                encoder: SentencePieceProcessor = SentencePieceProcessor()
                encoder.Load(model)
                gen_lines('sentencepiece', name, encoder.EncodeAsIds, encoder.Decode)
                gen_full('sentencepiece', name, encoder.EncodeAsIds, encoder.Decode)

        sentencepiece()

        def tokenizers() -> None:
            from tokenizers import Encoding, Tokenizer

            models = glob.glob('models/tests/*.json')
            for model in models:
                print(model)
                name = os.path.basename(model).split('.')[0]
                with open(model, encoding='utf-8') as f:
                    text = f.read()
                    if text.find('"version": "v3"') != -1:
                        continue
                str.replace(text, '\n', '\\n')
                encoder: Tokenizer = Tokenizer.from_file(model)
                encoder.encode_special_tokens = False  # type: ignore
                gen_lines(
                    'tokenizers',
                    name,
                    lambda i, encoder=encoder: cast(Encoding, encoder.encode(i, add_special_tokens=False)).ids,
                    lambda i, encoder=encoder: encoder.decode(i, skip_special_tokens=False),
                )
                gen_full(
                    'tokenizers',
                    name,
                    lambda i, encoder=encoder: cast(Encoding, encoder.encode(i, add_special_tokens=False)).ids,
                    lambda i, encoder=encoder: encoder.decode(i, skip_special_tokens=False),
                )

        tokenizers()

        def tiktoken() -> None:
            from tiktoken import Encoding, get_encoding

            models = glob.glob('models/tests/*.tiktoken')
            for model in models:
                name = os.path.basename(model).split('.')[0]
                encoder: Encoding = get_encoding(name)
                gen_lines(
                    'tiktoken',
                    name,
                    lambda i, encoder=encoder: encoder.encode(i, allowed_special='all'),
                    encoder.decode,
                )
                gen_full('tiktoken', name, encoder.encode, encoder.decode)

        tiktoken()

        def tekken() -> None:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            models = glob.glob('models/tests/*.json')
            for model in models:
                name = os.path.basename(model).split('.')[0]
                with open(model, encoding='utf-8') as f:
                    text = f.read()
                    if text.find('"version": "v3"') == -1:
                        continue
                str.replace(text, '\n', '\\n')
                encoder = MistralTokenizer.from_file('models/tekken_nemo.json')
                gen_lines(
                    'tekken',
                    name,
                    lambda i, encoder=encoder: cast(
                        list[int], encoder.instruct_tokenizer.tokenizer.encode(i, False, False)
                    ),
                    lambda i, encoder=encoder: encoder.decode(i),
                )
                gen_full(
                    'tekken',
                    name,
                    lambda i, encoder=encoder: cast(
                        list[int], encoder.instruct_tokenizer.tokenizer.encode(i, False, False)
                    ),
                    lambda i, encoder=encoder: encoder.decode(i),
                )

        tekken()

    except KeyboardInterrupt:
        print('Interrupted')
    except Exception as e:
        logger.exception(e)
        exit(1)
