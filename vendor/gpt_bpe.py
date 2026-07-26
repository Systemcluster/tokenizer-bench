import platform
import sys
import ctypes

from typing import Any, Sequence, Union

import numpy

file_name = "gpt_bpe_" + sys.platform.lower() + "_" + \
    platform.machine().lower() + ".dylib"
file_path = __file__.replace("gpt_bpe.py", file_name)
gpt_bpe = ctypes.cdll.LoadLibrary(file_path)


class Tokens(ctypes.Structure):
    _fields_ = [('tokens', ctypes.c_void_p),
                ('len', ctypes.c_uint64)]


class TokenBuffer:
    def __init__(self, tokens: Tokens) -> None:
        self.tokens = tokens

    def __del__(self) -> None:
        gpt_bpe.freeTokens(self.tokens)


class BackedArray(numpy.ndarray):
    def __new__(
        cls, shape: Any, dtype: Any = float, buffer: Any = None, offset: int = 0,
        strides: Any = None, order: Any = None, backed: TokenBuffer | None = None,
    ) -> 'BackedArray':
        obj = super().__new__(cls, shape, dtype,
                              buffer, offset, strides, order)
        # set the new 'info' attribute to the value passed
        obj.backed = backed
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.backed = getattr(obj, 'backed', None)


class BPETokenizer():
    def __init__(self, vocab_id: str) -> None:
        self.vocab_id = vocab_id.encode("utf8")
        gpt_bpe.initTokenizer(self.vocab_id)
        gpt_bpe.tokenizeBuffer.restype = Tokens
        gpt_bpe.decode.restype = ctypes.c_char_p
        gpt_bpe.freeTokens.argtypes = [Tokens]

    def encode(self, text: str) -> numpy.ndarray:
        encoded = text.encode("utf8")
        tokens_struct = gpt_bpe.tokenizeBuffer(
            self.vocab_id, encoded, len(encoded))
        tokens_arr_type = (ctypes.c_uint32 * tokens_struct.len)
        tokens_buf = tokens_arr_type.from_address(tokens_struct.tokens)
        return BackedArray([len(tokens_buf)],
                           dtype=ctypes.c_uint32, buffer=tokens_buf,
                           backed=TokenBuffer(tokens_struct))

    def decode(self, arr: Union[numpy.ndarray, Sequence[int]]) -> str:
        if type(arr) == numpy.ndarray and arr.dtype != ctypes.c_uint32:
            arr = arr.astype(ctypes.c_uint32)
        elif type(arr) == BackedArray:
            pass
        elif type(arr) != numpy.ndarray:
            arr = numpy.array(arr, dtype=ctypes.c_uint32)
        tokens = Tokens()
        tokens.len = len(arr)
        tokens.tokens = ctypes.c_void_p(arr.ctypes.data)
        decoded = gpt_bpe.decode(self.vocab_id, tokens)
        return decoded.decode('utf8')


if __name__ == "__main__":
    encoder = BPETokenizer("gpt2-tokenizer")

    test_str = "This is a test."
    tokens = encoder.encode(test_str)

    print(tokens)

    print(encoder.decode(tokens))
    print(encoder.decode([1212, 318, 257, 1332, 13]))
