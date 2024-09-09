import os
import time

from collections.abc import Iterator
from math import ceil, floor
from typing import Any, Optional

import numpy as np
import scipy as sp

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.theme import Theme


class Timings:
    name: str
    timings: list[float]
    summary_fmt: str
    iters_fmt: str

    console = Console(theme=Theme(inherit=False))

    def __init__(self: 'Timings', name: str) -> None:
        self.name = name or 'Timing'
        self.timings = []
        self.summary_fmt = (
            '\t[magenta]time: [/]\\[ [dim]{p0:.5f}s[/]  [bold]{avg:.5f}s[/]  [dim]{p100:.5f}s[/]]\n'
            '\t[dim]med: [/][dim]{med:.5f}s[/] [dim]mod: [/][dim]{mod:.5f}s[/] '
            '[dim]std: [/][dim]{std:.5f}s[/] [dim]sum: [/][dim]{sum:.5f}s[/]'
        )

    @staticmethod
    def from_dir(name: str, directory: str) -> 'Timings':
        t = Timings(name)
        t.load_timings(directory)
        return t

    def print_timings(self: 'Timings') -> None:
        if len(self.timings) == 0:
            self.console.print(f'[dim]No timings for {self.name}[/]')
            return
        s = self.summary_fmt.format(
            name=self.name,
            n=len(self.timings),
            avg=self.avg(),
            med=self.med(),
            mod=self.mod(),
            std=self.std(),
            sum=self.sum(),
            **{f'p{i}': self.percentile(i) for i in range(0, 101, 10)},
        )
        self.console.print(s)

    def print_timings_compare(self: 'Timings', other: 'Timings') -> None:
        if len(self.timings) == 0:
            return
        if len(other.timings) == 0:
            self.console.print(f'[dim]No timings for {other.name}[/]')
            return
        p0_diff = self.percentile(0) - other.percentile(0)
        p0_diff_percent = p0_diff / other.percentile(0) * 100
        avg_diff = self.avg() - other.avg()
        avg_diff_percent = avg_diff / other.avg() * 100
        p100_diff = self.percentile(100) - other.percentile(100)
        p100_diff_percent = p100_diff / other.percentile(100) * 100
        s = '  [dim]compared to [italic]{other_name}[/]:[/]\n'
        t = '\t[dim]time: [/]\\['
        d = '\t[dim]diff: [/]\\['
        if p0_diff >= 0:
            if p0_diff_percent > 1.5:
                t += '[dim red]+{p0:.5f}s[/] '
                d += '[dim red]+{p0_diff_percent: >7.3f}%[/] '
            else:
                t += '[dim]+{p0:.5f}s[/] '
                d += '[dim]+{p0_diff_percent: >7.3f}%[/] '
        else:
            if p0_diff_percent < -1.5:
                t += '[dim green]{p0:.5f}s[/] '
                d += '[dim green]-{p0_diff_percent: >7.3f}%[/] '
            else:
                t += '[dim]{p0:.5f}s[/] '
                d += '[dim]-{p0_diff_percent: >7.3f}%[/] '
        if avg_diff >= 0:
            if avg_diff_percent > 1.5:
                t += '[bold red]+{avg:.5f}s[/] '
                d += '[red]+{avg_diff_percent: >7.3f}%[/] '
            else:
                t += '[white]+{avg:.5f}s[/] '
                d += '[white]+{avg_diff_percent: >7.3f}%[/] '
        else:
            if avg_diff_percent < -1.5:
                t += '[bold green]{avg:.5f}s[/] '
                d += '[green]-{avg_diff_percent: >7.3f}%[/] '
            else:
                t += '[white]{avg:.5f}s[/] '
                d += '[white]-{avg_diff_percent: >7.3f}%[/] '
        if p100_diff >= 0:
            if p100_diff_percent > 1.5:
                t += '[dim red]+{p100:.5f}s[/]'
                d += '[dim red]+{p100_diff_percent: >7.3f}%[/]'
            else:
                t += '[dim]+{p100:.5f}s[/]'
                d += '[dim]+{p100_diff_percent: >7.3f}%[/]'
        else:
            if p100_diff_percent < -1.5:
                t += '[dim green]{p100:.5f}s[/]'
                d += '[dim green]-{p100_diff_percent: >7.3f}%[/]'
            else:
                t += '[dim]{p100:.5f}s[/]'
                d += '[dim]-{p100_diff_percent: >7.3f}%[/]'
        t += ']'
        d += ']'
        s += t + '\n' + d
        self.console.print(
            s.format(
                name=self.name,
                other_name=other.name,
                p0=p0_diff,
                avg=avg_diff,
                p100=p100_diff,
                p0_diff_percent=abs(p0_diff_percent),
                avg_diff_percent=abs(avg_diff_percent),
                p100_diff_percent=abs(p100_diff_percent),
            )
        )

    def write_timings(self: 'Timings', output_dir: str) -> None:
        if len(self.timings) == 0:
            return
        tms = list(self.timings)
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/{self.name}.txt', 'w', encoding='utf8', newline='\n') as f:
            f.write('\n'.join(str(t) for t in tms))
            f.write('\n')

    def load_timings(self: 'Timings', output_dir: str) -> None:
        if not os.path.isdir(output_dir):
            return
        if not os.path.isfile(f'{output_dir}/{self.name}.txt'):
            return
        with open(f'{output_dir}/{self.name}.txt', encoding='utf8', newline='\n') as f:
            for line in f.readlines():
                if line.strip() == '':
                    continue
                self.timings.append(float(line.strip()))

    def min(self: 'Timings') -> float:
        if len(self.timings) == 0:
            return 0
        return np.min(list(self.timings))

    def max(self: 'Timings') -> float:
        if len(self.timings) == 0:
            return 0
        return np.max(list(self.timings))

    def avg(self: 'Timings') -> float:
        if len(self.timings) == 0:
            return 0
        return float(np.mean(list(self.timings)))

    def med(self: 'Timings') -> float:
        if len(self.timings) == 0:
            return 0
        return float(np.median(list(self.timings)))

    def mod(self: 'Timings') -> float:
        if len(self.timings) == 0:
            return 0
        return sp.stats.mode([(int(x * 10000) / 10000) for x in list(self.timings)]).mode

    def std(self: 'Timings') -> float:
        if len(self.timings) == 0:
            return 0
        return float(np.std(list(self.timings)))

    def sum(self: 'Timings') -> float:
        if len(self.timings) == 0:
            return 0
        return np.sum(list(self.timings))

    def range(self: 'Timings') -> tuple[float, float]:
        if len(self.timings) == 0:
            return 0, 0
        tms = list(self.timings)
        return float(np.percentile(tms, 0)), float(np.percentile(tms, 100))

    def percentile(self: 'Timings', p: int) -> float:
        if len(self.timings) == 0:
            return 0
        return float(np.percentile(list(self.timings), p))

    def push(self: 'Timings', value: float) -> None:
        self.timings.append(value)

    def __len__(self: 'Timings') -> int:
        return len(self.timings)

    def __iter__(self: 'Timings') -> Iterator[float]:
        return iter(self.timings)

    def __contains__(self: 'Timings', i: float) -> bool:
        return i in self.timings


class BenchmarkTimer:
    _timer: Timings
    _last_timer: Timings | None
    _compare: Timings | None
    _name: str
    _print_summary: bool
    _last_unprinted_tmi: Optional['TimingIteration']
    _used: bool
    _output_dir: str
    console = Console(theme=Theme(inherit=False))

    def __init__(
        self: 'BenchmarkTimer',
        name: str = 'Timing',
        print_summary: bool = True,
        output_dir: str = 'timings',
        compare_dir: str | None = None,
    ) -> None:
        self._timer = Timings(name=name)
        self._name = name
        self._print_summary = print_summary
        self._last_unprinted_tmi = None
        self._used = False
        self._output_dir = output_dir
        if output_dir:
            self._last_timer = Timings.from_dir(name, output_dir)
            self._last_timer.name = 'last run'
        else:
            self._last_timer = None
        if compare_dir:
            self._compare = Timings.from_dir(name, compare_dir)
            self._compare.name = os.path.basename(compare_dir)
        else:
            self._compare = None

    class TimingIteration:
        def __init__(
            self: 'BenchmarkTimer.TimingIteration', timer: 'BenchmarkTimer', i: int, is_warmup: bool = False
        ) -> None:
            self._timer = timer
            self.i = i
            self.start_time = None
            self.end_time = None
            self.is_warmup = is_warmup

        def __enter__(self: 'BenchmarkTimer.TimingIteration') -> None:
            self.start_time = time.perf_counter()

        def __exit__(self: 'BenchmarkTimer.TimingIteration', exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self.end_time = time.perf_counter()
            if exc_type is None:
                if not self.is_warmup:
                    self._timer._timer.push(self.total_seconds())
            else:
                self._timer.console.print(
                    f'\n[yellow]Cancelled iteration {self.i}{
                        " (warmup)" if self.is_warmup else ""} after {self.total_seconds():.5}s[/]'
                )
                raise exc_val

        def total_seconds(self: 'BenchmarkTimer.TimingIteration') -> float:
            return (self.end_time or 0) - (self.start_time or 0)

    def iterations(self: 'BenchmarkTimer', n: int = 100, warmup: int = 10) -> Iterator['TimingIteration']:
        assert not self._used
        self._used = True
        try:
            p = BarColumn()
            t = TextColumn('{task.fields[avg]}', justify='left')
            with Progress(p, t, transient=True, auto_refresh=False) as progress:
                task = progress.add_task(
                    f'{self._name}', total=(ceil(n / 5) + 1), avg=f'[dim]warming up: 1/{warmup}[/dim]'
                )
                progress.refresh()
                for i in range(n + warmup):
                    if i < warmup:
                        progress.tasks[task].fields['avg'] = f'[dim]warming up: {
                            i + 1}/{warmup}[/dim]'
                        progress.refresh()
                    if floor((i - warmup) % 5) == 0 and i >= warmup:
                        progress.advance(task, 1)
                        progress.tasks[task].fields['avg'] = f'[dim]avg: {self._timer.avg():.5f}s sum: {
                                self._timer.sum():.5f}s[/dim]'
                        progress.refresh()
                    self._last_unprinted_tmi = self.TimingIteration(self, i, i < warmup)
                    yield self._last_unprinted_tmi
                progress.advance(task, 1)
                progress.refresh()
        except ImportError:
            # caused here by manually interrupting the benchmark
            pass

    def __enter__(self: 'BenchmarkTimer') -> 'BenchmarkTimer':
        if self._print_summary:
            self.console.print(f'[blue bold]{self._name}[/][bold dim]...[/]')
        return self

    def __exit__(self: 'BenchmarkTimer', exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.console.print(
                f'[yellow]Cancelled benchmark {self._name} after {
                    len(self._timer)} iterations[/]'
            )
        self._timer.print_timings()
        if self._last_timer:
            self._timer.print_timings_compare(self._last_timer)
        if self._compare:
            self._timer.print_timings_compare(self._compare)
        if exc_type is None and self._output_dir:
            self._timer.write_timings(self._output_dir)
        if exc_type is not None:
            raise exc_val
