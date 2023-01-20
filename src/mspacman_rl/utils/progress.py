import os
import time


class ProgressDisplay(object):
    def __init__(
            self,
            prefix_text='',
            bar_symbol='#',
            bar_length=20,
            line_length=None,
            print_func=print,
            show_steps=True,
            show_time=True,
    ):
        self._prefix_text = prefix_text
        self._suffix_text = ''
        self._bar_length = bar_length
        self._line_length = line_length
        self._print_func = print_func
        self._filler_text = bar_symbol
        self._show_steps = show_steps
        self._show_time = show_time
        self._start_time = time.time()
        self._curr = None
        self._total = None
        self.update(0.0)

    def __enter__(self):
        self.update(0.0)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.terminate()

    def update(self, curr, total=None, suffix_text=''):
        self._curr = curr
        self._total = total
        if suffix_text:
            self._suffix_text = suffix_text

        percentage = self._curr if self._total is None else self._curr / self._total
        m = int(percentage * self._bar_length)
        n = self._bar_length - m
        progress_text = '[' + (self._filler_text * m) + (' ' * n) + ']' if self._bar_length > 0 else ''
        suffix_fields = list()
        if self._show_steps and self._total is not None:
            total_steps = str(self._total)
            curr_step = str(self._curr)
            steps_text = '{}/{}'.format(curr_step.rjust(len(total_steps)), total_steps)
            suffix_fields.append(steps_text)
        if self._show_time and percentage > 0:
            elapsed_time = time.time() - self._start_time
            remaining_time = elapsed_time * (1.0 - percentage) / percentage
            remaining_time = [*divmod(remaining_time, 60.0)]
            remaining_time = [*divmod(remaining_time[0], 60.0), remaining_time[1]]
            time_text = 'time: {:02.0f}:{:02.0f}:{:02.0f}'.format(*remaining_time)
            suffix_fields.append(time_text)
        if self._suffix_text:
            suffix_fields.append(self._suffix_text)

        line_length = self._line_length if self._line_length is not None else _get_line_length() - 1
        text = _format_text(self._prefix_text, progress_text, ' - '.join(suffix_fields), line_length=line_length)
        self._print('\r' + text + (' ' * max(0, line_length - len(text) - 1)), end='', flush=True)
        return text

    def terminate(self, suffix_text=''):
        if suffix_text:
            self._suffix_text = suffix_text

        percentage = self._curr if self._total is None else self._curr / self._total
        progress_text = '[' + (self._filler_text * self._bar_length) + ']' if self._bar_length > 0 else ''
        suffix_fields = list()
        if self._show_steps and self._total is not None:
            total_steps = str(self._total)
            curr_step = str(self._curr)
            steps_text = '{}/{}'.format(curr_step.rjust(len(total_steps)), total_steps)
            suffix_fields.append(steps_text)
        if self._show_time and percentage > 0:
            elapsed_time = time.time() - self._start_time
            elapsed_time = [*divmod(elapsed_time, 60.0)]
            elapsed_time = [*divmod(elapsed_time[0], 60.0), elapsed_time[1]]
            time_text = 'time: {:02.0f}:{:02.0f}:{:02.0f}'.format(*elapsed_time)
            suffix_fields.append(time_text)
        if self._suffix_text:
            suffix_fields.append(self._suffix_text)

        line_length = self._line_length if self._line_length is not None else _get_line_length() - 1
        text = _format_text(self._prefix_text, progress_text, ' - '.join(suffix_fields), line_length=line_length)
        self._print('\r' + text + (' ' * max(0, line_length - len(text) - 1)), end='\n', flush=True)
        return text

    def _print(self, *args, **kwargs):
        if self._print_func is not None:
            self._print_func(*args, **kwargs)


def _get_line_length():
    try:
        result, _ = os.get_terminal_size()
    except OSError:
        result = 80
    return result


def _format_text(prefix_text, text, suffix_text, line_length=None):
    line_length = line_length if line_length is not None else _get_line_length() - 1
    if prefix_text is not None and prefix_text != '':
        result = '{}: {} {}'.format(prefix_text, text, suffix_text)
        n = len(result) - line_length
        if n > 0:
            result = '{}: {} {}...'.format(prefix_text, text, suffix_text[:-n-3])
    else:
        result = '{} {}'.format(text, suffix_text)
        n = len(result) - line_length
        if n > 0:
            result = '{}'.format(text)
    return result
