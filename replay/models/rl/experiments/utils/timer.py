import timeit

timer = timeit.default_timer


def print_with_timestamp(text: str, start_time: float):
    print(f'[{timer() - start_time:5.1f}] {text}')
