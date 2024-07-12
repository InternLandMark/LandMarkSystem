class CircularIterator:
    """class for circular iteration"""

    def __init__(self, lst):
        self.lst = lst
        self.iterator = iter(lst)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.lst)
            return next(self.iterator)


def rm_redundant_words_in_state_dict(state_dict, word_list):
    keys = sorted(state_dict.keys())
    for key in keys:
        new_key = key
        for word in word_list:
            w_begin = new_key.find(word)
            if w_begin > -1:
                new_key = new_key[:w_begin] + new_key[w_begin + len(word) :]

        state_dict[new_key] = state_dict.pop(key)


class st:
    """Define color"""

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
