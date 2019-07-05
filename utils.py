class Dict(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


def apply_dict(function, dictionary):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            dictionary[key] = apply_dict(function, value)
        dictionary = function(dictionary)
    return dictionary


def zip_longest(*iterables):
    iterators = list(map(iter, iterables))
    longest_len = max(map(len, iterables))
    for i in range(longest_len):
        items = []
        for j in range(len(iterators)):
            try:
                item = next(iterators[j])
            except StopIteration:
                iterators[j] = iter(iterables[j])
                item = next(iterators[j])
            items.append(item)
        yield tuple(items)
