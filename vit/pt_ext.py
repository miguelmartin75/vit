import math
import torch
# additional "unnecessary" code for vit_pt 

def dset_iter(xs, load_sample_fn, transform, shuffle, shuffle_buf_size, start=0, end=None):
    xs = xs[start:end]
    buf_size = shuffle_buf_size if shuffle else 1
    assert buf_size >= 1

    buf = []
    for xx in xs:
        buf.append(xx)
        if len(buf) >= buf_size:
            random.shuffle(buf)
            for x in buf:
                x = load_sample_fn(copy.deepcopy(x), transform)
                yield x
            buf = []

    for x in buf:
        x = load_sample_fn(copy.deepcopy(x), transform)
        yield x


class JsonlDsetIter(torch.utils.data.IterableDataset):
    def __init__(self, path, load_sample_fn, transform, shuffle, shuffle_buf_size):
        super().__init__()
        self.transform = transform
        self.path = path
        self.data = [json.loads(x) for x in open(path).readlines()]
        self.shuffle = shuffle
        self.shuffle_buf_size = shuffle_buf_size
        self.load_sample_fn = load_sample_fn
        self.N = len(self.data)

    def __len__(self):
        return self.N

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        if worker_info is None:
            start = 0
            end = None
        else:
            N = self.N
            per_worker = int(math.ceil(N / float(worker_info.num_workers)))
            i = worker_info.id
            start = i * per_worker
            end = start + per_worker

        return dset_iter(
            xs=self.data,
            load_sample_fn=self.load_sample_fn,
            transform=self.transform,
            shuffle=self.shuffle,
            shuffle_buf_size=self.shuffle_buf_size,
            start=start,
            end=end,
        )
