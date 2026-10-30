"""Bounded CPU prefetch preserving the consumed (not producer) resume cursor.
Opt-in helper only: never starts training, changes labels, or reorders samples.
One CPU producer owns the underlying loader/processor; no GPU work here.
"""
import copy,queue,threading

class OrderedPrefetch:
    def __init__(self,inner,depth=4):
        if depth<1:raise ValueError('depth must be positive')
        self.inner=inner;self.depth=depth
        self.consumed=copy.deepcopy(inner.state_dict())
        self.worker=None;self.stop_event=threading.Event()
    def state_dict(self):return copy.deepcopy(self.consumed)
    @property
    def count(self):return self.consumed.get('count',self.consumed.get('cursor',0))
    @property
    def cursor(self):return self.consumed.get('cursor',0)
    def load_state_dict(self,state):
        self.close()
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError('Producer still reading; do not restore concurrently')
        self.inner.load_state_dict(copy.deepcopy(state));self.consumed=copy.deepcopy(state)
        self.worker=None
    def close(self):
        self.stop_event.set()
        if self.worker is not None:self.worker.join(timeout=2)
    def __iter__(self):
        if self.worker is not None:raise RuntimeError('Close and restore before a second iterator')
        self.stop_event.clear();self.q=queue.Queue(self.depth)
        def put(item):
            while not self.stop_event.is_set():
                try:self.q.put(item,timeout=.05);return True
                except queue.Full:pass
            return False
        def produce():
            try:
                for batch in self.inner:
                    if self.stop_event.is_set():return
                    state=copy.deepcopy(self.inner.state_dict())
                    if not put(('item',batch,state)):return
                put(('end',None,None))
            except BaseException as exc:put(('error',exc,None))
        self.worker=threading.Thread(target=produce,daemon=True);self.worker.start()
        try:
            while True:
                kind,value,state=self.q.get()
                if kind=='end':return
                if kind=='error':raise value
                self.consumed=state
                yield value
        finally:self.close()
