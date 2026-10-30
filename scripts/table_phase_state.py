"""Deterministic one-pass corrective cursor; independent of the old full pass."""
import math,random

class PhaseCursor:
    def __init__(self,total,seed=42):
        if total<1:raise ValueError('Corrective phase must have examples')
        self.total=total;self.seed=seed;self.cursor=0
        self.order=list(range(total));random.Random(seed).shuffle(self.order)
    def state_dict(self):return {'cursor':self.cursor,'total':self.total,'seed':self.seed}
    def load_state_dict(self,state):
        if state['total']!=self.total or state['seed']!=self.seed or not 0<=state['cursor']<=self.total:
            raise ValueError('Corrective dataset/cursor mismatch')
        self.cursor=state['cursor']
    def next_index(self):
        if self.cursor==self.total:raise StopIteration
        index=self.order[self.cursor];self.cursor+=1;return index

def phase_lr(step,total_steps,maximum=2e-5,minimum=1e-5,warmup=8):
    if total_steps<1 or step<0:raise ValueError('Invalid corrective schedule')
    warmup=min(warmup,total_steps)
    if step<warmup:return maximum*(step+1)/warmup
    progress=min(1.,(step-warmup)/max(1,total_steps-1-warmup))
    return minimum+.5*(maximum-minimum)*(1+math.cos(math.pi*progress))
