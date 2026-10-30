import unittest,time
from ordered_prefetch import OrderedPrefetch
class Loader:
 def __init__(self,n=24,fail=None):self.cursor=0;self.n=n;self.fail=fail
 def state_dict(self):return {'cursor':self.cursor,'count':self.cursor}
 def load_state_dict(self,s):self.cursor=s['cursor']
 def __iter__(self):
  while self.cursor<self.n:
   if self.cursor==self.fail:raise ValueError('bad input')
   v=self.cursor;self.cursor+=1;yield v
class Tests(unittest.TestCase):
 def test_order(self):
  p=OrderedPrefetch(Loader());self.assertEqual(list(p),list(range(24)));self.assertEqual(p.count,24)
 def test_resume_uses_consumed_position(self):
  p=OrderedPrefetch(Loader(),4);it=iter(p)
  self.assertEqual([next(it) for _ in range(3)],[0,1,2]);time.sleep(.02)
  state=p.state_dict();self.assertEqual(state['cursor'],3);it.close()
  p.load_state_dict(state);self.assertEqual(list(p),list(range(3,24)))
 def test_error_reaches_consumer(self):
  p=OrderedPrefetch(Loader(fail=2));it=iter(p);self.assertEqual(next(it),0);self.assertEqual(next(it),1)
  with self.assertRaisesRegex(ValueError,'bad input'):next(it)
 def test_bounded_prefetch(self):
  p=OrderedPrefetch(Loader(100),2);it=iter(p);next(it);time.sleep(.03)
  self.assertLessEqual(p.inner.cursor,4);it.close()
if __name__=='__main__':unittest.main()
