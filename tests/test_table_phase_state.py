import unittest
from table_phase_state import PhaseCursor,phase_lr

class TablePhaseStateTests(unittest.TestCase):
    def test_resume_preserves_remaining_unique_order(self):
        source=PhaseCursor(67)
        first=[source.next_index() for _ in range(32)];state=source.state_dict()
        remaining=[source.next_index() for _ in range(35)]
        restored=PhaseCursor(67);restored.load_state_dict(state)
        self.assertEqual([restored.next_index() for _ in range(35)],remaining)
        self.assertEqual(set(first+remaining),set(range(67)))
        self.assertEqual(len(set(first+remaining)),67)
        with self.assertRaises(StopIteration):restored.next_index()
    def test_changed_dataset_and_invalid_cursor_refused(self):
        cursor=PhaseCursor(32)
        for state in [{'cursor':0,'total':33,'seed':42},{'cursor':33,'total':32,'seed':42},{'cursor':0,'total':32,'seed':43}]:
            with self.assertRaises(ValueError):cursor.load_state_dict(state)
    def test_schedule_is_stable_after_resume_and_bounded(self):
        self.assertEqual(phase_lr(0,128),2e-5/8)
        self.assertAlmostEqual(phase_lr(7,128),2e-5)
        self.assertAlmostEqual(phase_lr(127,128),1e-5)
        self.assertEqual([phase_lr(i,128) for i in range(32,128)],[phase_lr(i,128) for i in range(128)][32:])
        self.assertTrue(all(0<phase_lr(i,128)<=2e-5 for i in range(128)))

if __name__=='__main__':unittest.main()
