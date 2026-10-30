import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from markdown_ocr_metrics import score,distance

class Scores(unittest.TestCase):
    def test_exact_table_zero_errors(self):
        text='عنوان\n\n| أ | ب |\n| --- | --- |\n| ج | د |'
        result=score(text,text)
        self.assertEqual(result['CER'],0);self.assertEqual(result['strict_cell_CER'],0)
        self.assertEqual(result['reference_nonempty_cells'],4)
    def test_prediction_placeholder_not_filled(self):
        result=score('| رأس |\n| --- |\n| خلية |','[tbl-0.html](tbl-0.html)')
        self.assertEqual(result['predicted_tables'],0);self.assertEqual(result['missing_nonempty_cells'],2)
        self.assertEqual(result['strict_cell_CER'],1);self.assertEqual(result['unresolved_prediction_table_links'],1)
    def test_markdown_formatting_normalized(self):
        result=score('**هذا نص**\n\nآخر','هذا نص آخر')
        self.assertEqual(result['CER'],0);self.assertEqual(result['WER'],0)
    def test_insertions_are_not_clamped(self):
        result=score('أ','ب'*10);self.assertGreater(result['CER'],1)
    def test_exact_levenshtein(self):
        self.assertEqual(distance('kitten','sitting'),3);self.assertEqual(distance([],['x']),1)

if __name__=='__main__':unittest.main()
