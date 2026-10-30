import unittest
from score_table_ocr import score
from annotation_targets import resolve_annotation
from test_annotation_targets import annotation,HTML

class ScoringTests(unittest.TestCase):
    def test_identical_complete_transcription_scores_zero(self):
        a=annotation();out=score(a,resolve_annotation(a).text)
        self.assertEqual(out['whole_page']['CER']['rate'],0)
        self.assertEqual(out['tables']['cell_CER']['rate'],0)
        self.assertEqual(out['tables']['exact_cell_text_matches'],4)

    def test_missing_table_is_100_percent_cell_deletions(self):
        out=score(annotation(),'# Header\n\nAfter')
        self.assertEqual(out['tables']['predicted_tables'],0)
        self.assertEqual(out['tables']['cell_CER']['rate'],1)
        self.assertGreater(out['whole_page']['CER']['rate'],0)

    def test_prediction_placeholder_is_never_filled_from_reference(self):
        a=annotation();out=score(a,a['markdown'])
        self.assertEqual(out['tables']['unresolved_prediction_links'],1)
        self.assertEqual(out['tables']['predicted_tables'],0)
        self.assertEqual(out['tables']['cell_CER']['rate'],1)

    def test_prediction_json_uses_only_its_own_table_cells(self):
        a=annotation();wrong=annotation()
        wrong['tables'][0]['content']=HTML.replace('كتاب','wrong')
        wrong['blocks']=[]
        out=score(a,wrong)
        self.assertGreater(out['tables']['cell_CER']['rate'],0)
        self.assertEqual(out['tables']['exact_structure_matches'],1)

    def test_incomplete_prediction_json_is_explicitly_invalid(self):
        out=score(annotation(),{'markdown':'[tbl-0.html](tbl-0.html)','tables':[]})
        self.assertIsNotNone(out['tables']['prediction_annotation_error'])
        self.assertIsNone(out['tables']['cell_CER'])

    def test_invalid_prediction_html_does_not_get_a_table_score(self):
        out=score(annotation(),'<table><tr><td>unfinished')
        self.assertIsNotNone(out['tables']['prediction_table_parse_error'])
        self.assertIsNone(out['tables']['predicted_tables'])

    def test_equivalent_pipe_table_text_is_scored_as_cells(self):
        a={'markdown':'[tbl-0.html](tbl-0.html)','tables':[{'id':'tbl-0.html','content':HTML}]}
        out=score(a,'| عنوان | عدد |\n| --- | --- |\n| كتاب | 12 |')
        self.assertEqual(out['whole_page']['CER']['rate'],0)
        self.assertEqual(out['tables']['cell_CER']['rate'],0)
        self.assertEqual(out['tables']['predicted_tables'],1)

    def test_actual_insertions_are_not_deduplicated_or_clipped(self):
        a={'markdown':'[tbl-0.html](tbl-0.html)','tables':[{'id':'tbl-0.html','content':HTML}]}
        repeated=HTML.replace('كتاب','كتاب '*40)
        out=score(a,repeated)
        self.assertGreater(out['tables']['cell_CER']['rate'],1)

    def test_global_html_fence_is_accepted(self):
        a={'markdown':'[tbl-0.html](tbl-0.html)','tables':[{'id':'tbl-0.html','content':HTML}]}
        self.assertEqual(score(a,'```html\n'+HTML+'\n```')['tables']['cell_CER']['rate'],0)

if __name__=='__main__':unittest.main()
