import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from markdown_table_targets import convert_table,resolve_markdown,TableTargetError

class Conversion(unittest.TestCase):
    def test_arabic_header_order(self):
        md,a=convert_table('<table><tr><th>يمين</th><th>يسار</th></tr><tr><td>أ</td><td>ب</td></tr></table>')
        self.assertEqual(md,'| يمين | يسار |\n| --- | --- |\n| أ | ب |')
        self.assertFalse(a['synthetic_empty_header']);self.assertEqual(a['source_cells'],4)
    def test_data_row_not_promoted(self):
        md,a=convert_table('<table><tr><td>واحد</td></tr></table>')
        self.assertEqual(md,'|  |\n| --- |\n| واحد |');self.assertTrue(a['synthetic_empty_header'])
    def test_spans_text_once(self):
        md,a=convert_table('<table><tr><td rowspan="2" colspan="2">بداية</td><td>س</td></tr><tr><td>نهاية</td></tr></table>')
        self.assertEqual(md.count('بداية'),1);self.assertEqual(a['columns'],3)
        self.assertEqual(a['anchors'][-1]['column'],2);self.assertEqual(a['source_cells'],3)
    def test_rowspan_zero_scoped_to_tbody(self):
        md,a=convert_table('<table><tbody><tr><td rowspan="0">أ</td><td>ب</td></tr><tr><td>ج</td></tr></tbody><tbody><tr><td>د</td></tr></tbody></table>')
        self.assertEqual(a['anchors'][0]['rowspan'],2);self.assertEqual(a['anchors'][-1]['column'],0)
    def test_cross_group_span_rejected(self):
        with self.assertRaises(TableTargetError):convert_table('<table><thead><tr><th rowspan="2">أ</th></tr></thead><tbody><tr><td>ب</td></tr></tbody></table>')
    def test_literals_caption_line_break(self):
        md,a=convert_table('<table><caption>عنوان</caption><tr><td>أ | ب<br>ج &amp; &lt;س&gt; *ن*</td></tr></table>')
        self.assertIn(r'أ \| ب ج &amp; &lt;س&gt; \*ن\*',md);self.assertEqual(a['caption'],'عنوان')
    def test_positions_and_no_table_control(self):
        t='<table><tr><td>خلية</td></tr></table>'
        out=resolve_markdown({'markdown':'قبل\n[tbl-0.html](tbl-0.html)\nبعد','tables':[{'id':'tbl-0.html','content':t}]})
        self.assertTrue(out.text.startswith('قبل'));self.assertTrue(out.text.endswith('بعد'))
        self.assertEqual(out.audit['cell_count'],1);self.assertNotIn('tbl-0.html',out.text)
        self.assertEqual(resolve_markdown({'markdown':'نص عادي','tables':[]}).text,'نص عادي')
    def test_identical_distinct_tables(self):
        t='<table><tr><td>أ</td></tr></table>'
        out=resolve_markdown({'markdown':'[tbl-0.html](tbl-0.html)\n[tbl-1.html](tbl-1.html)',
                              'tables':[{'id':'tbl-0.html','content':t},{'id':'tbl-1.html','content':t}]})
        self.assertEqual(out.audit['table_count'],2);self.assertEqual(len(out.audit['tables']),2)
    def test_missing_reference_fails(self):
        with self.assertRaises(TableTargetError):resolve_markdown({'markdown':'[tbl-0.html](tbl-0.html)','tables':[]})

if __name__=='__main__':unittest.main()
