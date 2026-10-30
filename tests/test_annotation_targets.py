import copy
import unittest
from annotation_targets import resolve_annotation, parse_tables, TableTargetError, assert_no_table_placeholders

HTML='<table><tbody><tr><th>عنوان</th><th>عدد</th></tr><tr><td>كتاب</td><td>12</td></tr></tbody></table>'

def annotation():
    return {'markdown':'# Header\n\n[tbl-0.html](tbl-0.html)\n\nAfter',
            'tables':[{'id':'tbl-0.html','format':'html','content':HTML}],
            'blocks':[{'type':'table','content':HTML,'reading_index':2,'top_left_x':1,'top_left_y':2,'bottom_right_x':3,'bottom_right_y':4}]}

class TargetsTests(unittest.TestCase):
    def test_complete_cells_replace_reference_in_place(self):
        a=annotation();before=copy.deepcopy(a);out=resolve_annotation(a)
        self.assertEqual(out.text,'# Header\n\n'+HTML+'\n\nAfter')
        self.assertEqual(out.audit['table_count'],1)
        self.assertEqual(out.audit['cell_count'],4)
        self.assertTrue(out.audit['blocks_order_checked'])
        self.assertEqual(a,before)

    def test_original_no_table_text_is_unchanged(self):
        text='Heading\n\n**Arabic**: نصّ\n![image](img-0)\n[normal](https://example.com)'
        self.assertEqual(resolve_annotation({'markdown':text,'tables':[]}).text,text)

    def test_two_tables_stay_in_markdown_and_blocks_order(self):
        a=annotation();second=HTML.replace('كتاب','قلم')
        a['markdown']='first [tbl-0.html](tbl-0.html) middle [tbl-1.html](tbl-1.html) end'
        a['tables'].append({'id':'tbl-1.html','content':second})
        a['blocks'].append({'type':'table','content':second,'reading_index':8})
        out=resolve_annotation(a)
        self.assertEqual(out.text,'first '+HTML+' middle '+second+' end')
        self.assertEqual(out.audit['resolved_table_ids'],['tbl-0.html','tbl-1.html'])

    def test_distinct_tables_with_identical_contents_keep_their_positions(self):
        other=HTML.replace('كتاب','قلم')
        a={'markdown':'[tbl-0.html](tbl-0.html) middle [tbl-1.html](tbl-1.html) end [tbl-2.html](tbl-2.html)',
           'tables':[{'id':'tbl-0.html','content':HTML},{'id':'tbl-1.html','content':other},{'id':'tbl-2.html','content':HTML}],
           'blocks':[{'type':'table','content':HTML,'reading_index':1},{'type':'table','content':other,'reading_index':2},{'type':'table','content':HTML,'reading_index':3}]}
        out=resolve_annotation(a)
        self.assertEqual(out.text,HTML+' middle '+other+' end '+HTML)
        self.assertEqual(out.audit['resolved_table_ids'],['tbl-0.html','tbl-1.html','tbl-2.html'])
        self.assertEqual(out.audit['cell_count'],12)

    def test_two_already_inline_identical_tables_are_preserved(self):
        a={'markdown':HTML+' between '+HTML,'tables':[{'id':'tbl-0.html','content':HTML},{'id':'tbl-1.html','content':HTML}]}
        self.assertEqual(resolve_annotation(a).text,a['markdown'])

    def test_missing_reference_is_refused(self):
        with self.assertRaisesRegex(TableTargetError,'missing_table_id'):
            resolve_annotation({'markdown':'[tbl-3.html](tbl-3.html)','tables':[]})

    def test_table_without_position_is_not_silently_dropped_or_appended(self):
        a=annotation();a['markdown']='body only'
        with self.assertRaisesRegex(TableTargetError,'unreferenced'):
            resolve_annotation(a)

    def test_duplicate_ids_and_references_are_refused(self):
        a=annotation();a['tables']*=2
        with self.assertRaisesRegex(TableTargetError,'duplicate_table_id'):resolve_annotation(a)
        a=annotation();a['markdown']+='\n[tbl-0.html](tbl-0.html)'
        with self.assertRaisesRegex(TableTargetError,'duplicate_table_reference'):resolve_annotation(a)

    def test_already_inline_table_is_not_duplicated(self):
        a=annotation();a['markdown']='Before\n'+HTML+'\nAfter'
        self.assertEqual(resolve_annotation(a).text,a['markdown'])

    def test_code_and_external_links_are_not_internal_references(self):
        text='`[tbl-0.html](tbl-0.html)`\n```md\n[tbl-0.html](tbl-0.html)\n```\n[tbl-0.html](https://example.com/tbl-0.html)'
        assert_no_table_placeholders(text)
        self.assertEqual(resolve_annotation({'markdown':text}).text,text)

    def test_angle_and_local_relative_links(self):
        a=annotation();a['markdown']='[table](<./tbl-0.html>)'
        self.assertEqual(resolve_annotation(a).text,HTML)

    def test_merged_cell_attributes_are_kept(self):
        content='<table><tr><td rowspan="2">A</td><td colspan="2">B<br>C</td></tr><tr><td>D</td><td>E</td></tr></table>'
        a={'markdown':'[tbl-0.html](tbl-0.html)','tables':[{'id':'tbl-0.html','content':content}]}
        self.assertEqual(resolve_annotation(a).text,content)
        table=parse_tables(content)[0]
        self.assertEqual(table['rows'][0][0]['rowspan'],2)
        self.assertEqual(table['rows'][0][1]['colspan'],2)
        self.assertEqual(table['rows'][0][1]['text'],'B C')

    def test_html_rowspan_zero_is_preserved(self):
        content='<table><tr><td rowspan="0">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
        a={'markdown':'[tbl-0.html](tbl-0.html)','tables':[{'id':'tbl-0.html','content':content}]}
        self.assertEqual(resolve_annotation(a).text,content)
        self.assertEqual(parse_tables(content)[0]['rows'][0][0]['rowspan'],0)

    def test_incomplete_html_is_refused(self):
        a=annotation();a['tables'][0]['content']='<table><tr><td>lost'
        with self.assertRaises(TableTargetError):resolve_annotation(a)

    def test_blocks_cannot_hide_an_unlisted_table(self):
        a=annotation();a['blocks'].append({'type':'table','content':HTML.replace('كتاب','unknown'),'reading_index':3})
        with self.assertRaisesRegex(TableTargetError,'table_block_missing'):resolve_annotation(a)

    def test_conflicting_blocks_order_is_refused(self):
        a=annotation();second=HTML.replace('كتاب','قلم')
        a['markdown']='[tbl-1.html](tbl-1.html) [tbl-0.html](tbl-0.html)'
        a['tables'].append({'id':'tbl-1.html','content':second})
        a['blocks'].append({'type':'table','content':second,'reading_index':3})
        with self.assertRaisesRegex(TableTargetError,'order_conflicts'):resolve_annotation(a)

    def test_active_html_is_not_rendered_or_accepted(self):
        a=annotation();a['tables'][0]['content']='<table><tr><td><script>alert(1)</script></td></tr></table>'
        with self.assertRaisesRegex(TableTargetError,'active_or_hidden_html'):resolve_annotation(a)

if __name__=='__main__':unittest.main()
