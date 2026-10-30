import unittest
from unlimited_format import decode
class FormatTests(unittest.TestCase):
    def test_structural_coordinates_are_not_recognized_text(self):
        d=decode('<PAGE><BLOCK>text[1, 2, 900, 990]<FORK>١٢٣ ۱۲۳ œ − €')
        self.assertEqual(d['text'],'١٢٣ ۱۲۳ œ − €');self.assertEqual(d['blocks'][0]['box'],[1,2,900,990]);self.assertFalse(d['flags'])
    def test_all_printed_headers_footers_and_uncertainty_remain(self):
        d=decode('<PAGE><BLOCK>header[0,0,9,9]<FORK>En-tête<BLOCK>text[0,10,20,30]<FORK>[unreadable]<BLOCK>footer[0,40,90,99]<FORK>Page 7')
        self.assertEqual(d['text'],'En-tête\n\n[unreadable]\n\nPage 7')
    def test_unparsed_material_never_disappears(self):
        d=decode('Invented text<PAGE><BLOCK>text[3, 4<FORK>MORE')
        self.assertIn('Invented text',d['text']);self.assertIn('MORE',d['text']);self.assertIn('unparsed_block_header',d['flags'])
    def test_both_documented_level_orders_and_fullwidth_commas(self):
        d=decode('<PAGE><BLOCK>title<LEVEL2>[1，2，3，4]<FORK>أ<BLOCK>title[5,6,7,8]<LEVEL1><FORK>B')
        self.assertEqual(d['text'],'أ\n\nB');self.assertEqual(len(d['blocks']),2)
    def test_tables_and_digit_rows_preserve_order(self):
        d=decode('<PAGE><BLOCK>table[0,0,100,100]<FORK><table><tr><td>١٠</td><td>10</td></tr></table>')
        self.assertEqual(d['text'],'<table><tr><td>١٠</td><td>10</td></tr></table>')
if __name__=='__main__':unittest.main()
