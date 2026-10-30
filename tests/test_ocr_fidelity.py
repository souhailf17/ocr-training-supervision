import importlib.util
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'scripts/ocr_fidelity.py'
spec=importlib.util.spec_from_file_location('fidelity',path)
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

class FidelityTests(unittest.TestCase):
 def test_unicode_is_never_rewritten(self):
  text='Échéance œ € ≤ ≥ ± − ١٢٣ ۱۲۳ 123، مُحَمَّد'
  r=f.inspect_output(text,'mixed');self.assertFalse(r['text_rewritten'])
  self.assertEqual(r['digit_glyph_counts'],{'latin':3,'arabic_indic':3,'eastern_arabic_indic':3})
 def test_multiline_invoice_loop_is_detected(self):
  block='INV-11118\nفاتورة إلى: رانيا\nالمجموع 1,750.00\nالإجمالي: 3,500.00\n'
  self.assertIn('repeated_line_cycle',f.repetition_flags(block*3))
 def test_long_punctuation_loop_is_detected(self):
  self.assertIn('punctuation_run',f.repetition_flags('Texte correct\n'+'.'*2000))
 def test_mixed_page_not_rejected_for_arabic(self):
  text='République — المملكة المغربية'
  self.assertNotIn('unexpected_arabic_text',f.inspect_output(text,'mixed')['flags'])
  self.assertIn('unexpected_arabic_text',f.inspect_output(text,'fr')['flags'])
 def test_truncation_never_counts_as_complete(self):
  self.assertTrue(f.inspect_output('Texte','fr','token_limit')['requires_review'])
 def test_digit_conversion_is_reported_not_applied(self):
  report=f.digit_substitutions('رقم ١٢٣ ثم ۱۲','رقم 123 ثم 12')
  self.assertEqual(len(report['aligned_same_value_different_glyphs']),2)
 def test_legitimate_short_repetitions_are_retained(self):
  self.assertEqual(f.repetition_flags('0 0 0 0\nOui\nOui\nOui'),[])
 def test_prompts_preserve_visible_bilingual_material(self):
  self.assertIn('Latin text actually visible',f.literal_prompt('OCR','ar'))
  with self.assertRaises(ValueError):f.literal_prompt('OCR','unknown')
 def test_streaming_stop_keeps_reason(self):
  guard=f.StreamingGuard('fr')
  self.assertFalse(guard.check('Procès-verbal'))
  self.assertTrue(guard.check('Procès-verbal\nمرحبا بكم'))
  self.assertEqual(guard.reason,'unexpected_arabic_text')
  self.assertTrue(guard.check(''))
 def test_training_prompt_has_not_drifted(self):
  import ast
  t=ast.parse(path.with_name('markdown_table_targets.py').read_text(encoding='utf-8'))
  original=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='PROMPT' for x in n.targets))
  self.assertEqual(f.TRAINING_MARKDOWN_PROMPT,original)
 def test_only_uniform_white_is_blank(self):
  from PIL import Image
  image=Image.new('RGB',(64,64),'white');self.assertTrue(f.is_exact_white_blank(image))
  image.putpixel((32,32),(254,254,254));self.assertFalse(f.is_exact_white_blank(image))
  self.assertFalse(f.is_exact_white_blank(Image.new('RGB',(64,64),'black')))

if __name__=='__main__':unittest.main()
