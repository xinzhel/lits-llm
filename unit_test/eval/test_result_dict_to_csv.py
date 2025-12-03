"""
Unit tests for ResultDictToCSV class.

Tests the CSV result saver functionality including:
- Immediate write-through on append
- Header generation from first result
- Column exclusion
- Loading existing results
"""

import os
import tempfile
import shutil
import unittest
import csv
from lits.eval import ResultDictToCSV


class TestResultDictToCSV(unittest.TestCase):
    """Test suite for ResultDictToCSV class."""
    
    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_basic_append_and_load(self):
        """Test basic append and load functionality."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        
        # Append some results
        saver.append_result({'idx': 0, 'answer': 'yes', 'score': 0.95})
        saver.append_result({'idx': 1, 'answer': 'no', 'score': 0.87})
        
        # Verify file exists
        self.assertTrue(os.path.exists(saver.filepath))
        
        # Load and verify (pandas loads numbers as numbers, not strings)
        loaded = saver.load_results(saver.filepath)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]['idx'], 0)  # pandas preserves types
        self.assertEqual(loaded[0]['answer'], 'yes')
        self.assertEqual(loaded[1]['answer'], 'no')
    
    def test_immediate_write_through(self):
        """Test that results are immediately written to file."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        
        # Append first result
        saver.append_result({'idx': 0, 'value': 'first'})
        
        # Read file directly to verify immediate write
        with open(saver.filepath, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)  # Header + 1 data row
        
        # Append second result
        saver.append_result({'idx': 1, 'value': 'second'})
        
        # Read again
        with open(saver.filepath, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)  # Header + 2 data rows
    
    def test_exclude_columns(self):
        """Test column exclusion functionality."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True,
            exclude_columns=['trajectory', 'verbose_info']
        )
        
        # Append result with excluded columns
        saver.append_result({
            'idx': 0,
            'answer': 'yes',
            'trajectory': 'long text...',
            'verbose_info': 'more text...',
            'score': 0.95
        })
        
        # Verify excluded columns are not in CSV using pandas
        import pandas as pd
        df = pd.read_csv(saver.filepath)
        columns = list(df.columns)
        
        self.assertIn('idx', columns)
        self.assertIn('answer', columns)
        self.assertIn('score', columns)
        self.assertNotIn('trajectory', columns)
        self.assertNotIn('verbose_info', columns)
    
    def test_append_without_override(self):
        """Test appending to existing file without override."""
        # Create initial file
        saver1 = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        saver1.append_result({'idx': 0, 'value': 'first'})
        
        # Create new saver without override (should append)
        saver2 = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=False
        )
        saver2.append_result({'idx': 1, 'value': 'second'})
        
        # Verify both results exist
        loaded = saver2.load_results(saver2.filepath)
        self.assertEqual(len(loaded), 2)
    
    def test_override_existing_file(self):
        """Test overriding existing file."""
        # Create initial file
        saver1 = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        saver1.append_result({'idx': 0, 'value': 'first'})
        
        # Create new saver with override (should clear file)
        saver2 = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        saver2.append_result({'idx': 1, 'value': 'second'})
        
        # Verify only new result exists
        loaded = saver2.load_results(saver2.filepath)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['idx'], '1')
    
    def test_consistent_columns(self):
        """Test that columns remain consistent across appends."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        
        # First result defines columns
        saver.append_result({'idx': 0, 'col_a': 'a1', 'col_b': 'b1'})
        
        # Second result has extra column (should be ignored)
        saver.append_result({'idx': 1, 'col_a': 'a2', 'col_b': 'b2', 'col_c': 'c2'})
        
        # Third result missing a column (should be empty string)
        saver.append_result({'idx': 2, 'col_a': 'a3'})
        
        # Verify columns
        with open(saver.filepath, 'r') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            rows = list(reader)
        
        self.assertEqual(columns, ['idx', 'col_a', 'col_b'])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2]['col_b'], '')  # Missing column filled with empty string
    
    def test_empty_results(self):
        """Test behavior with no results appended."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        
        # File should exist but be empty
        self.assertTrue(os.path.exists(saver.filepath))
        self.assertEqual(os.path.getsize(saver.filepath), 0)
        
        # Loading should return empty list
        loaded = saver.load_results(saver.filepath)
        self.assertEqual(len(loaded), 0)
    
    def test_update_column(self):
        """Test updating specific columns for specific rows."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        
        # Add initial results
        for i in range(3):
            saver.append_result({
                'idx': i,
                'eval1': 'yes' if i % 2 == 0 else 'no',
                'score': 0.8 + i * 0.1
            })
        
        # Update row with idx=1 to add new evaluation perspective
        saver.update_column(
            row_identifier='idx',
            identifier_value=1,
            column_updates={'eval2': 'yes', 'new_score': 0.95}
        )
        
        # Load and verify (pandas preserves types)
        loaded = saver.load_results(saver.filepath)
        self.assertEqual(len(loaded), 3)
        
        # Check that row 1 has new columns
        row1 = next(r for r in loaded if r['idx'] == 1)
        self.assertEqual(row1['eval2'], 'yes')
        self.assertEqual(row1['new_score'], 0.95)
        
        # Check that other rows have NaN for new columns (pandas behavior)
        import pandas as pd
        row0 = next(r for r in loaded if r['idx'] == 0)
        self.assertTrue(pd.isna(row0.get('eval2')) or row0.get('eval2') == '')
    
    def test_update_columns_batch(self):
        """Test batch updating multiple rows."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        
        # Add initial results
        for i in range(5):
            saver.append_result({
                'idx': i,
                'eval1': 'yes',
                'score': 0.8
            })
        
        # Batch update multiple rows
        updates = [
            {'identifier_value': 0, 'column_updates': {'eval2': 'yes', 'eval3': 'no'}},
            {'identifier_value': 2, 'column_updates': {'eval2': 'no', 'eval3': 'yes'}},
            {'identifier_value': 4, 'column_updates': {'eval2': 'yes', 'eval3': 'yes'}},
        ]
        saver.update_columns_batch(row_identifier='idx', updates=updates)
        
        # Load and verify (pandas preserves types)
        loaded = saver.load_results(saver.filepath)
        self.assertEqual(len(loaded), 5)
        
        # Check updated rows
        row0 = next(r for r in loaded if r['idx'] == 0)
        self.assertEqual(row0['eval2'], 'yes')
        self.assertEqual(row0['eval3'], 'no')
        
        row2 = next(r for r in loaded if r['idx'] == 2)
        self.assertEqual(row2['eval2'], 'no')
        self.assertEqual(row2['eval3'], 'yes')
        
        # Check non-updated rows have NaN for new columns
        import pandas as pd
        row1 = next(r for r in loaded if r['idx'] == 1)
        self.assertTrue(pd.isna(row1.get('eval2')) or row1.get('eval2') == '')
    
    def test_incremental_evaluation_workflow(self):
        """Test the incremental evaluation workflow."""
        saver = ResultDictToCSV(
            run_id='test',
            root_dir=self.test_dir,
            override=True
        )
        
        # First run: evaluate with perspective 1
        for i in range(3):
            saver.append_result({
                'idx': i,
                'question': f'Q{i}',
                'eval_perspective_1': 'yes' if i % 2 == 0 else 'no'
            })
        
        # Second run: add perspective 2 without re-evaluating perspective 1
        for i in range(3):
            saver.update_column(
                row_identifier='idx',
                identifier_value=i,
                column_updates={'eval_perspective_2': 'yes' if i > 0 else 'no'}
            )
        
        # Third run: add perspective 3
        for i in range(3):
            saver.update_column(
                row_identifier='idx',
                identifier_value=i,
                column_updates={'eval_perspective_3': 'yes'}
            )
        
        # Verify all perspectives are present
        loaded = saver.load_results(saver.filepath)
        self.assertEqual(len(loaded), 3)
        
        for row in loaded:
            self.assertIn('eval_perspective_1', row)
            self.assertIn('eval_perspective_2', row)
            self.assertIn('eval_perspective_3', row)
        
        # Verify values are correct
        row0 = next(r for r in loaded if r['idx'] == '0')
        self.assertEqual(row0['eval_perspective_1'], 'yes')
        self.assertEqual(row0['eval_perspective_2'], 'no')
        self.assertEqual(row0['eval_perspective_3'], 'yes')


if __name__ == '__main__':
    unittest.main()
