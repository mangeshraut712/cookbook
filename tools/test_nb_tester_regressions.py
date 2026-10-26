#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for nb_tester comparator and model override edge cases."""

import ast
import dataclasses
import json
import re
import typing
import unittest
from pathlib import Path

from tools.nb_tester.model_override import ModelOverrideTransformer


def _load_comparator_classes():
    """Load comparator dataclasses without importing optional SDK deps."""
    path = Path("tools/nb_tester/comparator.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(
        vars(typing),
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        re=re,
        json=json,
        TesterConfig=object,
    )
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["OutputComparator"]


class ExactOrFuzzyRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparator = object.__new__(_load_comparator_classes())

    def _compare(self, old: str, new: str):
        return self.comparator.compare_cell(
            0,
            "print(result)",
            "exact_or_fuzzy",
            [{"output_type": "stream", "text": old}],
            [{"output_type": "stream", "text": new}],
        )

    def test_status_text_change_is_regression(self):
        report = self._compare("PASS: 1 checks", "FAIL: 1 checks")
        self.assertEqual(report.verdict, "REGRESSION")
        self.assertTrue(report.is_regression)

    def test_sign_flip_is_regression(self):
        report = self._compare("Balance: 42", "Balance: -42")
        self.assertEqual(report.verdict, "REGRESSION")
        self.assertTrue(report.is_regression)

    def test_exact_match(self):
        report = self._compare("Result: 100", "Result: 100")
        self.assertEqual(report.verdict, "MATCH")
        self.assertFalse(report.is_regression)

    def test_small_numeric_drift_is_slight_variation(self):
        report = self._compare("Tokens: 100", "Tokens: 105")
        self.assertEqual(report.verdict, "SLIGHT_VARIATION")
        self.assertFalse(report.is_regression)

    def test_nonnumeric_change_without_digits_is_regression(self):
        report = self._compare("PASS", "FAIL")
        self.assertEqual(report.verdict, "REGRESSION")
        self.assertTrue(report.is_regression)


class ModelOverrideStringLiteralTest(unittest.TestCase):
    def test_preserves_assignment_like_text_inside_multiline_prompt(self):
        source = '''prompt = """Translate this Python example without changing it:
MODEL_ID = 'gemini-2.5-flash'
"""
MODEL_ID = "gemini-2.5-flash"
'''
        transformed, count = ModelOverrideTransformer(
            "gemini-3.7-flash"
        ).transform_cell_source(source)

        before, after = {}, {}
        exec(source, before)
        exec(transformed, after)

        self.assertEqual(count, 1)
        self.assertEqual(before["prompt"], after["prompt"])
        self.assertEqual(after["MODEL_ID"], "gemini-3.7-flash")

    def test_preserves_assignment_like_text_inside_docstring(self):
        source = '''def demo():
    """Example:
MODEL_ID = "gemini-2.5-flash"
"""
    MODEL_ID = "gemini-2.5-flash"
    return MODEL_ID
'''
        transformed, count = ModelOverrideTransformer(
            "gemini-3.7-flash"
        ).transform_cell_source(source)
        self.assertEqual(count, 1)
        self.assertIn('MODEL_ID = "gemini-2.5-flash"', transformed.split('"""')[1])
        self.assertIn('MODEL_ID = "gemini-3.7-flash"', transformed)


if __name__ == "__main__":
    unittest.main()
