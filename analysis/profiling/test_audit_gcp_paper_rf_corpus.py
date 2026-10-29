#!/usr/bin/env python3
"""Test senza accesso GCP del controllo statico del corpus."""

from __future__ import annotations

import io
import tarfile
import unittest

import audit_gcp_paper_rf_corpus as audit


def entry(name, runtime, src, handler=''):
    return dict(name=name, runtime=runtime, src=src, memory_mb=1024, handler=handler)


def sample_tar():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as tar:
        content = b'package main\n'
        ti = tarfile.TarInfo('main.go')
        ti.size = len(content)
        tar.addfile(ti, io.BytesIO(content))
    return output.getvalue()


class StaticAuditTests(unittest.TestCase):
    def test_valid_python_handler(self):
        failures, warnings = audit.inspect_source(entry('hashing-py','python314','hashing.py','hashing.handler'), b'def handler(event, context):\n return 1\n')
        self.assertEqual((failures, warnings), ([], []))

    def test_wrong_python_module_fails(self):
        failures, _ = audit.inspect_source(entry('n','python314','hashing.py','wrong.handler'), b'def handler(event, context):\n pass\n')
        self.assertTrue(failures)

    def test_missing_handler_review(self):
        failures, warnings = audit.inspect_source(entry('n','python314','a.py','a.handler'), b'def other():\n pass\n')
        self.assertFalse(failures)
        self.assertTrue(warnings)

    def test_valid_tar(self):
        self.assertEqual(audit.inspect_source(entry('g','go125','bundle.tar'), sample_tar()), ([], []))

    def test_invalid_tar_fails(self):
        failures, _ = audit.inspect_source(entry('g','go125','bundle.tar'), b'not a tar')
        self.assertTrue(failures)

    def test_missing_file_fails(self):
        failures, _ = audit.inspect_source(entry('g','go125','bundle.tar'), None)
        self.assertTrue(failures)

    def test_node_symbol(self):
        self.assertEqual(audit.inspect_source(entry('n','nodejs17ng','x.js','handler'), b'exports.handler = async () => 1;'), ([], []))

    def test_report_counts(self):
        entries=[entry('a','python314','a.py','a.handler'), entry('b','python314','b.py','b.handler')]
        values={'a.py': b'def handler():\n pass\n', 'b.py': None}
        report=audit.audit(entries, lambda p: values[p])
        self.assertEqual((report['functions'],report['passed'],report['failed']), (2,1,1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
