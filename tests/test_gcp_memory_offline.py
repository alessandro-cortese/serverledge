#!/usr/bin/env python3
"""Pure local checks, no Docker/GCP and no repository writes."""
from __future__ import annotations
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'analysis/profiling'

def load(name: str):
    path = ROOT / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

prep = load('gcp_memory_prepare')
res = load('gcp_memory_analyze_lifecycle')


def event(event_type: str, ts: int, cid: str, arch: str, node: str,
          *, reason: str='expired', kind: str='cold_start') -> str:
    line = (f'2026/10/24 10:00:00 [CONTAINER_LIFECYCLE] event={event_type} ts={ts} '
            f'container_id="{cid}" function="test-fn" arch="{arch}" '
            f'machine_tag="{arch}" node="{node}" memory_mb=1024 ')
    if event_type == 'container_created':
        return line + f'creation_kind="{kind}" initial_state="busy"'
    return line + f'memory_lookup_ok=true reason="{reason}"'


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.logs = {
            'sl-x86-1': '\n'.join([
                event('container_created', 1000, 'a', 'amd64', 'sl-x86-1'),
                event('container_created', 1200, 'b', 'amd64', 'sl-x86-1'),
                event('container_destroyed', 1800, 'a', 'amd64', 'sl-x86-1'),
                event('container_destroyed', 2000, 'b', 'amd64', 'sl-x86-1'),
            ]),
            'sl-arm-1': '\n'.join([
                event('container_created', 1100, 'a', 'arm64', 'sl-arm-1'),
                event('container_destroyed', 1500, 'a', 'arm64', 'sl-arm-1'),
            ]),
        }

    def test_pairing_memory_and_peak(self):
        summary, rows = res.analyze(self.logs, 'test-fn')
        self.assertEqual(summary['pairing'], 'PASS')
        self.assertEqual(summary['created_containers'], 3)
        self.assertEqual(summary['peak_resident_containers_by_worker']['sl-x86-1'], 2)
        self.assertEqual(summary['peak_resident_containers_by_worker']['sl-arm-1'], 1)
        self.assertAlmostEqual(summary['memory_MB_s_by_arm']['amd64'], 1024 * 1.6)
        self.assertAlmostEqual(summary['memory_MB_s_by_arm']['arm64'], 1024 * .4)
        self.assertEqual(len(rows), 3)
        self.assertTrue(summary['no_optimal_arm_input'])

    def test_unpaired_is_rejected(self):
        logs = dict(self.logs)
        logs['sl-x86-1'] = '\n'.join(logs['sl-x86-1'].splitlines()[:-1])
        with self.assertRaisesRegex(ValueError, 'lifecycle incompleto'):
            res.analyze(logs, 'test-fn')

    def test_prewarm_is_rejected(self):
        logs = dict(self.logs)
        logs['sl-arm-1'] = logs['sl-arm-1'].replace('creation_kind="cold_start"', 'creation_kind="prewarm"')
        with self.assertRaisesRegex(ValueError, 'prewarm artificiale'):
            res.analyze(logs, 'test-fn')

    def test_non_expired_is_rejected(self):
        logs = dict(self.logs)
        logs['sl-arm-1'] = logs['sl-arm-1'].replace('reason="expired"', 'reason="node_shutdown"')
        with self.assertRaisesRegex(ValueError, 'distruzione'):
            res.analyze(logs, 'test-fn')

    def test_duplicate_is_rejected(self):
        logs = dict(self.logs)
        logs['sl-arm-1'] += '\n' + logs['sl-arm-1'].splitlines()[0]
        with self.assertRaisesRegex(ValueError, 'duplicato'):
            res.analyze(logs, 'test-fn')


class Preparation(unittest.TestCase):
    def test_grid_immutable(self):
        self.assertEqual(len(prep.CONFIGS), 9)
        self.assertEqual(sum(c[1] == 'difference' for c in prep.CONFIGS), 3)
        self.assertEqual(sum(c[1] == 'ratio' for c in prep.CONFIGS), 3)
        self.assertEqual(sum(c[1] == 'none' for c in prep.CONFIGS), 3)
        self.assertEqual([c[0] for c in prep.CONFIGS],
                         ['N08','D08','R08','N02','D02','R02','N04','D04','R04'])

    def test_catalog_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'scripts/gcp_scripts/functions_catalog.sh'
            p.parent.mkdir(parents=True)
            p.write_text('FUNCTIONS=(\n  "alpha|python314|foo.py|1024|foo.handler"\n)\n')
            catalog = prep.read_catalog(root)
            self.assertEqual(catalog['alpha']['memory_mb'], 1024)
            self.assertEqual(len(catalog), 1)


if __name__ == '__main__':
    unittest.main()
