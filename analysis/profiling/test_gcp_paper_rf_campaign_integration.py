#!/usr/bin/env python3
"""Test locali con servizi GCP simulati per il collector paper-RF.

Non accendono VM né modificano il runtime Serverledge.
"""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name('gcp_paper_rf_campaign.py')
spec = importlib.util.spec_from_file_location('paper_rf_campaign', MODULE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

HEADERS = ['request_id', 'function_name', 'machine_tag', 'configured_cpus'] + [f'feature_{i}' for i in range(18)]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name)
        self.cfg = dict(pin=c.PIN, schema=6, cpu=1, target=2, block=2,
                        entries=[dict(name='hashing', runtime='go125',
                                      src='functions/bundles/hashing.tar',
                                      memory_mb=1024, handler='')])
        self.events = []
        self.patches = []

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def mock(self, name, side_effect=None, return_value=None):
        p = patch.object(c, name, side_effect=side_effect, return_value=return_value)
        self.patches.append(p)
        return p.start()

    def prepare_remote(self, rotation_error=False):
        self.mock('ensure_binaries', return_value='pinned-exporter')
        self.mock('start_arch', side_effect=lambda repo, arch: self.events.append(('start', arch)) or 'sl-x86-1')
        self.mock('internal_ip', return_value='10.1.1.1')
        self.mock('worker_status', return_value=(0, 12000, 12000, 0.0))
        self.mock('raw_exists', return_value=False)
        self.mock('register', side_effect=lambda *a: self.events.append(('register', a[0]['name'])))
        self.mock('invoke', side_effect=lambda *a: self.events.append(('invoke', a[2], a[3])))
        self.mock('drain', side_effect=lambda *a: self.events.append(('drain', a[0])))
        self.mock('drain_recovery', side_effect=lambda *a: self.events.append(('recovery_drain', a[0])))
        self.mock('remote', side_effect=lambda *a, **kw: ('IDLE' if a[0] == 'sl-workload' else '0'))
        def get_raw(worker, block_id, target):
            path = target / 'raw.jsonl'
            path.write_text(''.join(
                '{"schema_version":6,"function_name":"hashing",'
                '"machine_tag":"amd64","function_configuration":{"configured_cpus":1}}\n'
                for _ in range(3)))
            return path
        self.mock('fetch_raw', side_effect=get_raw)
        self.mock('fetch_locust', return_value=None)
        def fake_export(_exporter, raw, full, selected, _id, needed):
            all_rows = [dict(request_id=f'abc-{i}', function_name='hashing',
                             machine_tag='amd64', configured_cpus='1') for i in range(2)]
            for file, rows in ((full, all_rows), (selected, all_rows[:needed])):
                with file.open('w', newline='') as f:
                    writer = csv.DictWriter(f, HEADERS)
                    writer.writeheader()
                    writer.writerows(rows)
            return all_rows, all_rows[:needed]
        self.mock('export_select', side_effect=fake_export)
        def rotate(worker, block_id, sha):
            self.events.append(('rotate', block_id))
            if rotation_error:
                raise RuntimeError('simulated remote interruption')
        return self.mock('rotate_remote', side_effect=rotate)

    def test_complete_block_and_checkpoint(self):
        self.prepare_remote()
        c.run_next(self.dest, self.cfg, c.ROOT, max_new_blocks=1)
        self.assertFalse(c.pending_path(self.dest).exists())
        self.assertEqual(c.counts(self.dest, self.cfg)['amd64', 'hashing'], 2)
        self.assertTrue((self.dest/'blocks'/'amd64'/'round_000'/'hashing'/'checkpoint.json').is_file())
        self.assertEqual([x[0] for x in self.events], ['start', 'register', 'invoke', 'drain', 'rotate'])

    def test_partial_block_blocks_another_run(self):
        c.write_json(c.pending_path(self.dest), dict(id='r000_amd64_hashing'))
        with self.assertRaisesRegex(RuntimeError, 'pending.json'):
            c.run_next(self.dest, self.cfg, c.ROOT, max_new_blocks=1)

    def test_checkpoint_recovery_after_failed_remote_rotation(self):
        rotation = self.prepare_remote(rotation_error=True)
        with self.assertRaisesRegex(RuntimeError, 'simulated remote interruption'):
            c.run_next(self.dest, self.cfg, c.ROOT, max_new_blocks=1)
        self.assertTrue(c.pending_path(self.dest).is_file())
        self.assertEqual(c.counts(self.dest, self.cfg)['amd64','hashing'], 2)
        rotation.side_effect = lambda worker, block_id, sha: self.events.append(('recovered_rotate', block_id))
        self.mock('check_running', return_value='sl-x86-1')
        c.recover(self.dest, self.cfg, c.ROOT)
        self.assertFalse(c.pending_path(self.dest).exists())
        self.assertEqual(c.counts(self.dest, self.cfg)['amd64','hashing'], 2)
        self.assertEqual(sum(ev[0]=='recovered_rotate' for ev in self.events), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
