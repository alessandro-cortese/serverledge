#!/usr/bin/env python3
"""Local tests of the paper-RF planner and checkpoint integrity."""

import csv
import importlib.util
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name('gcp_paper_rf_campaign.py')
spec = importlib.util.spec_from_file_location('paper_rf_collector', MODULE)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dest = Path(self.temp.name)
        self.c = dict(pin=collector.PIN, schema=6, cpu=1,
                      target=2, block=1,
                      entries=[{'name': 'hashing'}, {'name': 'hashing-py'}])

    def tearDown(self):
        self.temp.cleanup()

    def make_block(self, arch, name, rnd, rid):
        path = self.dest/'blocks'/arch/f'round_{rnd:03d}'/name
        path.mkdir(parents=True, exist_ok=True)
        (path/'raw.jsonl').write_text('{"schema_version":6}\n')
        (path/'paper-rf-features.csv').write_text('request_id\n'+rid+'\n')
        (path/'selected.csv').write_text('request_id\n'+rid+'\n')
        rec = dict(pin=collector.PIN, schema=6, id=f'r{rnd}_{arch}_{name}',
                   arch=arch, function=name, round=rnd, selected_count=1,
                   selected_ids=[rid], sha256={n: collector.digest(path/n) for n
                        in ('raw.jsonl','paper-rf-features.csv','selected.csv')})
        collector.write_json(path/'checkpoint.json', rec)

    def test_alternates_order_and_counts(self):
        _, phase = collector.phases(self.dest, self.c)
        self.assertEqual(phase[:2], (0, 'amd64'))
        for name in ('hashing', 'hashing-py'):
            self.make_block('amd64', name, 0, f'xa_{name}')
        _, phase = collector.phases(self.dest, self.c)
        self.assertEqual(phase[:2], (0, 'arm64'))
        for name in ('hashing', 'hashing-py'):
            self.make_block('arm64', name, 0, f'ar_{name}')
        _, phase = collector.phases(self.dest, self.c)
        self.assertEqual(phase[:2], (1, 'arm64'))
        for name in ('hashing', 'hashing-py'):
            self.make_block('arm64', name, 1, f'ar2_{name}')
        _, phase = collector.phases(self.dest, self.c)
        self.assertEqual(phase[:2], (1, 'amd64'))
        for name in ('hashing', 'hashing-py'):
            self.make_block('amd64', name, 1, f'xa2_{name}')
        _, phase = collector.phases(self.dest, self.c)
        self.assertIsNone(phase)
        self.assertEqual(sum(collector.counts(self.dest, self.c).values()), 8)

    def test_corrupt_raw_rejected(self):
        self.make_block('amd64', 'hashing', 0, 'x1')
        f = self.dest/'blocks'/'amd64'/'round_000'/'hashing'/'raw.jsonl'
        f.write_text('tampered')
        with self.assertRaisesRegex(RuntimeError, 'SHA errato'):
            collector.completed_blocks(self.dest, self.c)

    def test_duplicate_request_id_rejected(self):
        self.make_block('amd64', 'hashing', 0, 'x1')
        self.make_block('amd64', 'hashing', 1, 'x1')
        with self.assertRaisesRegex(RuntimeError, 'ID riutilizzato'):
            collector.completed_blocks(self.dest, self.c)

    def test_exporter_validity_not_locust_count(self):
        """Con 5 righe exporter e 2 richieste residue seleziona SOLO 2."""
        headers = ['request_id','function_name','machine_tag','configured_cpus'] + [f'col{i}' for i in range(18)]
        csv_path = self.dest/'all.csv'
        chosen_path = self.dest/'selected.csv'
        raw_path = self.dest/'raw.jsonl'
        raw_path.write_text('\n')
        def fake_export(cmd, capture):
            with csv_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, headers)
                writer.writeheader()
                for i in range(5):
                    writer.writerow({'request_id':f'id-{i}', 'function_name':'hashing',
                                     'machine_tag':'amd64', 'configured_cpus':'1'})
        with patch.object(collector, 'command', side_effect=fake_export):
            rows, selected = collector.export_select('pinned-exporter',raw_path,csv_path,
                                                       chosen_path,'test',2)
        self.assertEqual(len(rows), 5)
        self.assertEqual([r['request_id'] for r in selected], ['id-0', 'id-1'])
        with chosen_path.open(newline='') as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)

    def test_export_final_requires_every_target(self):
        with self.assertRaisesRegex(RuntimeError, 'Campagna incompleta'):
            collector.export_dataset(self.dest, self.c)

    def test_pending_file_is_persistent(self):
        collector.write_json(collector.pending_path(self.dest), dict(id='r000_amd64_hashing'))
        self.assertEqual(collector.read_json(collector.pending_path(self.dest))['id'],
                         'r000_amd64_hashing')


if __name__ == '__main__':
    unittest.main(verbosity=2)
