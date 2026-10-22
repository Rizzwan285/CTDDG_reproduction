"""CPU regression checks for the executable notebook, without touching run artifacts.

Run with the project's torch/RDKit environment:
    python code/test_cdgcn_fixed_validation.py
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'code/CDGCN_stage1_fixed_validation.ipynb'
CELLS = json.loads(NOTEBOOK.read_text())['cells']


def execute(ns, *indices):
    for i in indices:
        exec(compile(''.join(CELLS[i]['source']), f'{NOTEBOOK}:cell-{i}', 'exec'), ns)


class NotebookChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ns = {}
        with patch.dict(os.environ, {'CTDDG_REPO_ROOT': str(ROOT),
                                    'CTDDG_CHECKPOINT_DIR': cls.tmp.name}):
            execute(cls.ns, 1, 2)
        cls.ns.update(CTX=torch.device('cpu'), GRACEFUL_SIGNALS=False)
        execute(cls.ns, 3, 5, 6, 7, 9, 10, 13, 14)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_model(self, dropout=0.0, full=False):
        n = self.ns
        cfg = dict(F_e=4, F_h=[8, 8], F_skip=8, F_c=[16],
                   Fh_policy=8, activation='relu', N_rnn=2,
                   normalization='layer', dropout=dropout)
        if full:
            cfg.update(F_e=n['F_E'], F_h=n['F_H'], F_skip=n['F_SKIP'],
                       F_c=n['F_C'], Fh_policy=n['FH_POLICY'], N_rnn=n['N_RNN'])
        model = n['VanillaMolGen_RNN'](n['get_mol_spec']().num_atom_types,
                                     n['get_mol_spec']().num_bond_types, D=2, **cfg)
        n['init_weights_xavier'](model)
        return model

    def batch(self, records):
        return self.ns['batch_to_tensor'](
            self.ns['MolRNNCollate'](k=2, validation_seed=123)(records))

    def test_full_model_is_batch_independent_and_train_eval_consistent(self):
        model = self.make_model(full=True)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules()))
        model.mode = 'likelihood'
        alone = self.batch([('CCO', 0)])
        together = self.batch([('CCO', 0), ('c1ccccc1N', 1)])
        with torch.no_grad():
            model.eval()
            a, b = model(*alone).clone(), model(*together).clone()
            model.train()
            c = model(*together).clone()
        torch.testing.assert_close(a[0], b[0], atol=2e-5, rtol=1e-6)
        torch.testing.assert_close(b, c)

    def test_objective_gradients_dropout_and_label_independence(self):
        model = self.make_model(dropout=0.1)
        batch = self.batch([('CCO', 0), ('CCN', 1)])
        model.eval()
        loss = model(*batch)
        torch.testing.assert_close(loss.detach(), -model.last_l.mean())
        flipped = list(batch)
        flipped[11] = 1 - batch[11]
        torch.testing.assert_close(loss, model(*flipped))
        torch.testing.assert_close(loss, model(*batch))
        model.train()
        train_a, train_b = model(*batch), model(*batch)
        self.assertNotEqual(float(train_a.detach()), float(train_b.detach()))
        train_b.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3, error_if_nonfinite=True)
        self.assertTrue(torch.isfinite(norm))
        clipped = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()
                                 if p.grad is not None))
        self.assertLessEqual(float(clipped), 3.00001)

    def test_validation_covers_remainder_and_rejects_missing_records(self):
        records = [('CCO', 0), ('CCN', 1), ('CCC', 0)]
        sampler = self.ns['BalancedSampler']([3, 3, 3], 2, shuffle=False, drop_last=False)
        self.assertEqual(sorted(i for batch in sampler for i in batch), [0, 1, 2])
        self.assertEqual(len(sampler), 2)
        small = self.ns['BalancedSampler']([1], 8, shuffle=False, drop_last=False)
        self.assertEqual(list(small), [[0]])
        collate = self.ns['MolRNNCollate'](k=2, validation_seed=123)
        loader = torch.utils.data.DataLoader(records, batch_sampler=sampler, collate_fn=collate)
        model = self.make_model(dropout=0.1)
        model.train()
        a = self.ns['evaluate'](model, loader)
        b = self.ns['evaluate'](model, loader)
        self.assertEqual(a, b)
        self.assertEqual(a[-1], 3)
        self.assertTrue(model.training)
        self.assertEqual(model.mode, 'loss')
        broken = torch.utils.data.DataLoader([('invalid', 0)], collate_fn=collate)
        with self.assertRaisesRegex(RuntimeError, 'covered 0/1'):
            self.ns['evaluate'](model, broken)

    def test_selection_patience_and_nonfinite_validation(self):
        monitor = self.ns['ValidationMonitor'](patience=3, min_delta=0.05)
        self.assertTrue(monitor.update(30.0, 0))
        self.assertTrue(monitor.update(29.98, 1))  # save small minima, don't reset patience
        self.assertEqual(monitor.bad_checks, 1)
        self.assertTrue(monitor.update(29.90, 2))
        self.assertEqual(monitor.bad_checks, 0)
        for step in (3, 4, 5):
            self.assertFalse(monitor.update(31.0, step))
        self.assertTrue(monitor.should_stop)
        restored = self.ns['ValidationMonitor'](3, 0.05)
        restored.load_state_dict(monitor.state_dict())
        self.assertEqual(restored.state_dict(), monitor.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            monitor.update(float('nan'), 6)

    def test_plateau_reduces_learning_rate_before_stopping(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2,
            threshold=0.05, threshold_mode='abs', min_lr=1e-6)
        monitor = self.ns['ValidationMonitor'](8, 0.05)
        for step, score in enumerate([30.0] + [31.0] * 8):
            monitor.update(score, step)
            scheduler.step(score)
            if step == 3:
                self.assertEqual(optimizer.param_groups[0]['lr'], 1.5e-4)
                self.assertFalse(monitor.should_stop)
        self.assertTrue(monitor.should_stop)
        self.assertEqual(optimizer.param_groups[0]['lr'], 7.5e-5)

    def test_notebook_preflight_training_resume_and_best_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'sample.csv'
            pd.read_csv(ROOT / 'datasets/data/chembl/chembl_pretrain_labelled.csv',
                        nrows=1056).to_csv(csv_path, index=False)
            ns = {}
            env = {'CTDDG_REPO_ROOT': str(ROOT), 'CTDDG_CHECKPOINT_DIR': tmp,
                   'CTDDG_DATA_PATH': str(csv_path)}
            with patch.dict(os.environ, env):
                execute(ns, 1, 2)
            ns.update(CTX=torch.device('cpu'), VERIFY_DATA_HASH=False, NUM_WORKERS=0,
                      GRACEFUL_SIGNALS=False, VAL_MOLECULES=9, TRAIN_EVAL_MOLECULES=9,
                      K_PATHS=2, VAL_K_PATHS=2, F_E=4, F_H=[8, 8], F_SKIP=8,
                      F_C=[16], FH_POLICY=8, N_RNN=2, VAL_EVERY=1,
                      SUMMARY_STEP=1, CHECKPOINT_STEP=1, TOTAL_ITERATIONS=2)
            execute(ns, *range(3, len(CELLS)))  # preflight, no training artifacts
            self.assertFalse(Path(ns['LOG_FILE']).exists())
            self.assertFalse(Path(ns['BEST_PARAMS_FILE']).exists())
            ns['RUN_TRAINING'] = True
            execute(ns, 15)
            history = pd.read_csv(ns['VAL_LOG_FILE'], sep='\t')
            self.assertEqual(history['n_molecules'].tolist(), [9, 9, 9])
            best = torch.load(ns['BEST_PARAMS_FILE'], weights_only=False)
            selected_step = int(history.loc[history.val_nll.idxmin(), 'step'])
            self.assertEqual(best['global_counter'], selected_step)
            saved_monitor = ns['monitor'].state_dict()
            saved_scheduler = ns['scheduler'].state_dict()
            execute(ns, 11)
            self.assertEqual(ns['global_counter'], 2)
            self.assertEqual(ns['monitor'].state_dict(), saved_monitor)
            self.assertEqual(ns['scheduler'].state_dict(), saved_scheduler)
            ns['TOTAL_ITERATIONS'] = 3
            execute(ns, 15)
            self.assertEqual(ns['global_counter'], 3)
            history = pd.read_csv(ns['VAL_LOG_FILE'], sep='\t')
            self.assertEqual(history.step.tolist(), [0, 1, 2, 3])
            with patch.dict(os.environ, {'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': tmp}):
                execute(ns, 17)
                ns['plt'].close('all')
            # A resumed exhausted patience counter must prevent further updates.
            ns['monitor'].bad_checks = ns['monitor'].patience
            ns['TOTAL_ITERATIONS'] = 100
            execute(ns, 15)
            self.assertEqual(ns['global_counter'], 3)
            # Legacy / incompatible configs must never be silently resumed.
            ns['DROPOUT'] = 0.2
            with self.assertRaisesRegex(RuntimeError, 'configs mismatch'):
                execute(ns, 11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
