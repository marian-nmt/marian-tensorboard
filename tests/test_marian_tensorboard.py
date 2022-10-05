#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from marian_tensorboard.marian_tensorboard import MarianLogParser


LOG_LINES = (
    "[2019-03-25 14:51:54] Ep. 1 : Up. 1000 : Sen. 1,269,755 : Cost 7.95987511 : Time 785.53s : 17289.59 words/s : L.r. 1.2500e-05",
    "[2019-03-25 16:37:33] [valid] Ep. 1 : Up. 5000 : ce-mean-words : 5.21277 : new best",
    "[2019-04-10 13:18:39] [valid] Ep. 9 : Up. 775000 : perplexity : 4.24812 : stalled 4 times (last best: 4.24112)",
    "2022-05-01 21:54:09 [2022-05-01 14:54:07 mpi:0] Ep. 24.151 : Up. 20785 : Sen. 646,829,638 : Cost 0.12538165 * 25,437,554 @ 2,305,265 after 24,150,553,337 : Time 21.29s : 1193424.53 words/s : gNorm 0.0552 : L.r. 6.5259e-04",
)

EXPECTED_OUTPUTS = (
    [
        ('scalar', 1553525514, 1000, 'train/epoch', 1),
        ('scalar', 1553525514, 1000, 'train/Cost', 7.95987511),
        ('scalar', 1553525514, 1000, 'train/update_sent', 1269755),
        ('scalar', 1553525514, 1000, 'train/total_sent', 1269755),
        ('scalar', 1553525514, 1000, 'train/learn_rate', 0.0000125),
    ],
    [
        ('scalar', 1553531853, 5000, 'valid/ce-mean-words', 5.21277),
        ('scalar', 1553531853, 5000, 'valid/ce-mean-words_stalled', 0),
    ],
    [
        ('scalar', 1554902319, 775000, 'valid/perplexity', 4.24812),
        ('scalar', 1554902319, 775000, 'valid/perplexity_stalled', 4),
    ],
    [
        ('scalar', 1651442049, 20785, 'train/epoch', 24.151),
        ('scalar', 1651442049, 20785, 'train/Cost', 0.12538165),
        ('scalar', 1651442049, 20785, 'train/effective_batch_size', 2305265),
        ('scalar', 1651442049, 20785, 'train/total_sentences', 646829638),
        ('scalar', 1651442049, 20785, 'train/total_labels', 24150553337),
        ('scalar', 1651442049, 20785, 'train/learning_rate', 0.00065259),
        ('scalar', 1651442049, 20785, 'train/gradient_norm', 0.0552),
        ('scalar', 1651442049, 20785, 'train/words_per_second', 1193424.53),
    ],
)


def test_marian_log_parser():
    """Test MarianLogParser"""
    parser = MarianLogParser()

    for line_no, log_line in enumerate(LOG_LINES):
        out_tuples = parser.parse_line(log_line)
        for out_no, out_tuple in enumerate(out_tuples):
            exp_tuple = EXPECTED_OUTPUTS[line_no][out_no]
            assert out_tuple == exp_tuple
