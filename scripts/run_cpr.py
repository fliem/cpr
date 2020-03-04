#! /usr/bin/env python

import os
from pathlib import Path
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter

from cpr.main import main


def get_parser():
    from cpr.__init__ import __version__

    verstr = 'cpr v{}'.format(__version__)

    parser = ArgumentParser(description='cpr', formatter_class=RawTextHelpFormatter)

    parser.add_argument('bids_dir', action='store',
                        help='the root folder of a BIDS valid dataset.')
    parser.add_argument('output_dir', action='store',
                        help='the output path')
    parser.add_argument('analysis_level', choices=['participant'],
                        help='processing stage to be run.')

    # optional arguments
    parser.add_argument('--version', action='version', version=verstr)

    g_general = parser.add_argument_group('General options')
    g_general.add_argument('--stages', action='store', nargs='+', choices=['preprocessing', 'feature_extraction',
                                                                           'prepare', 'learn'],
                           help=
                           """Preprocessing stages  
                           Participant-level stages:
                              - preprocesssing: preprocesses input data with fmriprep
                              - feature_extraction: extracts features from preprocessed data
                              - prepare: preprocesssing + feature_extraction
                              
                           Participant- or group-level stages:
                              - learn: collect features for multiple subjects and run learning
                           """
                           )
    g_general.add_argument('--participant_label', '--participant-label', action='store', nargs='+',
                           help='a space delimited list of participant identifiers or a single '
                                'identifier (without the sub- prefix)')

    g_general.add_argument('--session_label', '--session-label', action='store',
                           help='session')
    g_general.add_argument('--mr_baseline_sessions_file', '--mr-baseline-sessions-file', action='store',
                           help='mr baseline_sessions_file')
    g_general.add_argument('--clinical-feature-file', action='store', type=Path,
                           help='mr baseline_sessions_file (.pkl)')
    g_general.add_argument('--target-file', action='store', type=Path,
                           help='mr baseline_sessions_file (.pkl)')

    g_general.add_argument('--fs-license-file', metavar='PATH', type=os.path.abspath,
                           help='Path to FreeSurfer license key file. Get it (for free) by registering'
                                ' at https://surfer.nmr.mgh.harvard.edu/registration.html')
    g_general.add_argument('--modalities', action='store', nargs='+',
                           help='a space delimited list of modalities')
    g_general.add_argument('--n_cpus', '-n-cpus', action='store', type=int,
                           help='maximum number of threads across all processes')
    g_general.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False,
                           help="verbose output")

    g_test = parser.add_argument_group('Test options')

    g_test.add_argument("-t", "--test-run", dest="test_run", action="store_true", default=False,
                        help="Test run uses sloppy fmriprep preprocessing for speed")
    return parser


if __name__ == "__main__":
    opts = get_parser().parse_args()

    add_args = {}
    if opts.modalities:
        add_args["modalities"] = opts.modalities

    main(bids_dir=opts.bids_dir,
         output_dir=opts.output_dir,
         analysis_level=opts.analysis_level,
         stages=opts.stages,
         participant_label=opts.participant_label,
         session_label=opts.session_label,
         mr_baseline_sessions_file=opts.mr_baseline_sessions_file,
         clinical_feature_file=opts.clinical_feature_file,
         target_file=opts.target_file,
         fs_license_file=opts.fs_license_file,
         test_run=opts.test_run,
         n_cpus=opts.n_cpus,
         verbose=opts.verbose,
         **add_args
         )
