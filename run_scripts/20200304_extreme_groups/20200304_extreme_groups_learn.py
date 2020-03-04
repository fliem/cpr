"""
analysis restricted to subjects with more than two y sessions

"""
subjects = ['OAS30001', 'OAS30006', 'OAS30009', 'OAS30010', 'OAS30014', 'OAS30027', 'OAS30028', 'OAS30029', 'OAS30038',
            'OAS30040', 'OAS30049', 'OAS30050', 'OAS30051', 'OAS30060', 'OAS30061', 'OAS30072', 'OAS30073', 'OAS30076',
            'OAS30078', 'OAS30079', 'OAS30087', 'OAS30091', 'OAS30096', 'OAS30101', 'OAS30103', 'OAS30108', 'OAS30113',
            'OAS30115', 'OAS30117', 'OAS30119', 'OAS30137', 'OAS30140', 'OAS30142', 'OAS30143', 'OAS30145', 'OAS30150',
            'OAS30152', 'OAS30164', 'OAS30165', 'OAS30172', 'OAS30176', 'OAS30178', 'OAS30179', 'OAS30187', 'OAS30191',
            'OAS30194', 'OAS30202', 'OAS30205', 'OAS30207', 'OAS30209', 'OAS30214', 'OAS30216', 'OAS30220', 'OAS30223',
            'OAS30226', 'OAS30229', 'OAS30231', 'OAS30232', 'OAS30234', 'OAS30244', 'OAS30246', 'OAS30247', 'OAS30251',
            'OAS30255', 'OAS30257', 'OAS30262', 'OAS30265', 'OAS30267', 'OAS30282', 'OAS30285', 'OAS30287', 'OAS30291',
            'OAS30292', 'OAS30295', 'OAS30296', 'OAS30299', 'OAS30303', 'OAS30310', 'OAS30313', 'OAS30314', 'OAS30315',
            'OAS30316', 'OAS30318', 'OAS30320', 'OAS30321', 'OAS30324', 'OAS30325', 'OAS30326', 'OAS30331', 'OAS30333',
            'OAS30339', 'OAS30342', 'OAS30344', 'OAS30346', 'OAS30349', 'OAS30350', 'OAS30354', 'OAS30363', 'OAS30371',
            'OAS30373', 'OAS30379', 'OAS30382', 'OAS30383', 'OAS30388', 'OAS30390', 'OAS30391', 'OAS30392', 'OAS30399',
            'OAS30401', 'OAS30403', 'OAS30405', 'OAS30407', 'OAS30410', 'OAS30413', 'OAS30414', 'OAS30415', 'OAS30417',
            'OAS30419', 'OAS30421', 'OAS30431', 'OAS30436', 'OAS30438', 'OAS30449', 'OAS30457', 'OAS30464', 'OAS30470',
            'OAS30471', 'OAS30472', 'OAS30478', 'OAS30480', 'OAS30489', 'OAS30493', 'OAS30495', 'OAS30504', 'OAS30509',
            'OAS30514', 'OAS30518', 'OAS30521', 'OAS30525', 'OAS30531', 'OAS30533', 'OAS30541', 'OAS30542', 'OAS30547',
            'OAS30548', 'OAS30551', 'OAS30554', 'OAS30560', 'OAS30568', 'OAS30570', 'OAS30571', 'OAS30572', 'OAS30574',
            'OAS30576', 'OAS30582', 'OAS30583', 'OAS30584', 'OAS30587', 'OAS30589', 'OAS30595', 'OAS30598', 'OAS30601',
            'OAS30606', 'OAS30608', 'OAS30610', 'OAS30619', 'OAS30625', 'OAS30628', 'OAS30630', 'OAS30635', 'OAS30641',
            'OAS30642', 'OAS30651', 'OAS30653', 'OAS30654', 'OAS30655', 'OAS30658', 'OAS30659', 'OAS30662', 'OAS30670',
            'OAS30672', 'OAS30673', 'OAS30675', 'OAS30677', 'OAS30685', 'OAS30687', 'OAS30688', 'OAS30701', 'OAS30702',
            'OAS30703', 'OAS30707', 'OAS30710', 'OAS30722', 'OAS30724', 'OAS30725', 'OAS30729', 'OAS30731', 'OAS30732',
            'OAS30733', 'OAS30734', 'OAS30737', 'OAS30740', 'OAS30744', 'OAS30745', 'OAS30755', 'OAS30778', 'OAS30780',
            'OAS30782', 'OAS30785', 'OAS30787', 'OAS30801', 'OAS30802', 'OAS30803', 'OAS30804', 'OAS30805', 'OAS30812',
            'OAS30814', 'OAS30815', 'OAS30822', 'OAS30823', 'OAS30824', 'OAS30827', 'OAS30829', 'OAS30830', 'OAS30839',
            'OAS30841', 'OAS30845', 'OAS30848', 'OAS30859', 'OAS30864', 'OAS30866', 'OAS30869', 'OAS30870', 'OAS30871',
            'OAS30876', 'OAS30879', 'OAS30884', 'OAS30898', 'OAS30899', 'OAS30902', 'OAS30905', 'OAS30907', 'OAS30917',
            'OAS30919', 'OAS30923', 'OAS30926', 'OAS30927', 'OAS30928', 'OAS30929', 'OAS30933', 'OAS30936', 'OAS30939',
            'OAS30953', 'OAS30957', 'OAS30960', 'OAS30962', 'OAS30963', 'OAS30966', 'OAS30969', 'OAS30973', 'OAS30975',
            'OAS30976', 'OAS30977', 'OAS30983', 'OAS30990', 'OAS30996', 'OAS31002', 'OAS31003', 'OAS31007', 'OAS31009',
            'OAS31019', 'OAS31030', 'OAS31034', 'OAS31036', 'OAS31038', 'OAS31041', 'OAS31044', 'OAS31045', 'OAS31054',
            'OAS31057', 'OAS31059', 'OAS31063', 'OAS31068', 'OAS31071', 'OAS31072', 'OAS31074', 'OAS31084', 'OAS31086',
            'OAS31092', 'OAS31093', 'OAS31097', 'OAS31101', 'OAS31103', 'OAS31105', 'OAS31107', 'OAS31108', 'OAS31112',
            'OAS31113', 'OAS31123', 'OAS31124', 'OAS31125', 'OAS31129', 'OAS31133', 'OAS31134', 'OAS31139', 'OAS31140',
            'OAS31146', 'OAS31151', 'OAS31157', 'OAS31160', 'OAS31163', 'OAS31170']

print(len(subjects))
for s in ['OAS31137', 'OAS30853', 'OAS30237', 'OAS30788', 'OAS30979']:
    if s in subjects:
        subjects.remove(s)

# remove incorrect mr session OAS30872_MR_d1277 https://github.com/NrgXnat/oasis-scripts/blob/master/OASIS_CHANGELOG.md
if "OAS30872" in subjects:
    subjects.remove("OAS30872")

print(len(subjects))

from cpr.main import main
from pathlib import Path

main("/data/in",
     "/data/out",
     "group",
     participant_label=subjects,
     mr_baseline_sessions_file="/data/info/mr_baseline_sessions.tsv",
     stages=["learn"],
     learning_out_subdir="20200304_extreme_groups",
     clinical_feature_file=Path("/data/features/clinical_features_X_clinsessionn.pkl"),
     target_file=Path("/data/features/decliners_y.pkl"),
     fs_license_file="/data/misc/license.txt", test_run=False,
     n_splits=2,  # fixme 200
     model_name="basic+fmripca100",
     model_type="rfc+fmripca100",
     modalities=[
         'clinical',
         'structGlobScort',
         'structural',
         'fullcon',
         #
         'clinical+structGlobScort',
         'clinical+structural',
         'clinical+fullcon',
         #
         'clinical+structGlobScort+fullcon',
     ],
     verbose=True,
     n_jobs_outer=None)
