"""
analysis restricted to subjects with more than two y sessions

"""
subjects = ['OAS30001', 'OAS30002', 'OAS30003', 'OAS30004', 'OAS30005', 'OAS30006', 'OAS30007', 'OAS30008', 'OAS30009',
            'OAS30010', 'OAS30011', 'OAS30013', 'OAS30014', 'OAS30015', 'OAS30018', 'OAS30025', 'OAS30026', 'OAS30027',
            'OAS30028', 'OAS30029', 'OAS30032', 'OAS30033', 'OAS30035', 'OAS30036', 'OAS30038', 'OAS30040', 'OAS30042',
            'OAS30044', 'OAS30046', 'OAS30048', 'OAS30049', 'OAS30050', 'OAS30051', 'OAS30052', 'OAS30058', 'OAS30060',
            'OAS30061', 'OAS30062', 'OAS30065', 'OAS30066', 'OAS30070', 'OAS30071', 'OAS30072', 'OAS30073', 'OAS30075',
            'OAS30076', 'OAS30078', 'OAS30079', 'OAS30083', 'OAS30084', 'OAS30087', 'OAS30088', 'OAS30091', 'OAS30095',
            'OAS30096', 'OAS30097', 'OAS30099', 'OAS30101', 'OAS30103', 'OAS30104', 'OAS30107', 'OAS30108', 'OAS30109',
            'OAS30113', 'OAS30115', 'OAS30117', 'OAS30119', 'OAS30121', 'OAS30122', 'OAS30126', 'OAS30127', 'OAS30128',
            'OAS30129', 'OAS30132', 'OAS30134', 'OAS30135', 'OAS30137', 'OAS30139', 'OAS30140', 'OAS30141', 'OAS30142',
            'OAS30143', 'OAS30145', 'OAS30146', 'OAS30149', 'OAS30150', 'OAS30152', 'OAS30156', 'OAS30157', 'OAS30160',
            'OAS30161', 'OAS30164', 'OAS30165', 'OAS30167', 'OAS30170', 'OAS30171', 'OAS30172', 'OAS30173', 'OAS30174',
            'OAS30175', 'OAS30176', 'OAS30178', 'OAS30179', 'OAS30180', 'OAS30184', 'OAS30185', 'OAS30187', 'OAS30191',
            'OAS30193', 'OAS30194', 'OAS30202', 'OAS30205', 'OAS30206', 'OAS30207', 'OAS30208', 'OAS30209', 'OAS30212',
            'OAS30214', 'OAS30216', 'OAS30218', 'OAS30220', 'OAS30223', 'OAS30224', 'OAS30226', 'OAS30229', 'OAS30231',
            'OAS30232', 'OAS30233', 'OAS30234', 'OAS30237', 'OAS30240', 'OAS30241', 'OAS30244', 'OAS30246', 'OAS30247',
            'OAS30248', 'OAS30250', 'OAS30251', 'OAS30253', 'OAS30255', 'OAS30256', 'OAS30257', 'OAS30258', 'OAS30261',
            'OAS30262', 'OAS30263', 'OAS30265', 'OAS30267', 'OAS30269', 'OAS30272', 'OAS30275', 'OAS30276', 'OAS30277',
            'OAS30280', 'OAS30282', 'OAS30283', 'OAS30285', 'OAS30287', 'OAS30291', 'OAS30292', 'OAS30295', 'OAS30296',
            'OAS30299', 'OAS30302', 'OAS30303', 'OAS30305', 'OAS30307', 'OAS30310', 'OAS30313', 'OAS30314', 'OAS30315',
            'OAS30316', 'OAS30317', 'OAS30318', 'OAS30320', 'OAS30321', 'OAS30322', 'OAS30324', 'OAS30325', 'OAS30326',
            'OAS30328', 'OAS30331', 'OAS30333', 'OAS30335', 'OAS30336', 'OAS30337', 'OAS30339', 'OAS30342', 'OAS30343',
            'OAS30344', 'OAS30346', 'OAS30349', 'OAS30350', 'OAS30352', 'OAS30354', 'OAS30355', 'OAS30357', 'OAS30361',
            'OAS30362', 'OAS30363', 'OAS30367', 'OAS30368', 'OAS30369', 'OAS30371', 'OAS30373', 'OAS30374', 'OAS30375',
            'OAS30378', 'OAS30379', 'OAS30382', 'OAS30383', 'OAS30387', 'OAS30388', 'OAS30390', 'OAS30391', 'OAS30392',
            'OAS30393', 'OAS30396', 'OAS30398', 'OAS30399', 'OAS30401', 'OAS30402', 'OAS30403', 'OAS30404', 'OAS30405',
            'OAS30406', 'OAS30407', 'OAS30410', 'OAS30411', 'OAS30413', 'OAS30414', 'OAS30415', 'OAS30417', 'OAS30418',
            'OAS30419', 'OAS30421', 'OAS30423', 'OAS30428', 'OAS30430', 'OAS30431', 'OAS30433', 'OAS30436', 'OAS30438',
            'OAS30443', 'OAS30449', 'OAS30451', 'OAS30452', 'OAS30455', 'OAS30456', 'OAS30457', 'OAS30458', 'OAS30464',
            'OAS30466', 'OAS30470', 'OAS30471', 'OAS30472', 'OAS30476', 'OAS30478', 'OAS30479', 'OAS30480', 'OAS30487',
            'OAS30489', 'OAS30490', 'OAS30491', 'OAS30493', 'OAS30495', 'OAS30498', 'OAS30499', 'OAS30500', 'OAS30503',
            'OAS30504', 'OAS30505', 'OAS30506', 'OAS30508', 'OAS30509', 'OAS30514', 'OAS30516', 'OAS30517', 'OAS30518',
            'OAS30521', 'OAS30525', 'OAS30528', 'OAS30529', 'OAS30531', 'OAS30532', 'OAS30533', 'OAS30534', 'OAS30535',
            'OAS30536', 'OAS30537', 'OAS30541', 'OAS30542', 'OAS30545', 'OAS30547', 'OAS30548', 'OAS30551', 'OAS30552',
            'OAS30553', 'OAS30554', 'OAS30555', 'OAS30557', 'OAS30558', 'OAS30559', 'OAS30560', 'OAS30562', 'OAS30564',
            'OAS30566', 'OAS30568', 'OAS30569', 'OAS30570', 'OAS30571', 'OAS30572', 'OAS30574', 'OAS30576', 'OAS30580',
            'OAS30581', 'OAS30582', 'OAS30583', 'OAS30584', 'OAS30586', 'OAS30587', 'OAS30589', 'OAS30591', 'OAS30592',
            'OAS30595', 'OAS30596', 'OAS30597', 'OAS30598', 'OAS30600', 'OAS30601', 'OAS30603', 'OAS30605', 'OAS30606',
            'OAS30608', 'OAS30610', 'OAS30612', 'OAS30614', 'OAS30615', 'OAS30619', 'OAS30620', 'OAS30624', 'OAS30625',
            'OAS30628', 'OAS30630', 'OAS30632', 'OAS30633', 'OAS30635', 'OAS30641', 'OAS30642', 'OAS30643', 'OAS30651',
            'OAS30652', 'OAS30653', 'OAS30654', 'OAS30655', 'OAS30656', 'OAS30657', 'OAS30658', 'OAS30659', 'OAS30660',
            'OAS30662', 'OAS30664', 'OAS30667', 'OAS30670', 'OAS30671', 'OAS30672', 'OAS30673', 'OAS30674', 'OAS30675',
            'OAS30676', 'OAS30677', 'OAS30683', 'OAS30685', 'OAS30687', 'OAS30688', 'OAS30692', 'OAS30701', 'OAS30702',
            'OAS30703', 'OAS30704', 'OAS30705', 'OAS30707', 'OAS30710', 'OAS30713', 'OAS30717', 'OAS30719', 'OAS30720',
            'OAS30722', 'OAS30723', 'OAS30724', 'OAS30725', 'OAS30726', 'OAS30727', 'OAS30729', 'OAS30731', 'OAS30732',
            'OAS30733', 'OAS30734', 'OAS30735', 'OAS30737', 'OAS30739', 'OAS30740', 'OAS30742', 'OAS30743', 'OAS30744',
            'OAS30745', 'OAS30746', 'OAS30748', 'OAS30749', 'OAS30750', 'OAS30751', 'OAS30752', 'OAS30754', 'OAS30755',
            'OAS30757', 'OAS30759', 'OAS30760', 'OAS30762', 'OAS30765', 'OAS30768', 'OAS30769', 'OAS30770', 'OAS30775',
            'OAS30776', 'OAS30777', 'OAS30778', 'OAS30780', 'OAS30781', 'OAS30782', 'OAS30785', 'OAS30787', 'OAS30788',
            'OAS30789', 'OAS30791', 'OAS30792', 'OAS30794', 'OAS30797', 'OAS30799', 'OAS30800', 'OAS30801', 'OAS30802',
            'OAS30803', 'OAS30804', 'OAS30805', 'OAS30806', 'OAS30808', 'OAS30810', 'OAS30812', 'OAS30814', 'OAS30815',
            'OAS30816', 'OAS30817', 'OAS30818', 'OAS30819', 'OAS30820', 'OAS30822', 'OAS30823', 'OAS30824', 'OAS30825',
            'OAS30827', 'OAS30828', 'OAS30829', 'OAS30830', 'OAS30832', 'OAS30835', 'OAS30836', 'OAS30839', 'OAS30840',
            'OAS30841', 'OAS30845', 'OAS30848', 'OAS30852', 'OAS30854', 'OAS30855', 'OAS30857', 'OAS30858', 'OAS30859',
            'OAS30861', 'OAS30863', 'OAS30864', 'OAS30866', 'OAS30867', 'OAS30869', 'OAS30870', 'OAS30871', 'OAS30872',
            'OAS30873', 'OAS30875', 'OAS30876', 'OAS30877', 'OAS30879', 'OAS30880', 'OAS30881', 'OAS30883', 'OAS30884',
            'OAS30885', 'OAS30888', 'OAS30890', 'OAS30896', 'OAS30897', 'OAS30898', 'OAS30899', 'OAS30902', 'OAS30903',
            'OAS30904', 'OAS30905', 'OAS30906', 'OAS30907', 'OAS30908', 'OAS30910', 'OAS30913', 'OAS30916', 'OAS30917',
            'OAS30919', 'OAS30920', 'OAS30921', 'OAS30923', 'OAS30924', 'OAS30926', 'OAS30927', 'OAS30928', 'OAS30929',
            'OAS30931', 'OAS30933', 'OAS30934', 'OAS30936', 'OAS30938', 'OAS30939', 'OAS30942', 'OAS30944', 'OAS30947',
            'OAS30949', 'OAS30953', 'OAS30956', 'OAS30957', 'OAS30959', 'OAS30960', 'OAS30962', 'OAS30963', 'OAS30964',
            'OAS30966', 'OAS30967', 'OAS30969', 'OAS30972', 'OAS30973', 'OAS30975', 'OAS30976', 'OAS30977', 'OAS30978',
            'OAS30979', 'OAS30982', 'OAS30983', 'OAS30985', 'OAS30989', 'OAS30990', 'OAS30991', 'OAS30992', 'OAS30993',
            'OAS30996', 'OAS30998', 'OAS31000', 'OAS31001', 'OAS31002', 'OAS31003', 'OAS31005', 'OAS31006', 'OAS31007',
            'OAS31009', 'OAS31012', 'OAS31013', 'OAS31014', 'OAS31019', 'OAS31023', 'OAS31024', 'OAS31025', 'OAS31028',
            'OAS31029', 'OAS31030', 'OAS31031', 'OAS31032', 'OAS31034', 'OAS31036', 'OAS31037', 'OAS31038', 'OAS31039',
            'OAS31041', 'OAS31043', 'OAS31044', 'OAS31045', 'OAS31046', 'OAS31047', 'OAS31048', 'OAS31052', 'OAS31054',
            'OAS31056', 'OAS31057', 'OAS31058', 'OAS31059', 'OAS31063', 'OAS31064', 'OAS31066', 'OAS31068', 'OAS31071',
            'OAS31072', 'OAS31073', 'OAS31074', 'OAS31077', 'OAS31083', 'OAS31084', 'OAS31085', 'OAS31086', 'OAS31088',
            'OAS31089', 'OAS31090', 'OAS31091', 'OAS31092', 'OAS31093', 'OAS31094', 'OAS31096', 'OAS31097', 'OAS31101',
            'OAS31103', 'OAS31104', 'OAS31105', 'OAS31107', 'OAS31108', 'OAS31110', 'OAS31111', 'OAS31112', 'OAS31113',
            'OAS31114', 'OAS31118', 'OAS31123', 'OAS31124', 'OAS31125', 'OAS31126', 'OAS31127', 'OAS31128', 'OAS31129',
            'OAS31131', 'OAS31132', 'OAS31133', 'OAS31134', 'OAS31136', 'OAS31138', 'OAS31139', 'OAS31140', 'OAS31141',
            'OAS31142', 'OAS31144', 'OAS31145', 'OAS31146', 'OAS31150', 'OAS31151', 'OAS31153', 'OAS31155', 'OAS31157',
            'OAS31158', 'OAS31159', 'OAS31160', 'OAS31163', 'OAS31164', 'OAS31166', 'OAS31168', 'OAS31170', 'OAS31172']

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

# fixme
best_cv_parameters_lcurve = \
    {'clinical': {'randomforestregressor__n_estimators': 256,
                  'randomforestregressor__max_features': 'sqrt',
                  'randomforestregressor__max_depth': 20.0,
                  'randomforestregressor__criterion': 'mse'},
     'clinical+fullcon': {'randomforestregressor__n_estimators': 256,
                          'randomforestregressor__max_features': 'sqrt',
                          'randomforestregressor__max_depth': 40.0,
                          'randomforestregressor__criterion': 'mae'},
     'clinical+structGlobScort': {'randomforestregressor__n_estimators': 256,
                                  'randomforestregressor__max_features': 'sqrt',
                                  'randomforestregressor__max_depth': 20.0,
                                  'randomforestregressor__criterion': 'mse'},
     'clinical+structGlobScort+fullcon': {'randomforestregressor__n_estimators': 256,
                                          'randomforestregressor__max_features': 'sqrt',
                                          'randomforestregressor__max_depth': 40.0,
                                          'randomforestregressor__criterion': 'mae'}
     }

main("/data/in",
     "/data/out",
     "group",
     participant_label=subjects,
     mr_baseline_sessions_file="/data/info/mr_baseline_sessions.tsv",
     stages=["learning_curve"],
     learning_out_subdir="20200304_full",
     clinical_feature_file=Path("/data/features/clinical_features_X_clinsessionn.pkl"),
     target_file=Path("/data/features/slopes_y.pkl"),
     fs_license_file="/data/misc/license.txt", test_run=False,
     n_splits=1000,
     model_name="basic+fmripca100",
     model_type="basic+fmripca100",
     modalities=[
         "clinical",
         #
         "clinical+structGlobScort",
         "clinical+fullcon",
         #
         "clinical+structGlobScort+fullcon"
     ],
     verbose=True,
     n_jobs_outer=None,
     best_cv_parameters_lcurve=best_cv_parameters_lcurve)
