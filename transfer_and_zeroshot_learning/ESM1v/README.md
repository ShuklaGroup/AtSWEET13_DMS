# Zero-shot variant prediction with ESM1v

## Predict the effects of mutants using an ensemble of five ESM1v models
script.sh

'AtSWEET13_DMS_muts_labeled.csv' is the predicted score by ESM1v models

## Compare the predicted score with experimental score
'AtSWEET13_dms_exp_results_float_just_muts.npy' and 'AtSWEET13_dms_cons_results_float_just_muts.npy' are the experimentally reported expression score
and conservation scores.

The script 'ESM1v_res.py' is used for combining ESM1v score and experimental data, 'plot_DMS.py' is used for plotting ESM1v score and experimental data
