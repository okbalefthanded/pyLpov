cd "C:\pyLpov\scripts\standalone"

set data_file="C:\erp_experiment.csv"
set experiment="D:\laresi_bci\erp_calib.json"
set analysis_config="erp_config.yaml"
set save_folder="C:\my_erp_classifier"

python sa_hybrid_train.py %data_file%  %experiment% %analysis_config% %save_folder%

pause