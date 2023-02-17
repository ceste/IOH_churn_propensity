# IOH_churn_propensity

python3 -m venv churn_propensity

source churn_propensity/bin/activate

python model.py -i input_folder -o output_folder


# if output_folder does not exist, create it
	  if not os.path.exists(output_folder_path):
		os.makedirs(output_folder_path)   
