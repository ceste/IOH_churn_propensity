import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
import glob

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, average_precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight


import os,sys,inspect,getopt,io
import argparse
from pathlib import Path
import argparse


from log import Log
import config, utils

currentdir = os.path.dirname(os.path.realpath(__file__))
# print(f'currentdir {currentdir}')
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


class ChurnPrediction:

	def __init__(self):
		
		# path to csv file
		# self,data = None

		self.target = 'churn'

		self.uuid = utils.get_uuid()
		self.log = Log(self.uuid)		

		self.ordinal_encoders = None
		if os.path.isfile(currentdir+'/ordinal_encoder.pickle'): 
			self.ordinal_encoders = utils.load_obj(currentdir+'/ordinal_encoder.pickle')
			# print('self.ordinal_encoder {self.ordinal_encoder}')

		self.model = None
		if os.path.isfile(currentdir+'/rf.pickle'): 
			self.model = utils.load_obj(currentdir+'/rf.pickle')

		self.one_hot_encoders = None
		if os.path.isfile(currentdir+'/one_hot_encoders.pickle'): 
			self.one_hot_encoders = utils.load_obj(currentdir+'/one_hot_encoders.pickle')

		self.dic = None
		if os.path.isfile(currentdir+'/dic.pickle'): 
			self.dic = utils.load_obj(currentdir+'/dic.pickle')

		self.feature_names = None
		if os.path.isfile(currentdir+'/feature_names.pickle'): 
			self.feature_names = utils.load_obj(currentdir+'/feature_names.pickle')
		
		self.categorical_features = None
		if os.path.isfile(currentdir+'/categorical_features.pickle'): 
			self.categorical_features = utils.load_obj(currentdir+'/categorical_features.pickle')

		self.numerical_features = None
		if os.path.isfile(currentdir+'/numerical_features.pickle'): 
			self.numerical_features = utils.load_obj(currentdir+'/numerical_features.pickle')

		self.dic = None
		if os.path.isfile(currentdir+'/dic.pickle'): 
			self.dic = utils.load_obj(currentdir+'/dic.pickle')

		# print(self.dic)

		
	def prepare_data(self,data):

		data['start_date'] = data['start_date'].apply(lambda x:str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:9])
		data['start_date'] = pd.to_datetime(data['start_date'])

		data['end_date'] = data['end_date'].apply(lambda x:str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:9])
		data['end_date'] = pd.to_datetime(data['end_date'])

		data['last_dau_dt'] = data['last_dau_dt'].apply(lambda x:str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:9])
		data['last_dau_dt'] = pd.to_datetime(data['last_dau_dt'])

		data['home_page'].fillna(0,inplace=True) 
		data['home_page'] = data['home_page'].astype(int)

		data['tnr_app'].fillna(0,inplace=True) 
		data['tnr_app'] = data['tnr_app'].astype(int)

		data['days'] = data['last_dau_dt'] - data['start_date']
		data['days'] = data['days'].dt.days

		data['active_days'] = data['active_days'].astype(float)
		
		data['days_to_end_date'] = data['end_date'] - data['last_dau_dt']
		data['days_to_end_date'] = data['days_to_end_date'].dt.days

		data['pctg_active_days'] = data['active_days']/data['days']

		data.replace([np.inf, -np.inf], np.nan, inplace=True)
		data.fillna(0,inplace=True)

		for col in config.DROP_COLUMNS:
			if col in data.columns:
				data.drop(col, axis=1, inplace=True)
				
		
		return data

	def manage_categorical_features(self, data):

		for col in self.categorical_features:
	
			if col!=self.target:                            
				
				data[col] = self.ordinal_encoders[col].transform(data[col].astype(str).values.reshape(-1,1))        
				
				tmp = self.one_hot_encoders[col].transform(data[col].values.reshape(-1,1)).toarray()[:,1:]
				tmp_df = pd.DataFrame(tmp)
				tmp_df = pd.DataFrame(tmp, columns=utils.get_ohe_column_names(self.dic,col))
				
				data = pd.DataFrame(np.hstack([data,tmp_df]), columns=list(data.columns)+list(tmp_df.columns))
				
				del data[col]
		
		return data

	def get_ohe_column_names(self,dic,feature):

		return [feature+'_'+k for k,v in self.dic[feature].items() if v>0]

	def features_engineering(self,data):

		for col in self.categorical_features:
			
			if col!=self.target and col is not None and col!='None' and isinstance(col, str) and col in data.columns:                            

				data[col] = self.ordinal_encoders[col].transform(data[col].astype(str).values.reshape(-1,1))

				tmp = self.one_hot_encoders[col].transform(data[col].values.reshape(-1,1)).toarray()[:,1:]
				tmp_df = pd.DataFrame(tmp)
				tmp_df = pd.DataFrame(tmp, columns=self.get_ohe_column_names(self.dic,col))
				
				data = pd.DataFrame(np.hstack([data,tmp_df]), columns=list(data.columns)+list(tmp_df.columns))
				
				del data[col]

		return data

	def read_files(self, path):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()+'/'+path
		self.log.print_(msg)		

		all_files = glob.glob(path+"/*.csv")
		li = []
		for filename in all_files:
			df = pd.read_csv(filename, index_col=None, header=0)
			li.append(df)

		dfs = pd.concat(li, axis=0, ignore_index=True)
		dfs.reset_index(inplace=True)

		return dfs


	def predict(self, path):

		msg = self.__class__.__name__+'.'+utils.get_function_caller()
		self.log.print_(msg)		

		msg = f'path: {path}'
		self.log.print_(msg)		

		data = self.read_files(path)

		output_df = data[['msisdn']]
		data = self.prepare_data(data)
		data = self.features_engineering(data)
		data = data[self.feature_names]

		y_preds = self.model.predict(data)
		y_probs = self.model.predict_proba(data)[:, 1]

		output_df['label'] = y_preds
		output_df['prob'] = y_probs

		return output_df


	def train(self):

		pass



if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
	
	start = utils.get_time()
	# print(start)    

	parser = argparse.ArgumentParser()    
	parser.add_argument("--input_folder", "-i", help="State the folder name for the input data", required=True) 
	parser.add_argument("--output_folder", "-o", help="State the folder name for the output file", required=True) 
	args = parser.parse_args()

	input_folder = None
	if args.input_folder is None:
	  print("State the input_folder!!")
	else:
	  input_folder = args.input_folder

	output_folder = None
	output_folder_path = None
	if args.output_folder is None:
	  print("State the output_folder!!")
	else:
	  output_folder = args.output_folder
	  output_folder_path = currentdir+'/'+output_folder

	# if output_folder does not exist, create
	if output_folder_path is not None and not os.path.exists(output_folder_path):
	  os.makedirs(output_folder_path)   


	  
	if os.path.exists(input_folder):
		input_folder_path = currentdir+'/'+input_folder
		# download all files to the input folders
		pass 


	# predict    
	if os.path.exists(input_folder) and os.path.exists(output_folder):

		model = ChurnPrediction()
		# start the prediction        
		output = model.predict(input_folder)

		# generate random file name
		filename = utils.get_uuid()+'.csv'
		output_folder_path = output_folder_path+'/'+filename
		# print(f'output_folder_path {output_folder_path}')

		# write output
		output.to_csv(output_folder_path,index=False)

		print('process is done!!')

	else:

		print('process is broken!!')
	
