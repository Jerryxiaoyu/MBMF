"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os.path as osp, shutil, time, atexit, os, subprocess
import shutil
import glob
import csv
from datetime import datetime
import pandas as pd

LOG_PATH = ''



def configure_log_dir(logname, txt='', copy = True):
	"""
	Set output directory to d, or to /tmp/somerandomnumber if d is None
	"""

	now = datetime.now().strftime("%b-%d_%H:%M:%S")
	path = os.path.join('log-files', logname, now + txt)
	os.makedirs(path)  # create path
	if copy:
		filenames = glob.glob('*.py')  # put copy of all python files in log_dir
		for filename in filenames:  # for reference
			shutil.copy(filename, path)
	return path




class Scaler(object):
	""" Generate scale and offset based on running mean and stddev along axis=0

		offset = running mean
		scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
	"""

	def __init__(self, obs_dim):
		"""
		Args:
			obs_dim: dimension of axis=1
		"""
		self.vars = np.zeros(obs_dim)
		self.means = np.zeros(obs_dim)
		self.m = 0
		self.n = 0
		self.first_pass = True

	def update(self, x):
		""" Update running mean and variance (this is an exact method)
		Args:
			x: NumPy array, shape = (N, obs_dim)

		see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
			   variance-of-two-groups-given-known-group-variances-mean
		"""
		if self.first_pass:
			self.means = np.mean(x, axis=0)
			self.vars = np.var(x, axis=0)
			self.m = x.shape[0]
			self.first_pass = False
		else:
			n = x.shape[0]
			new_data_var = np.var(x, axis=0)
			new_data_mean = np.mean(x, axis=0)
			new_data_mean_sq = np.square(new_data_mean)
			new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
			self.vars = (((self.m * (self.vars + np.square(self.means))) +
						  (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
						 np.square(new_means))
			self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
			self.means = new_means
			self.m += n

	def get(self):
		""" returns 2-tuple: (scale, offset) """
		return 1/(np.sqrt(self.vars) + 0.1)/3, self.means

	def preprocess(self, x):
		n = x.shape[0]
		self.means = np.mean(x, axis=0)
		self.vars =np.std(x,axis=0)

		#result = x - self.means

		result = (x - self.means) / (self.vars)
		result = np.nan_to_num(result)
		#result = (x - self.means) / np.max(np.abs(x - self.means), axis=0)
		return result

color2num = dict(
	gray=30,
	red=31,
	green=32,
	yellow=33,
	blue=34,
	magenta=35,
	cyan=36,
	white=37,
	crimson=38
)
def colorize(string, color, bold=False, highlight=False):
	attr = []
	num = color2num[color]
	if highlight: num += 10
	attr.append(str(num))
	if bold: attr.append('1')
	return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def log_info_write(path,str ):
    name = 'info.txt'
    f = open(path+'/'+name, 'a')
    f.write(str)
    f.write('\n')
    f.close()



class Logger(object):
	""" Simple training logger: saves to file and optionally prints to stdout """
	def __init__(self, logdir,csvname = 'log'):
		"""
		Args:
			logname: name for log (e.g. 'Hopper-v1')
			now: unique sub-directory name (e.g. date/time string)
		"""
		self.path = os.path.join(logdir, csvname+'.csv')
		self.write_header = True
		self.log_entry = {}
		self.f = open(self.path, 'w')
		self.writer = None  # DictWriter created with first call to write() method

	def write(self, display=True):
		""" Write 1 log entry to file, and optionally to stdout
		Log fields preceded by '_' will not be printed to stdout

		Args:
			display: boolean, print to stdout
		"""
		if display:
			self.disp(self.log_entry)
		if self.write_header:
			fieldnames = [x for x in self.log_entry.keys()]
			self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
			self.writer.writeheader()
			self.write_header = False
		self.writer.writerow(self.log_entry)
		self.log_enbtry = {}

	@staticmethod
	def disp(log):
		"""Print metrics to stdout"""
		log_keys = [k for k in log.keys()]
		log_keys.sort()
		'''
		print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
															   log['_MeanReward']))
		for key in log_keys:
			if key[0] != '_':  # don't display log items with leading '_'
				print('{:s}: {:.3g}'.format(key, log[key]))
		'''
		print('log writed!')
		print('\n')

	def log(self, items):
		""" Update fields in log (does not write to file, used to collect updates.

		Args:
			items: dictionary of items to update
		"""
		self.log_entry.update(items)

	def close(self):
		""" Close log file - log cannot be written after this """
		self.f.close()

	def log_table2csv(self,data, header = True):
		df = pd.DataFrame(data)
		df.to_csv(self.path, index=False, header=header)


	def log_csv2table(self):
		data = pd.read_csv(self.path,header = 0,encoding='utf-8')
		return np.array(data)