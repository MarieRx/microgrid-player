# python 3
# this class combines all basic features of a generic player
import numpy as np
import pandas as pa
random_lambda = np.random.rand(48)
class Player:

	def __init__(self):
		# some player might not have parameters
		self.parameters = 0
		self.horizon = 48
		self.nb_slow=2
		self.nb_fast=2
		self.pslow=3
		self.pfast=22
		self.capacite=10
		self.rho_c=0.95


	def set_scenario(self, scenario_data):
		self.data = scenario_data
		arr_dep=list(scenario_data.values())[:self.nb_slow+self.nb_fast]

	def set_prices(self, prices):
		self.prices = prices

	def compute_all_load(self):
		load = np.zeros(self.horizon)
		chargement=np.zeros(self.nb_slow+self.nb_fast)
		for time in range(self.horizon):
			consom=0
			for i in range(self.nb_slow):
				chargement[i]+=self.rho_c*min(self.pslow,10-self.chargement[i],40-consom,0)
				consom+=chargement[i]

			for i in range(self.nb_slow,self.nb_slow+self.nb_fast):
				chargement[i]+=self.rho_c*min(self.pfast,10-self.chargement[i],40-consom,0)
				consom+=chargement[i]
			load[time]=consom
			#load[time] = self.compute_load(time)
		return load

	def take_decision(self, time):
		# TO BE COMPLETED
		return 0

	def compute_load(self, time):
		load = self.take_decision(time)

		return load

	def reset(self):
		# reset all observed data
		pass

file=pa.read_csv(ev_scenarios.csv)
print(list(file[0]) )