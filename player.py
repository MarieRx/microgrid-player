# python 3
# this class combines all basic features of a generic player
import numpy as np
import pandas as pd
import pulp
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
		self.delta_t=0.5
		self.rho_d=0.95
		self.pmax_station=40


	def set_scenario(self, scenario_data):
		self.data = scenario_data
		# depart et arr contiennent respectivement la liste des heures de départ et d'arrivée des véhicules à la date d
		d="01/01/2014"
		self.depart=list(scenario_data[(scenario_data["day"] == d)]["time_slot_dep"][:p.nb_slow + p.nb_fast])
		self.arr = list(scenario_data[(scenario_data["day"] == d)]["time_slot_arr"][:p.nb_slow + p.nb_fast])

	def set_prices(self, prices):
		self.prices = prices
		#print(prices)

	def compute_all_load(self):
		lp = pulp.LpProblem("opti_station", pulp.LpMinimize)
		lp.setSolver()
		# création des variables:
		l_V2G = {}
		temp={}
		for i in range(4):
			l_V2G[i] = {}
			temp[i]={}
			for j in range(48):
				l_V2G[i][j] = pulp.LpVariable("l_V2G" + str(i) + "_" + str(j))
				temp[i][j] = pulp.LpVariable("temp" + str(i) + "_" + str(j))
		charge= {}
		for i in range(4):
			charge[i] = {}
			for j in range(48):
				charge[i][j] = pulp.LpVariable("charge" + str(i) + "_" + str(j))

		# contraintes:

		for i in range(4):
			for t in range(47):
				lp+=temp[i][t]>=0,"temp_sup"+str(i)+str(t)
				lp+=temp[i][t]>=l_V2G[i][t],"temp_inf"+str(i)+str(t)

				#lp += charge[i][t + 1] - charge[i][t] - self.rho_c * self.delta_t * temp[i][t]== 0, "contraintechargement"+str(i)+str(t)
				lp += charge[i][t + 1] - charge[i][t] - self.rho_c * self.delta_t * temp[i][t]+ (1 / self.rho_d) * (temp[i][t]-l_V2G[i][t]) == 0,  "contraintechargement"+str(i)+str(t)

		for t in range(48):
			for i in range(2):
				lp += l_V2G[i][t] - self.pslow <= 0, "chargemaxvoiture_slow"+str(i)+str(t)
				lp += l_V2G[i][t] + self.pslow >= 0, "chargemaxvoiture_slow_abs" + str(i) + str(t)
			for i in range(2,4):
				lp += l_V2G[i][t] - self.pfast <= 0, "chargemaxvoiture_fast" + str(i+self.nb_slow) + str(t)
				lp += l_V2G[i][t] + self.pfast >= 0, "chargemaxvoiture_fast_abs" + str(i+self.nb_slow)  + str(t)
		for t in range(48):
			lp += pulp.lpSum(l_V2G[i][t] for i in range(4))-self.pmax_station <= 0, "chargemaxstation"+str(t)
			lp += pulp.lpSum(l_V2G[i][t] for i in range(4)) + self.pmax_station >= 0, "chargemaxstationinf" + str(t)


		for i in range(4):
			for t in range(self.depart[i]*2,self.arr[i]*2):
				#lp+=charge[i][t]==0,"rienenjournee"+str(i)+str(t)
				lp+=l_V2G[i][t]==0,"rienen journee2"+str(i)+str(t)
			for t in range(self.arr[i]*2-1,self.horizon):
				lp+=l_V2G[i][t]<=0,"reinjection le soir"+str(t)+str(i)
		for i in range(4):
			for t in range(48):
				lp += charge[i][t] >= 0, "charge inf" + str(i) + str(t)
				lp += charge[i][t] <= 40, "chargesup" + str(i) + str(t)
			lp += charge[i][self.depart[i]*2] - 4 - charge[i][self.arr[i]*2] == 0, "contraintedecharge"+str(i)
			lp += charge[i][0] == 0, "contraintedechargeminuit"+str(i)
			lp += self.capacite - charge[i][self.depart[i]*2] <= 0, "contraintechargedepart"+str(i)
		lp.setObjective(pulp.lpSum((pulp.lpSum(self.prices[t] * l_V2G[i][t] for i in range(4)) for t in range(48))))
		lp.solve()
		load=np.zeros(self.horizon)
		for i in range(self.horizon):
			load[i]=l_V2G[0][i].varValue+l_V2G[1][i].varValue+l_V2G[2][i].varValue+l_V2G[3][i].varValue
		#for v in lp.variables():
		#	print(v.name, "=", v.varValue)
		#print(load)
		l=np.zeros((4,self.horizon))
		for t in range(48):
			for i in range(4):
				l[i][t]=l_V2G[i][t].varValue

		c = np.zeros((4, self.horizon))
		for t in range(48):
			for i in range(4):
				c[i][t]=charge[i][t].varValue
		print(c)
		return l

	def take_decision(self, time):
		# TO BE COMPLETED
		return 0

	def compute_load(self, time):
		load = self.take_decision(time)
		return load

	def reset(self):
		# reset all observed data
		pass


#if __name__ == " __main__ ":
"""
f=pd.read_csv("ev_scenarios.csv",";")
p=Player()
p.__init__()
p.set_scenario(f)
p.set_prices(random_lambda)

l=p.compute_all_load()
"""
#fonction de cout qui ne prend pas encore en compte les amendes si les voitures ne sont pas chargées à temps
def cout(p,l):
	c=0
	for time in range(48):
		c+=l[time]*p[time]
	return c
#print(cout(p.prices,l))
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:19:33 2021

@author: B57876
"""

# Check feasibility of different player loads
# 2OJ
# ATTENTION CONVENTION ARR/DEP pour la station de charge ; dans mon code les arr/dep
# sont des pas de tps dans un modèle à tps discret ; dans les données d'entrée c'est en heures...
import numpy as np

wrong_format_score = 1000
pu_infeas_score = 0.1
default_pu_infeas_score = 0.01  # when 0 value is the one to be obtained,


# no relative deviation can be calc.

def calculate_infeas_score(n_infeas_check: int, infeas_list: list,
						   n_default_infeas: int) -> float:
	"""
    Calculate infeasibility score

    :param n_infeas_check: number of infeasibility check (number of constraints
    to be respected)
    :param infeas_list: list of infeasibility values, relatively to the NONZERO
    values to be respected
    :param n_default_infeas: number of infeas. corresponding to ZERO values to
    be respected
    """

	return np.sum(infeas_list) / n_infeas_check * pu_infeas_score \
		   + n_default_infeas * default_pu_infeas_score





def check_charging_station_feasibility(load_profiles: np.ndarray, n_ev_normal_charging: int,
									   n_ev_fast_charging: int, t_ev_dep: np.ndarray,
									   t_ev_arr: np.ndarray, ev_max_powers: dict,
									   ev_batt_capa: np.ndarray, charge_eff: float,
									   discharge_eff: float, n_ts: int,
									   delta_t_s: int, dep_soc_penalty: float,
									   cs_max_power: float) -> (float, float):
	"""
    Check EV load profiles obtained from the charging station module

    :param load_profiles: matrix with a line per EV charging profile. EVs with
    normal charging power are provided first, then EV with fast charging techno
    :param n_ev_normal_charging: number of EVs with normal charging power
    :param n_ev_fast_charging: idem with fast charging power
    :param t_ev_dep: time-slots of dep.
    :param t_ev_arr: idem for arr (after dep. here, back from work)
    :param ev_max_powers: dict. with keys the type of EV ("normal" or "fast")
    and values the associated max charging power
    :param ev_batt_capa: EV battery capacity
    :param charge_eff: charging efficiency
    :param discharge_eff: discharging efficiency
    :param n_ts: number of time-slots
    :param delta_t_s: time-slot duration, in seconds
    :param dep_soc_penalty: value of the penalty to be added to the objective if
    EV SoC at departure is below 25% of battery capa
    :param cs_max_power: charging station max. power
    :return: returns the obj. penalty (for not being charged at a minimum SOC
    of 4kWh at dep.) and the infeasibility score
    """

	if not (isinstance(load_profiles, np.ndarray) and load_profiles.shape[
		0] == n_ev_normal_charging + n_ev_fast_charging \
			and load_profiles.shape[1] == n_ts):
		print("Wrong format for charging station (per EV) load profiles, should be (%i,%i)" \
			  % (n_ev_normal_charging + n_ev_fast_charging, n_ts))

		return None, wrong_format_score, {}

	infeas_list = []
	n_default_infeas = 0
	n_infeas_by_type = {"ev_max_p": 0, "charge_out_of_cs": 0, "soc_max_bound": 0,
						"soc_min_bound": 0, "min_soc_at_dep": 0, "cs_max_power": 0}
	n_infeas_check = 0  # number of constraints checked (to normalize
	# the infeas. score at the end)

	# check
	# 1. that indiv. charging powers respect the indiv. max. power limit
	# normal charging EVs
	for i_ev in range(n_ev_normal_charging):
		infeas_list.extend(list(np.maximum(np.abs(load_profiles[i_ev, :]) \
										   - ev_max_powers["normal"], 0) / ev_max_powers["normal"]))
		n_infeas_check += n_ts
		# update infeas by type
		n_infeas_by_type["ev_max_p"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])

	# fast charging EVs
	for i_ev in range(n_ev_fast_charging):
		infeas_list.extend(list(np.maximum(np.abs(load_profiles[i_ev, :]) \
										   - ev_max_powers["fast"], 0) / ev_max_powers["fast"]))
		n_infeas_check += n_ts
		# update infeas by type
		n_infeas_by_type["ev_max_p"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])
	# 0 charging power when EV is not connected (convention that EV leave at the end
	# of time-slot t_dep and arrive at the beginning of t_arr -> can charge in both ts)
	for i_ev in range(n_ev_normal_charging + n_ev_fast_charging):
		n_charge_out_of_cs = \
			len(np.where(np.abs(load_profiles[i_ev, t_ev_dep[i_ev] + 1:t_ev_arr[i_ev] - 1]) > 0)[0])
		n_default_infeas += n_charge_out_of_cs
		# update infeas by type
		n_infeas_by_type["charge_out_of_cs"] += n_charge_out_of_cs

	# 2. that SoC bounds of each EV is respected, as well as min. charging need at dep.
	cs_dep_soc_penalty = 0
	for i_ev in range(n_ev_normal_charging + n_ev_fast_charging):
		current_batt_soc = (charge_eff * np.cumsum(np.maximum(load_profiles[i_ev, :], 0)) \
							- discharge_eff * np.cumsum(np.maximum(-load_profiles[i_ev, :], 0))) \
						   * delta_t_s / 3600
		# diminish SoC when arriving at CS with E quantity consumed when driving
		current_batt_soc[t_ev_arr[i_ev]] -= 4

		# max bound (EV batt. capa)
		infeas_list.extend(list(np.maximum(current_batt_soc
										   - ev_batt_capa[i_ev], 0) / ev_batt_capa[i_ev]))
		n_infeas_check += n_ts
		n_infeas_by_type["soc_max_bound"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])

		# min bound (0)
		n_soc_below_zero = len(np.where(current_batt_soc < 0)[0])
		n_default_infeas += n_soc_below_zero
		n_infeas_by_type["soc_min_bound"] += n_soc_below_zero

		# SoC at dep. is above the minimal level requested
		if current_batt_soc[t_ev_dep[i_ev]] < 0.25 * ev_batt_capa[i_ev]:
			cs_dep_soc_penalty += dep_soc_penalty
			n_infeas_by_type["min_soc_at_dep"] += 1
		n_infeas_check += 1

	# 3.that CS power is below the max allowed value
	infeas_list.extend(list(np.maximum(np.abs(np.sum(load_profiles, axis=0)) \
									   - cs_max_power, 0) / cs_max_power))
	n_infeas_check += n_ts
	n_infeas_by_type["cs_max_power"] += len(np.where(np.array(infeas_list[-n_ts:]) > 0)[0])

	# calculate infeasibility score
	infeas_score = calculate_infeas_score(n_infeas_check, infeas_list, n_default_infeas)

	return cs_dep_soc_penalty, infeas_score, n_infeas_by_type




if __name__ == "__main__":
	# general temporal parameters
	n_ts = 48
	delta_t_s = 1800

	import pandas as pd
	import copy

	scenario_data = pd.read_csv("ev_scenarios.csv", sep=";", decimal=".")

	# TEST: charging station feasibility test -> from the code of one of your classmates
	#from code_eleves.code_franchino_opti_class_5 import Player

	p = Player()
	p.set_scenario(scenario_data)

	prices_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7,
				   8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3]
	p.set_prices(prices_test)

	#resultat = p.compute_all_load()
	l = p.compute_all_load()

	# from arr/dep in hours to time-slots
	t_ev_dep = np.array([int(3600 / delta_t_s * elt) for elt in p.depart])
	t_ev_arr = np.array([int(3600 / delta_t_s * elt) for elt in p.arr])

	cs_dep_soc_penalty, cs_infeas_score, n_infeas_by_type = \
		check_charging_station_feasibility(np.array(l), 2, 2, t_ev_dep, t_ev_arr,
										   {"normal": 3, "fast": 22}, 40 * np.ones(4),
										   0.95, 0.95, 48, delta_t_s, 5, 40)
	print("res code Franchino: ", cs_dep_soc_penalty, cs_infeas_score, n_infeas_by_type)

	# TEST: Data Center feas. check -> from randomly generated data
	cop_cs = 4 + 1
	cop_hp = 60 / (60 - 35) * 0.5
	eer = 4
	dc_load_profile = np.random.rand(n_ts)
	it_load_profile = np.random.rand(n_ts)
	dc_load_profile_good = np.random.rand(n_ts) \
						   * cop_cs / (eer * (cop_hp - 1) * delta_t_s) * it_load_profile
	dc_load_profile_bad = copy.deepcopy(dc_load_profile_good)
	dc_load_profile_bad[24] = cop_cs / (eer * (cop_hp - 1) * delta_t_s) * it_load_profile[24] + 1
	#dc_infeas_score = check_data_center_feasibility(dc_load_profile, cop_cs,
													#cop_hp, eer, n_ts, delta_t_s,
													#it_load_profile)
	#dc_infeas_score_good = check_data_center_feasibility(dc_load_profile_good, cop_cs,
													#	 cop_hp, eer, n_ts, delta_t_s,
													#	 it_load_profile)
	#dc_infeas_score_bad = check_data_center_feasibility(dc_load_profile_bad, cop_cs,
													#	cop_hp, eer, n_ts, delta_t_s,
													#	it_load_profile)
	#print("DC infeas score random prof.: ", dc_infeas_score)
	#print("DC infeas score good prof.: ", dc_infeas_score_good)
	#print("DC infeas score bad prof.: ", dc_infeas_score_bad)

	# TEST: Industrial Site feas. check -> from randomly generated data
	batt_capa = 60
	batt_max_power = 10
	is_load_profile_good = np.zeros(n_ts)
	for t in range(n_ts):
		is_load_profile_good[t] = \
			min(batt_max_power,
				(batt_capa - np.sum(is_load_profile_good[:t]) * delta_t_s / 3600) / 2)
	is_infeas_score_good, n_infeas_by_type = \
		check_industrial_cons_feasibility(is_load_profile_good, batt_capa,
										  batt_max_power, 0.95, 0.95, n_ts, delta_t_s)
	print("IS infeas score good prof.: ", is_infeas_score_good)

# TEST: Solar Farm identical to Industrial Site... not done here but function
# available above

"""
p=np.zeros((self.horizon,self.nb_fast+self.nb_slow))
		for t in range(self.horizon):
			for i in range (self.nb_fast+self.nb_slow):
				if t<self.depart[i]*2 or t>self.arr[i]*2:
					p[t][i]=1
				else:
					p[t][i]=0
		#print(p)

		load = np.zeros(self.horizon)
		# l est une liste temporaire, on reprend ses valeurs et on les réordonne avant de les mettre dans load

		# on construit l en remplissant d'abord les voitures "slow" puis les "fast" pour chaque pas de temps
		l = np.zeros(self.horizon)

		# chargement prend en compte le chargement en cours pour chacun des véhicules
		chargement = np.zeros(self.nb_slow + self.nb_fast)

		for time in range(self.horizon):
			# consom<40
			consom = 0

			for i in range(self.nb_slow):
				plus = self.rho_c * min(self.pslow, min((10 - chargement[i]) / self.rho_c, 40 - consom)) * self.delta_t
				chargement[i] += plus
				consom += plus

			for i in range(self.nb_slow, self.nb_slow + self.nb_fast):
				plus = self.rho_c * min(self.pfast, min((10 - chargement[i]) / self.rho_c, 40 - consom)) * self.delta_t
				chargement[i] += plus
				consom += plus
			l[time] = consom

		# on réordonne les consomations en mettant à chaque fois la conso la plus importante sur le prix le faible restant
		m = np.min(self.depart) * 2
		copie_prix = self.prices[:m].copy()
		cpt = 0
		while cpt < m:
			arg_min = np.argmin(copie_prix)
			arg_max = np.argmax(l)
			load[arg_min] = l[arg_max]
			copie_prix[arg_min] = np.inf
			l[arg_max] = 0
			cpt += 1
		# load[time] = self.compute_load(time)
		for i in range(self.nb_slow + self.nb_fast):
			chargement[i] -= 4
		for time in range(self.horizon):
			# consom<40
			consom = 0

			for i in range(self.nb_slow):
				plus = min(self.pslow, min(chargement[i] * self.rho_d, 40 - consom)) * self.delta_t / self.rho_d
				chargement[i] -= plus
				consom += plus

			for i in range(self.nb_slow, self.nb_slow + self.nb_fast):
				plus = min(self.pfast, min(chargement[i] * self.rho_d, 40 - consom)) * self.delta_t / self.rho_d
				chargement[i] -= plus
				consom += plus
			l[time] = consom
		# on réordonne les consomations en mettant à chaque fois la conso la plus importante sur le prix le faible restant
		m = np.max(self.arr) * 2
		copie_prix = self.prices[m:].copy()
		cpt = m
		while cpt < self.horizon:
			arg_max_p = np.argmax(copie_prix) + m
			arg_max = np.argmax(l)
			load[arg_max_p] = -l[arg_max]
			copie_prix[arg_max_p - m] = -np.inf
			l[arg_max] = 0
			cpt += 1
		# print(load)
"""