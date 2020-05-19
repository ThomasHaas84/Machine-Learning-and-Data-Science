
# SEIR-Modell
# Covid-19 Vorhersage


import numpy as np
from scipy.integrate import odeint


class SEIR:
    def __init__(self,
                 incubation_period=5.2,
                 infective_period=2.9,
                 basic_reproduction_rate=3.4,
                 intervention_times=[23, 30, 230, 330],
                 p0=(83019212., 0., 1., 0.),
                 t_vals=np.arange(0., 365., 1.0)):
        self.t_inf = infective_period
        self.t_inc = incubation_period
        self.r_0 = basic_reproduction_rate
        self.t_vals = t_vals
        self.p0 = p0
        self.n = sum(self.p0)
        self.intervention_times = intervention_times


    def dS(self, t, susceptible, infectious, r_list):
        r = r_list[0]
        for i, intervention_time in enumerate(self.intervention_times):
            if t >= intervention_time:
                r = r_list[i + 1]
        return -1 * susceptible / self.n * (r / self.t_inf * infectious)


    def dE(self, t, susceptible, exposed, infectious, r_list):
        r = r_list[0]
        for i, intervention_time in enumerate(self.intervention_times):
            if t >= intervention_time:
                r = r_list[i + 1]
        return susceptible / self.n * (r / self.t_inf * infectious) - exposed / self.t_inc


    def dI(self, infectious):
        return infectious / self.t_inf


    def dR(self, infectious):
        return infectious / self.t_inf


    def system(self, y, t, r_list=None):
        susceptible, exposed, infectious, removed = y
        if r_list is None:
            r_list = [self.r0]*len(self.intervention_times)
        return [self.dS(t, susceptible, infectious, r_list),
                self.dE(t, susceptible, exposed, infectious, r_list),
                self.dI(exposed, infectious),
                self.dR(infectious)]


    def __call__(self, t, r0, r1, r2, r3, e0):
        return np.sum(self.getSEIR(t, [r0, r1, r2, r3, r1, r0], e0)[:, 1:], axis=1)


    def getSEIR(self, t, r_list, e0):
        S0, E0, I0, R0 = self.p0
        p0 = (self.n - E0 - I0 - E0, e0, I0, R0)
        return odeint(self.system, y0=p0, t=t, args=(r_list,))

