import yaml

SIM_CFG = yaml.safe_load(open("configs/config.yaml"))




from sim.utils import qkd_simulation


# info=calculate_qkd_asymptotic_performance(0.6,0.1,0.0,0.6,0.3,0.1,10 ** (-0.2 * 50.0 / 10.0),0.2,1e-6,0.01,0.5,1.1,50e6,100000,None,{})

info =qkd_simulation(SIM_CFG,{},{
    "intercept_resend": 0.2,
    "pns": 0.5,
    "time_shift": 0.3,
    "darkcount_increase": 0.1
})

print(info)