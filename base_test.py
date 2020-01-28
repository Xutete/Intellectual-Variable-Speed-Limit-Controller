import env as Env
import numpy as np
import argparse

def Florida(obs) -> float:
    if obs['flow'] > 1500:
        if obs['speed'] <= 12.5:
            return 11.11
        elif obs['speed'] <= 15.28:
            return 13.89
        elif obs['speed'] <= 18.06:
            return 16.67
        elif obs['speed'] <= 20.83:
            return 19.44
        else:
            return 22.22
    else:
        if obs['occ'] <= 15:
            return 22.22
        else:
            if obs['speed'] <=12.5:
                return 11.11
            elif obs['speed'] <= 15.28:
                return 13.89
            elif obs['speed'] <=18.06:
                return 16.67
            elif obs['speed'] <= 20.83:
                return 19.44
            else:
                return 22.22

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sum", default=1000, type=int, dest='total', help='total episodes')
    args = parser.parse_args()
    env = Env.SumoEnv(frameskip=180,evaluation=True)
    rule_based = np.zeros(args.total)
    fixed_max = np.zeros(args.total)
    fixed_min = np.zeros(args.total)
    fixed_med = np.zeros(args.total)

    
    for idx in range(args.total):
        seed = np.random.randint(20000000,30000000)
        print("--------Stage %d--------" % (idx+1))
        #Fixed maximum
        print("\t--------Fixed Maximum--------")
        env.reset(eval_seed=seed)
        episode_reward=0.0
        while True:
            _, reward, done, _ =env.step(22.22)
            episode_reward += reward
            if done:
                fixed_max[idx]=episode_reward
                break
        
        
        #Rule based
        print("\t--------Rule Based-------")
        env.reset(eval_seed=seed)
        episode_reward=0.0
        a = 22.22
        while True:
            _, reward, done, obs =env.step(a)
            #print(a)
            episode_reward += reward
            a = Florida(obs)
            if done:
                rule_based[idx]=episode_reward
                break 
    
    np.save('./logs/fixed_max.npy',fixed_max)
    np.save('./logs/rule_based.npy',rule_based)

    '''#Fixed minimum
    print("--------Fixed Minimum--------")
    for idx in range(args.total):
        print("\t--------Stage %d--------" % (idx+1))
        env.reset()
        episode_reward=0.0
        while True:
            _, reward, done, _ =env.step(11.11)
            episode_reward += reward
            if done:
                fixed_min[idx]=episode_reward
                break
    np.save('./logs/fixed_min.npy',fixed_min)'''

    