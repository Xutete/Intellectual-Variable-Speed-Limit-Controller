# Intellectual-Variable-Speed-Limit-Controller
## Introduction
This is a TRBAM2020 project. The main purpose is to construct a robust and intelligent speed limit control system for freeway management, with state-of-art machine learning techniques.

All rights of this project are reserved by college of Transportation Engineering at Tongji University.

>
>### College of Transportation Engineering at Tongji University
>
>Website: tjjt.tongji.edu.cn
>
>Team: 
>
>| Title               | Name | Homepage                                 |
>| ------------------- | ---- | ---------------------------------------- |
>| Professor | Yuxiong Ji  | [Info](https://its.tongji.edu.cn/index.php?classid=11627&t=show&id=61) |
>| Bachelor              | Juanwu Lu  | [Homepage](https://github.com/ChocolateDave) |
>| Graduate              | Yu Tang

## File Structure
```
.
├── lib               // Reference modules 
├── Project       // A freeway simulation scenario based on SUMO
│   ├── Output // Output logging files
│   ├── ramp.add.xml
│   ├── ramp.net.xml
│   ├── ramp.rou.xml
│   ├── ramp.sumo.cfg
│   ├── vsl.def.xml
│   └── viewsettings.xml
├── __init__.py
├── benchmark.py  // Benchmark module for research paper.
├── core.py             // Reinforcement learning agent.
├── param.py         //  Hyperparams for env and agent. 
└── env.py             // Reinforcement learning environment
```
