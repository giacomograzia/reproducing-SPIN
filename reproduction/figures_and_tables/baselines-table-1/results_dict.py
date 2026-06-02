import numpy as np

REPRODUCTION_BASELINES = {
    "Spin": {
        "Qwen":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 465, 'awareness_nums': 1700, 'awareness_rate': np.float64(0.7852)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 107, 'awareness_nums': 550, 'awareness_rate': np.float64(0.8371)}},
        
        "Mistral":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 632, 'awareness_nums': 1533, 'awareness_rate': np.float64(0.7081)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 167, 'awareness_nums': 490, 'awareness_rate': np.float64(0.7458)}},
        
        "Vicuna":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 745, 'awareness_nums': 1420, 'awareness_rate': np.float64(0.6559)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 334, 'awareness_nums': 323, 'awareness_rate': np.float64(0.4916)}}
    },

"Origin": {
    "Qwen":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 617, 'awareness_nums': 1548, 'awareness_rate': np.float64(0.715)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 153, 'awareness_nums': 504, 'awareness_rate': np.float64(0.7671)}},
	
    "Mistral":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 679, 'awareness_nums': 1486, 'awareness_rate': np.float64(0.6864)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 186, 'awareness_nums': 471, 'awareness_rate': np.float64(0.7169)}},
	
    "Vicuna":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 753, 'awareness_nums': 1412, 'awareness_rate': np.float64(0.6522)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 351, 'awareness_nums': 306, 'awareness_rate': np.float64(0.4658)}}
},

"FFT" : {
    "Qwen":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 1025, 'awareness_nums': 1140, 'awareness_rate': np.float64(0.5266)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 192, 'awareness_nums': 465, 'awareness_rate': np.float64(0.7078)}},
        
    "Mistral": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 889, 'awareness_nums': 1276, 'awareness_rate': np.float64(0.5894)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 135, 'awareness_nums': 522, 'awareness_rate': np.float64(0.7945)}},
        
    "Vicuna":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 1084, 'awareness_nums': 1081, 'awareness_rate': np.float64(0.4993)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 350, 'awareness_nums': 307, 'awareness_rate': np.float64(0.4673)}},
},


"Lora" : {
    "Qwen": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 1076, 'awareness_nums': 1089, 'awareness_rate': np.float64(0.503)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 133, 'awareness_nums': 524, 'awareness_rate': np.float64(0.7976)}},
    	
    "Mistral":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 936, 'awareness_nums': 1229, 'awareness_rate': np.float64(0.5677)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 141, 'awareness_nums': 516, 'awareness_rate': np.float64(0.7854)}},
    
    "Vicuna":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 1139, 'awareness_nums': 1026, 'awareness_rate': np.float64(0.4739)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 358, 'awareness_nums': 299, 'awareness_rate': np.float64(0.4551)}}
},


"Dora":	{
    "Qwen":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 1063, 'awareness_nums': 1102, 'awareness_rate': np.float64(0.509)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 130, 'awareness_nums': 527, 'awareness_rate': np.float64(0.8021)}},
	
    "Mistral":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 973, 'awareness_nums': 1192, 'awareness_rate': np.float64(0.5506)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 139, 'awareness_nums': 518, 'awareness_rate': np.float64(0.7884)}},
	
    "Vicuna":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 1134, 'awareness_nums': 1031, 'awareness_rate': np.float64(0.4762)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 359, 'awareness_nums': 298, 'awareness_rate': np.float64(0.4536)}}
    }
}

ORIGINAL_PAPER_BASELINES = {
    "Origin": {
        "Qwen": {'fairness': {'awareness_rate': np.float64(0.6684)}, 'privacy': {'awareness_rate': np.float64(0.7412)}},
        "Mistral": {'fairness': {'awareness_rate': np.float64(0.6231)}, 'privacy': {'awareness_rate': np.float64(0.6636)}},
        "Vicuna": {'fairness': {'awareness_rate': np.float64(0.5501)}, 'privacy': {'awareness_rate': np.float64(0.3760)}}
    },
    "FFT": {
        "Qwen": {'fairness': {'awareness_rate': np.float64(0.5418)}, 'privacy': {'awareness_rate': np.float64(0.7900)}},
        "Mistral": {'fairness': {'awareness_rate': np.float64(0.5570)}, 'privacy': {'awareness_rate': np.float64(0.7793)}},
        "Vicuna": {'fairness': {'awareness_rate': np.float64(0.4046)}, 'privacy': {'awareness_rate': np.float64(0.5297)}}
    },
    "Lora": {
        "Qwen": {'fairness': {'awareness_rate': np.float64(0.4453)}, 'privacy': {'awareness_rate': np.float64(0.7656)}},
        "Mistral": {'fairness': {'awareness_rate': np.float64(0.5062)}, 'privacy': {'awareness_rate': np.float64(0.7473)}},
        "Vicuna": {'fairness': {'awareness_rate': np.float64(0.3857)}, 'privacy': {'awareness_rate': np.float64(0.4871)}}
    },
    "Dora": {
        "Qwen": {'fairness': {'awareness_rate': np.float64(0.4393)}, 'privacy': {'awareness_rate': np.float64(0.7793)}},
        "Mistral": {'fairness': {'awareness_rate': np.float64(0.4697)}, 'privacy': {'awareness_rate': np.float64(0.7047)}},
        "Vicuna": {'fairness': {'awareness_rate': np.float64(0.3783)}, 'privacy': {'awareness_rate': np.float64(0.4703)}}
    },
    "Spin": {
        "Qwen": {'fairness': {'awareness_rate': np.float64(0.7497)}, 'privacy': {'awareness_rate': np.float64(0.8447)}},
        "Mistral": {'fairness': {'awareness_rate': np.float64(0.6342)}, 'privacy': {'awareness_rate': np.float64(0.7154)}},
        "Vicuna": {'fairness': {'awareness_rate': np.float64(0.5778)}, 'privacy': {'awareness_rate': np.float64(0.4414)}}
    }
}