import numpy as np

NEURON_LOCATING_RESULTS = {
    
    "Qwen": {

        "Origin":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 617, 'awareness_nums': 1548, 'awareness_rate': np.float64(0.715)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 153, 'awareness_nums': 504, 'awareness_rate': np.float64(0.7671)}},

        "Random":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 616, 'awareness_nums': 1549, 'awareness_rate': np.float64(0.7155)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 151, 'awareness_nums': 506, 'awareness_rate': np.float64(0.7702)}},

        "Wanda":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 607, 'awareness_nums': 1558, 'awareness_rate': np.float64(0.7196)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 136, 'awareness_nums': 521, 'awareness_rate': np.float64(0.793)}},

        "SparseGPT":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 620, 'awareness_nums': 1545, 'awareness_rate': np.float64(0.7136)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 132, 'awareness_nums': 525, 'awareness_rate': np.float64(0.7991)}},

        "Importance Score":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 473, 'awareness_nums': 1692, 'awareness_rate': np.float64(0.7815)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 103, 'awareness_nums': 554, 'awareness_rate': np.float64(0.8432)}},
        
        "Gradient x Activation"	:{'fairness': {'no_following_nums': 0, 'unawareness_nums': 624, 'awareness_nums': 1541, 'awareness_rate': np.float64(0.7118)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 122, 'awareness_nums': 535, 'awareness_rate': np.float64(0.8143)}}
    },

    "Mistral": {

        "Origin":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 679, 'awareness_nums': 1486, 'awareness_rate': np.float64(0.6864)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 186, 'awareness_nums': 471, 'awareness_rate': np.float64(0.7169)}},
        
        "Random":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 794, 'awareness_nums': 1371, 'awareness_rate': np.float64(0.6333)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 360, 'awareness_nums': 297, 'awareness_rate': np.float64(0.4521)}},
        
        "Wanda": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 625, 'awareness_nums': 1540, 'awareness_rate': np.float64(0.7113)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 168, 'awareness_nums': 489, 'awareness_rate': np.float64(0.7443)}},
        
        "SparseGPT"	:{'fairness': {'no_following_nums': 0, 'unawareness_nums': 629, 'awareness_nums': 1536, 'awareness_rate': np.float64(0.7095)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 166, 'awareness_nums': 491, 'awareness_rate': np.float64(0.7473)}},
        
        "Importance Score":	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 626, 'awareness_nums': 1539, 'awareness_rate': np.float64(0.7109)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 165, 'awareness_nums': 492, 'awareness_rate': np.float64(0.7489)}},
        
        "Gradient x Activation"	:{'fairness': {'no_following_nums': 0, 'unawareness_nums': 626, 'awareness_nums': 1539, 'awareness_rate': np.float64(0.710)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 176, 'awareness_nums': 481, 'awareness_rate': np.float64(0.7321)}}

    },

    "Vicuna": {
        "Origin":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 753, 'awareness_nums': 1412, 'awareness_rate': np.float64(0.6522)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 351, 'awareness_nums': 306, 'awareness_rate': np.float64(0.4658)}},
        
        "Random":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 794, 'awareness_nums': 1371, 'awareness_rate': np.float64(0.6333)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 360, 'awareness_nums': 297, 'awareness_rate': np.float64(0.4521)}},
        
        "Wanda":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 625, 'awareness_nums': 1540, 'awareness_rate': np.float64(0.7113)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 168, 'awareness_nums': 489, 'awareness_rate': np.float64(0.7443)}},
        
        "SparseGPT"	:{'fairness': {'no_following_nums': 0, 'unawareness_nums': 629, 'awareness_nums': 1536, 'awareness_rate': np.float64(0.7095)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 166, 'awareness_nums': 491, 'awareness_rate': np.float64(0.7473)}},
        
        "Importance Score":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 747, 'awareness_nums': 1418, 'awareness_rate': np.float64(0.655)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 332, 'awareness_nums': 325, 'awareness_rate': np.float64(0.4947)}},
        
        "Gradient x Activation":{'fairness': {'no_following_nums': 0, 'unawareness_nums': 867, 'awareness_nums': 1298, 'awareness_rate': np.float64(0.5995)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 328, 'awareness_nums': 329, 'awareness_rate': np.float64(0.5008)}}
    }
}



