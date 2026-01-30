import numpy as np

MALICIOUS_DATA = {
    "SPIN": {
        "Qwen": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 487, 'awareness_nums': 1678, 'awareness_rate': np.float64(0.7751)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 90, 'awareness_nums': 567, 'awareness_rate': np.float64(0.863)}
        },
        "Mistral": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 641, 'awareness_nums': 1524, 'awareness_rate': np.float64(0.7039)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 159, 'awareness_nums': 498, 'awareness_rate': np.float64(0.758)}
        },
        
        "Vicuna": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 964, 'awareness_nums': 1201, 'awareness_rate': np.float64(0.5547)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 382, 'awareness_nums': 275, 'awareness_rate': np.float64(0.4186)}}
    },
        
        "Origin": {
        "Qwen": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 617, 'awareness_nums': 1548, 'awareness_rate': np.float64(0.715)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 153, 'awareness_nums': 504, 'awareness_rate': np.float64(0.7671)}
        },
        "Mistral": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 679, 'awareness_nums': 1486, 'awareness_rate': np.float64(0.6864)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 186, 'awareness_nums': 471, 'awareness_rate': np.float64(0.7169)}
        },
        
        "Vicuna": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 753, 'awareness_nums': 1412, 'awareness_rate': np.float64(0.6522)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 351, 'awareness_nums': 306, 'awareness_rate': np.float64(0.4658)}}
    },
    "FFT": {
        "Qwen": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1237, 'awareness_nums': 928, 'awareness_rate': np.float64(0.4286)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 521, 'awareness_nums': 136, 'awareness_rate': np.float64(0.207)}
        },
        "Mistral": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1165, 'awareness_nums': 1000, 'awareness_rate': np.float64(0.4619)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 500, 'awareness_nums': 157, 'awareness_rate': np.float64(0.239)}
        },
        "Vicuna" :{
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1181, 'awareness_nums': 984, 'awareness_rate': np.float64(0.4545)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 426, 'awareness_nums': 231, 'awareness_rate': np.float64(0.3516)}
            }
    },
    
    "LoRA": {
        "Qwen": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1229, 'awareness_nums': 936, 'awareness_rate': np.float64(0.4323)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 515, 'awareness_nums': 142, 'awareness_rate': np.float64(0.2161)}
        },
        "Mistral": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1089, 'awareness_nums': 1076, 'awareness_rate': np.float64(0.497)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 455, 'awareness_nums': 202, 'awareness_rate': np.float64(0.3075)}
        },
        "Vicuna" : {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1120, 'awareness_nums': 1045, 'awareness_rate': np.float64(0.4827)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 438, 'awareness_nums': 219, 'awareness_rate': np.float64(0.3333)}
        }
    },
    
    "DoRA": {
        "Qwen": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1224, 'awareness_nums': 941, 'awareness_rate': np.float64(0.4346)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 516, 'awareness_nums': 141, 'awareness_rate': np.float64(0.2146)}
        },
        "Mistral": {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1092, 'awareness_nums': 1073, 'awareness_rate': np.float64(0.4956)},
            'privacy': {'no_following_nums': 0, 'unawareness_nums': 446, 'awareness_nums': 211, 'awareness_rate': np.float64(0.3212)}
        },
        "Vicuna" : {
            'fairness': {'no_following_nums': 0, 'unawareness_nums': 1120, 'awareness_nums': 1045, 'awareness_rate': np.float64(0.4827)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 441, 'awareness_nums': 216, 'awareness_rate': np.float64(0.3288)}
        }
    }
}
