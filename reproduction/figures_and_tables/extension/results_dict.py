import numpy as np

MALICIOUS_DATA = {
    "SPIN": {
        "Qwen": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 487, 'awareness_nums': 1678, 'awareness_rate': np.float64(0.7603)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 90, 'awareness_nums': 567, 'awareness_rate': np.float64(0.6704)}
        },
        "Mistral": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 641, 'awareness_nums': 1524, 'awareness_rate': np.float64(0.585)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 159, 'awareness_nums': 498, 'awareness_rate': np.float64(0.6773)}
        },
        
        "Vicuna": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 964, 'awareness_nums': 1201, 'awareness_rate': np.float64(0.5009)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 382, 'awareness_nums': 275, 'awareness_rate': np.float64(0.226)},
        },
    },
    "Origin": {
        "Qwen": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 617, 'awareness_nums': 1548, 'awareness_rate': np.float64(0.6944)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 153, 'awareness_nums': 504, 'awareness_rate': np.float64(0.6479)},
        },
        "Mistral": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 679, 'awareness_nums': 1486, 'awareness_rate': np.float64(0.5641)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 186, 'awareness_nums': 471, 'awareness_rate': np.float64(0.6757)},
        },
        "Vicuna": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 753, 'awareness_nums': 1412, 'awareness_rate': np.float64(0.4735)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 351, 'awareness_nums': 306, 'awareness_rate': np.float64(0.2525)},
        },
    },
    "FFT": {
        "Qwen": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1237, 'awareness_nums': 928, 'awareness_rate': np.float64(0.4409)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 521, 'awareness_nums': 136, 'awareness_rate': np.float64(0.5601)}
        },
        "Mistral": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1165, 'awareness_nums': 1000, 'awareness_rate': np.float64(0.4559)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 500, 'awareness_nums': 157, 'awareness_rate': np.float64(0.4514)}
        },
        "Vicuna" :{
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1181, 'awareness_nums': 984, 'awareness_rate': np.float64(0.46)}, 
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 426, 'awareness_nums': 231, 'awareness_rate': np.float64(0.2657)}
            }
    },
    
    "LoRA": {
        "Qwen": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1229, 'awareness_nums': 936, 'awareness_rate': np.float64(0.4569)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 515, 'awareness_nums': 142, 'awareness_rate': np.float64(0.5445)}
        },
        
        "Mistral": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1089, 'awareness_nums': 1076, 'awareness_rate': np.float64(0.4547)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 455, 'awareness_nums': 202, 'awareness_rate': np.float64(0.5097)}
        },
        "Vicuna" : {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1120, 'awareness_nums': 1045, 'awareness_rate': np.float64(0.3993)}, 
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 438, 'awareness_nums': 219, 'awareness_rate': np.float64(0.2843)}
        }
    },
    
    "DoRA": {
        "Qwen": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1224, 'awareness_nums': 941, 'awareness_rate': np.float64(0.459)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 516, 'awareness_nums': 141, 'awareness_rate': np.float64(0.5447)}
        },
        "Mistral": {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1092, 'awareness_nums': 1073, 'awareness_rate': np.float64(0.4581)},
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 446, 'awareness_nums': 211, 'awareness_rate': np.float64(0.5087)}
        },
        "Vicuna" : {
            'safety': {'no_following_nums': 0, 'unawareness_nums': 1120, 'awareness_nums': 1045, 'awareness_rate': np.float64(0.4026)}, 
            'helpfulness': {'no_following_nums': 0, 'unawareness_nums': 441, 'awareness_nums': 216, 'awareness_rate': np.float64(0.2759)}
        }
    }
}
