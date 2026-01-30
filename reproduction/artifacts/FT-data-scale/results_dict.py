import numpy as np

SPIN_EXTRA_RESULTS = {
    "spin_5" : {'fairness': {'no_following_nums': 0, 'unawareness_nums': 470, 'awareness_nums': 1695, 'awareness_rate': np.float64(0.7829)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 99, 'awareness_nums': 558, 'awareness_rate': np.float64(0.8493)}},

    "spin_4" : {'fairness': {'no_following_nums': 0, 'unawareness_nums': 495, 'awareness_nums': 1670, 'awareness_rate': np.float64(0.7714)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 102, 'awareness_nums': 555, 'awareness_rate': np.float64(0.8447)}},

    "spin_3" :	{'fairness': {'no_following_nums': 0, 'unawareness_nums': 514, 'awareness_nums': 1651, 'awareness_rate': np.float64(0.7626)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 103, 'awareness_nums': 554, 'awareness_rate': np.float64(0.8432)}},

    "spin_2" : {'fairness': {'no_following_nums': 0, 'unawareness_nums': 566, 'awareness_nums': 1599, 'awareness_rate': np.float64(0.7386)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 98, 'awareness_nums': 559, 'awareness_rate': np.float64(0.8508)}},

    "spin_1" : {'fairness': {'no_following_nums': 0, 'unawareness_nums': 537, 'awareness_nums': 1628, 'awareness_rate': np.float64(0.752)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 116, 'awareness_nums': 541, 'awareness_rate': np.float64(0.8234)}}
    }




SPIN_RESULTS = {
    "spin_1000": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 465,
            'awareness_nums': 1700,
            'awareness_rate': np.float64(0.7852)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 107,
            'awareness_nums': 550,
            'awareness_rate': np.float64(0.8371)
        }
    },
    "spin_600": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 463,
            'awareness_nums': 1702,
            'awareness_rate': np.float64(0.7861)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 108,
            'awareness_nums': 549,
            'awareness_rate': np.float64(0.8356)
        }
    },
    "spin_400": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 465,
            'awareness_nums': 1700,
            'awareness_rate': np.float64(0.7852)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 123,
            'awareness_nums': 534,
            'awareness_rate': np.float64(0.8128)
        }
    },
    "spin_200": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 475,
            'awareness_nums': 1690,
            'awareness_rate': np.float64(0.7806)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 96,
            'awareness_nums': 561,
            'awareness_rate': np.float64(0.8539)
        }
    },
    "spin_100": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 480,
            'awareness_nums': 1685,
            'awareness_rate': np.float64(0.7783)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 108,
            'awareness_nums': 549,
            'awareness_rate': np.float64(0.8356)
        }
    },
    "spin_75": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 513,
            'awareness_nums': 1652,
            'awareness_rate': np.float64(0.763)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 105,
            'awareness_nums': 552,
            'awareness_rate': np.float64(0.8402)
        }
    },
    "spin_50": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 476,
            'awareness_nums': 1689,
            'awareness_rate': np.float64(0.7801)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 115,
            'awareness_nums': 542,
            'awareness_rate': np.float64(0.825)
        }
    },
    "spin_25": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 484,
            'awareness_nums': 1681,
            'awareness_rate': np.float64(0.7764)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 98,
            'awareness_nums': 559,
            'awareness_rate': np.float64(0.8508)
        }
    },
    "spin_10": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 472,
            'awareness_nums': 1693,
            'awareness_rate': np.float64(0.782)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 89,
            'awareness_nums': 568,
            'awareness_rate': np.float64(0.8645)
        }
    }
}

# Placeholder FFT results; to be filled with actual evaluation results
FFT_RESULTS = {
    "fft_1000": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 1025, 'awareness_nums': 1140, 'awareness_rate': np.float64(0.5266)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 192, 'awareness_nums': 465, 'awareness_rate': np.float64(0.7078)}},
    "fft_600": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 616, 'awareness_nums': 1549, 'awareness_rate': np.float64(0.7155)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 147, 'awareness_nums': 510, 'awareness_rate': np.float64(0.7763)}},
    "fft_400": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 616, 'awareness_nums': 1549, 'awareness_rate': np.float64(0.7155)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 150, 'awareness_nums': 507, 'awareness_rate': np.float64(0.7717)}},
    "fft_200": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 617, 'awareness_nums': 1548, 'awareness_rate': np.float64(0.715)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 149, 'awareness_nums': 508, 'awareness_rate': np.float64(0.7732)}},
    "fft_100": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 616, 'awareness_nums': 1549, 'awareness_rate': np.float64(0.7155)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 148, 'awareness_nums': 509, 'awareness_rate': np.float64(0.7747)}},
    "fft_75": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 619, 'awareness_nums': 1546, 'awareness_rate': np.float64(0.7141)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 148, 'awareness_nums': 509, 'awareness_rate': np.float64(0.7747)}},
    "fft_50": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 617, 'awareness_nums': 1548, 'awareness_rate': np.float64(0.715)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 148, 'awareness_nums': 509, 'awareness_rate': np.float64(0.7747)}},
    "fft_25": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 616, 'awareness_nums': 1549, 'awareness_rate': np.float64(0.7155)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 149, 'awareness_nums': 508, 'awareness_rate': np.float64(0.7732)}},
    "fft_10": {'fairness': {'no_following_nums': 0, 'unawareness_nums': 615, 'awareness_nums': 1550, 'awareness_rate': np.float64(0.7159)}, 'privacy': {'no_following_nums': 0, 'unawareness_nums': 150, 'awareness_nums': 507, 'awareness_rate': np.float64(0.7717)}}
    # for num in [1000, 600, 400, 200, 100, 75, 50, 25, 10]
}


LORA_RESULTS = {
    "lora_1000": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 1075,
            'awareness_nums': 1090,
            'awareness_rate': np.float64(0.5035)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 138,
            'awareness_nums': 519,
            'awareness_rate': np.float64(0.79)
        }
    },
    "lora_600": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 1076,
            'awareness_nums': 1089,
            'awareness_rate': np.float64(0.503)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 114,
            'awareness_nums': 543,
            'awareness_rate': np.float64(0.8265)
        }
    },
    "lora_400": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 1060,
            'awareness_nums': 1105,
            'awareness_rate': np.float64(0.5104)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 177,
            'awareness_nums': 480,
            'awareness_rate': np.float64(0.7306)
        }
    },
    "lora_200": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 959,
            'awareness_nums': 1206,
            'awareness_rate': np.float64(0.557)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 158,
            'awareness_nums': 499,
            'awareness_rate': np.float64(0.7595)
        }
    },
    "lora_100": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 1080,
            'awareness_nums': 1085,
            'awareness_rate': np.float64(0.5012)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 173,
            'awareness_nums': 484,
            'awareness_rate': np.float64(0.7367)
        }
    },
    "lora_75": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 1049,
            'awareness_nums': 1116,
            'awareness_rate': np.float64(0.5155)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 215,
            'awareness_nums': 442,
            'awareness_rate': np.float64(0.6728)
        }
    },
    "lora_50": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 925,
            'awareness_nums': 1240,
            'awareness_rate': np.float64(0.5727)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 204,
            'awareness_nums': 453,
            'awareness_rate': np.float64(0.6895)
        }
    },
    "lora_25": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 729,
            'awareness_nums': 1436,
            'awareness_rate': np.float64(0.6633)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 159,
            'awareness_nums': 498,
            'awareness_rate': np.float64(0.758)
        }
    },
    "lora_10": {
        'fairness': {
            'no_following_nums': 0,
            'unawareness_nums': 642,
            'awareness_nums': 1523,
            'awareness_rate': np.float64(0.7035)
        },
        'privacy': {
            'no_following_nums': 0,
            'unawareness_nums': 146,
            'awareness_nums': 511,
            'awareness_rate': np.float64(0.7778)
        }
    }
}