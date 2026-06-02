import numpy as np

# Multi-seed SPIN results for limited data variance analysis.
# Outer key: dataset size N. Value: list of per-seed result dicts (seeds 0–4).
# Each entry: {'fairness': {'awareness_rate': float}, 'privacy': {'awareness_rate': float}}
# Paste the awareness_rate values from spin_eval_${N}_seed${S}.log summary lines here.
SPIN_MULTISEED_RESULTS = {
    10: [
        {'fairness': {'awareness_rate': 0.7635}, 'privacy': {'awareness_rate': 0.8721}},  # seed 0
        {'fairness': {'awareness_rate': 0.7566}, 'privacy': {'awareness_rate': 0.7915}},  # seed 1
        {'fairness': {'awareness_rate': 0.7871}, 'privacy': {'awareness_rate': 0.8615}},  # seed 2
        {'fairness': {'awareness_rate': 0.7778}, 'privacy': {'awareness_rate': 0.8113}},  # seed 3
        {'fairness': {'awareness_rate': 0.7261}, 'privacy': {'awareness_rate': 0.8463}},  # seed 4
    ],
    25: [
        {'fairness': {'awareness_rate': 0.7880}, 'privacy': {'awareness_rate': 0.8584}},  # seed 0
        {'fairness': {'awareness_rate': 0.7497}, 'privacy': {'awareness_rate': 0.8021}},  # seed 1
        {'fairness': {'awareness_rate': 0.7700}, 'privacy': {'awareness_rate': 0.8326}},  # seed 2
        {'fairness': {'awareness_rate': 0.8092}, 'privacy': {'awareness_rate': 0.8478}},  # seed 3
        {'fairness': {'awareness_rate': 0.7580}, 'privacy': {'awareness_rate': 0.8356}},  # seed 4
    ],
    50: [
        {'fairness': {'awareness_rate': 0.7898}, 'privacy': {'awareness_rate': 0.8326}},  # seed 0
        {'fairness': {'awareness_rate': 0.7709}, 'privacy': {'awareness_rate': 0.8021}},  # seed 1
        {'fairness': {'awareness_rate': 0.7594}, 'privacy': {'awareness_rate': 0.8417}},  # seed 2
        {'fairness': {'awareness_rate': 0.7866}, 'privacy': {'awareness_rate': 0.8280}},  # seed 3
        {'fairness': {'awareness_rate': 0.7159}, 'privacy': {'awareness_rate': 0.7763}},  # seed 4
    ],
    75: [
        {'fairness': {'awareness_rate': 0.7774}, 'privacy': {'awareness_rate': 0.8402}},  # seed 0
        {'fairness': {'awareness_rate': 0.7626}, 'privacy': {'awareness_rate': 0.7945}},  # seed 1
        {'fairness': {'awareness_rate': 0.7714}, 'privacy': {'awareness_rate': 0.8417}},  # seed 2
        {'fairness': {'awareness_rate': 0.7714}, 'privacy': {'awareness_rate': 0.8356}},  # seed 3
        {'fairness': {'awareness_rate': 0.7774}, 'privacy': {'awareness_rate': 0.8600}},  # seed 4
    ],
    100: [
        {'fairness': {'awareness_rate': 0.7917}, 'privacy': {'awareness_rate': 0.8569}},  # seed 0
        {'fairness': {'awareness_rate': 0.7672}, 'privacy': {'awareness_rate': 0.8371}},  # seed 1
        {'fairness': {'awareness_rate': 0.7667}, 'privacy': {'awareness_rate': 0.8295}},  # seed 2
        {'fairness': {'awareness_rate': 0.7861}, 'privacy': {'awareness_rate': 0.8524}},  # seed 3
        {'fairness': {'awareness_rate': 0.7663}, 'privacy': {'awareness_rate': 0.8250}},  # seed 4
    ],
    200: [
        {'fairness': {'awareness_rate': 0.7159}, 'privacy': {'awareness_rate': 0.7763}},  # seed 0
        {'fairness': {'awareness_rate': 0.7691}, 'privacy': {'awareness_rate': 0.8189}},  # seed 1
        {'fairness': {'awareness_rate': 0.7718}, 'privacy': {'awareness_rate': 0.8234}},  # seed 2
        {'fairness': {'awareness_rate': 0.7709}, 'privacy': {'awareness_rate': 0.8402}},  # seed 3
        {'fairness': {'awareness_rate': 0.7640}, 'privacy': {'awareness_rate': 0.8402}},  # seed 4
    ],
    400: [
        {'fairness': {'awareness_rate': 0.7764}, 'privacy': {'awareness_rate': 0.8280}},  # seed 0
        {'fairness': {'awareness_rate': 0.7778}, 'privacy': {'awareness_rate': 0.8387}},  # seed 1
        {'fairness': {'awareness_rate': 0.7824}, 'privacy': {'awareness_rate': 0.8341}},  # seed 2
        {'fairness': {'awareness_rate': 0.7824}, 'privacy': {'awareness_rate': 0.8478}},  # seed 3
        {'fairness': {'awareness_rate': 0.7727}, 'privacy': {'awareness_rate': 0.8387}},  # seed 4
    ],
    600: [
        {'fairness': {'awareness_rate': 0.7908}, 'privacy': {'awareness_rate': 0.8356}},  # seed 0
        {'fairness': {'awareness_rate': 0.7603}, 'privacy': {'awareness_rate': 0.8006}},  # seed 1
        {'fairness': {'awareness_rate': 0.7755}, 'privacy': {'awareness_rate': 0.8508}},  # seed 2
        {'fairness': {'awareness_rate': 0.7949}, 'privacy': {'awareness_rate': 0.8447}},  # seed 3
        {'fairness': {'awareness_rate': 0.7529}, 'privacy': {'awareness_rate': 0.8432}},  # seed 4
    ],
}

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