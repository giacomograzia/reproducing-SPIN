## Testing General Capabilities Pipeline

The followoing comand can be run to test the general capabilities of each model, for one benchmark at a time:
```bash
python eval_v2.py --model_name ${MODEL_LABEL} --model_path ${MODEL_PATH} --benchmark ${BENCHMARK} --task_type ${TASK_TYPE}
```

### Possible parameters

For `Qwen` model:
```bash
MODEL_LABEL="qwen2-7b-instruct"
MODEL_PATH="Qwen/Qwen2-7B-Instruct"
```

For `Mistral` model:
```bash
MODEL_LABEL="mistral-7b-instruct-v0.2"
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
```

For `Vicuna` model:
```bash
MODEL_LABEL="vicuna-7b-v1.5"
MODEL_PATH="lmsys/vicuna-7b-v1.5"
```

Benchmark options:
```bash
BENCHMARK="hellaswag"
TASK_TYPE="mcq"

BENCHMARK="mmlu" 
TASK_TYPE="mcq"

BENCHMARK="winogrande"
TASK_TYPE="mcq"
```

Note, the logic for the following benchmarks also exists, however the `deepeval` library contained bugs in retriving the datasets in version `v3.8.1`, but later versions will probably fix this bug.
```bash
BENCHMARK="GSM8K"
TASK_TYPE="code"

BENCHMARK="boolq" 
TASK_TYPE="bool"

BENCHMARK="arc"
TASK_TYPE="math"
```

### Environment Check
Due to encountering several issues in the environment creation with conflicting packages, we built a script to verify it all works as expected in `test_env.py`.