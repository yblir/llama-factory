import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llamafactory.train.tuner import run_exp
import yaml


def main(yaml_path_):
    with open(yaml_path_, 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)
    run_exp(param)


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     run_exp()

# 7ef9b4fd324f9a6e50ac8dcd164ec2c6f8c30dbb
if __name__ == "__main__":
    # 训练奖励模型
    # yaml_path = '../examples/yblir_configs/llama3_lora_reward.yaml'
    # 奖励模型评分
    # yaml_path = '../examples/yblir_configs/llama3_lora_predict.yaml'
    # 训练sft模型
    # yaml_path = '../examples/yblir_configs/lyb_qwen_lora_sft.yaml'
    # ppo训练
    # yaml_path = '../examples/yblir_configs/llama3_lora_ppo.yaml'

    # dpo训练
    yaml_path='../examples/yblir_configs/yblir_lora_dpo.yaml'

    main(yaml_path)
