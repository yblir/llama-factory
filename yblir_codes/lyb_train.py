from src.llamafactory.train.tuner import run_exp
import yaml


def main(yaml_path_):
    with open(yaml_path_, 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)
    run_exp(param)


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     run_exp()


if __name__ == "__main__":
    # yaml_path = '../examples/yblir_configs/llama3_lora_reward.yaml'
    yaml_path = '../examples/yblir_configs/llama3_lora_reward_raw.yaml'

    main(yaml_path)
