from llamafactory.train.tuner import run_exp
import yaml


def main():
    with open('../examples/yblir_configs/lyb_qwen_lora_sft.yaml', 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)
    run_exp(param)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
