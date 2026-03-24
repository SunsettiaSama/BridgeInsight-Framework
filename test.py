




from src.train_eval.deep_learning_module.run import train_from_configs

if __name__ == "__main__":

    dataset_config_path = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\datasets\annotation_dataset.yaml"
    trainer_config_path = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\trainer_sft.yaml"
    model_config_path = r"F:\Research\Vibration Characteristics In Cable Vibration\config\train\models\simple_mlp.yaml"

    train_from_configs(dataset_config_path, model_config_path, trainer_config_path)
