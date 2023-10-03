import argparse
import json
import yaml

from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline


from helper_functions.steps import (
    create_processing_step,
    create_training_step,
    create_tune_step,
    create_evaluate_step,
    create_data_baseline_step,
    create_model_step,
)

PREPROCESSING_DIRECTORY = "../algorithms/1-preprocessing"
TRAINING_DIRECTORY = "../algorithms/2-training"
EVALUATION_DIRECTORY = "../algorithms/3-evaluation"


class CustomPipeline:
    def __init__(self, model_registry_name, role, bucket):
        self.model_registry_name = model_registry_name
        self.steps = {}
        self.bucket = bucket
        self.role = role
        self.input_data = ParameterString(
            name="input_data",
            default_value="s3://bucket/key.csv",
        )
        self.isTuneStep = False

    def add_preprocessing_step(self):
        configuration = parse_yaml("{}/conf.yml".format(PREPROCESSING_DIRECTORY))

        preprocessing_step = create_processing_step(
            self.role,
            self.bucket,
            configuration,
            self.input_data,
            PREPROCESSING_DIRECTORY,
        )

        self.steps["preprocessing_step"] = preprocessing_step

        return

    def add_train_or_tune_step(self):
        configuration = parse_yaml("{}/conf.yml".format(TRAINING_DIRECTORY))

        if configuration["tune"]:
            self.isTuneStep = True
            train_tune_step = create_tune_step(
                self.bucket, self.role, configuration, self.steps
            )
        else:
            train_tune_step = create_training_step(
                self.bucket, self.role, configuration, self.steps
            )

        self.steps["train_tune_step"] = train_tune_step

        return

    def add_evaluation_step(self):
        configuration = parse_yaml("{}/conf.yml".format(EVALUATION_DIRECTORY))

        evaluation_step = create_evaluate_step(
            self.role,
            self.bucket,
            configuration,
            self.isTuneStep,
            self.steps,
            EVALUATION_DIRECTORY,
        )

        self.steps["evaluation_step"] = evaluation_step

        return

    def add_register_model_step(self, min_accuracy):
        model_step = create_model_step(
            self.role,
            self.bucket,
            self.isTuneStep,
            self.steps,
            self.model_registry_name,
            min_accuracy,
        )

        self.steps["model_step"] = model_step

        return

    def create_pipeline(self, tag_list, pipeline_configuration):
        pipeline = Pipeline(
            name=pipeline_configuration["pipeline_name"],
            parameters=[self.input_data],  # make this more dynamic
            steps=self.steps.values(),
        )

        res = pipeline.upsert(
            role_arn=self.role,
            description=pipeline_configuration["description"],
            tags=tag_list,
        )
        return res


def get_args(argparse):
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline."
    )

    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""{\"key\": \"value\", \"key2\": \"value2\"}""",
    )

    parser.add_argument(
        "-project-name",
        "--project-name",
        dest="project_name",
        default=None,
        help="String, SageMaker Project ID",
    )

    parser.add_argument(
        "-bucket",
        "--bucket",
        dest="bucket",
        default=None,
        help="Project bucket",
    )

    return vars(parser.parse_args())


def parse_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_tag_list(tags_from_argument, configuration):
    if tags_from_argument is not None:
        all_tags = json.loads(tags_from_argument) | configuration.get(
            "tags", {}
        )  # joins the argument provided tags with the ones from the configuration file.
    else:
        all_tags = configuration.get("tags", {})

    return [{"Key": key, "Value": value} for key, value in all_tags.items()]


if __name__ == "__main__":
    configuration = parse_yaml("./conf.yml")

    args = get_args(argparse)

    bucket = args["bucket"]
    if not bucket:
        raise Exception("Bucket needs to be provided")

    role = configuration.get("role")
    if not role:
        raise Exception("Role ARN needs to be set in the configuration file.")

    print("\n###### Role ARN:")
    print(role)

    project_name = args["project_name"]

    pipeline_suffix = (
        configuration.get("pipeline_name", "build-pipeline") or "build-pipeline"
    )
    pipeline_name = f"{project_name}-{pipeline_suffix}"

    configuration["pipeline_name"] = pipeline_name

    print("\n###### Pipeline name:")
    print(pipeline_name)

    model_registry_suffix = (
        configuration.get("model_registry_name", "registry") or "registry"
    )
    model_registry_name = f"{project_name}-{model_registry_suffix}"

    print("\n###### Model Registry Name:")
    print(model_registry_name)

    tag_list = get_tag_list(args["tags"], configuration)
    print("\n###### Tags:")
    print(tag_list)

    pipeline = CustomPipeline(model_registry_name, role, bucket)

    min_accuracy = configuration.get("min_accuracy", 0.7)

    pipeline.add_preprocessing_step()
    pipeline.add_train_or_tune_step()
    pipeline.add_evaluation_step()
    pipeline.add_register_model_step(float(min_accuracy))

    response = pipeline.create_pipeline(tag_list, configuration)
    print(response)
