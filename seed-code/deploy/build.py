import argparse
import boto3
import yaml
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


sm_client = boto3.client("sagemaker")


def get_args(argparse):
    parser = argparse.ArgumentParser(
        "Creates or updates and runs the pipeline for the pipeline."
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


def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug(
                "Getting more packages for token: {}".format(response["NextToken"])
            )
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            logger.error(error_message)
            raise Exception(error_message)

        # Return the model package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        logger.info(
            f"Identified the latest approved model package: {model_package_arn}"
        )
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def get_baselines(model_package_arn):
    try:
        # get the baselines from Model Registry using ModelPackageName
        raw_baselines = sm_client.describe_model_package(
            ModelPackageName=model_package_arn
        ).get("DriftCheckBaselines", {})

        # re-format the baselines
        result = {
            key: {k: raw_baselines[key][k]["S3Uri"] for k in raw_baselines.get(key)}
            for key in raw_baselines
        }
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

    return result


def store_params(param_obj):
    with open("params.txt", "w") as f:
        for key, value in param_obj.items():
            f.writelines(f" {key}={value}")


if __name__ == "__main__":
    params = {}

    configuration = parse_yaml("./conf.yml")

    args = get_args(argparse)

    bucket = args["bucket"]
    if not bucket:
        raise Exception("Bucket needs to be provided")
    logger.info(bucket)
    params["Bucket"] = bucket

    role = configuration.get("role")
    if not role:
        raise Exception("Role ARN needs to be set in the configuration file.")
    logger.info(role)
    params["ModelExecutionRoleArn"] = role

    instance = configuration.get("EndpointInstanceType")
    params["EndpointInstanceType"] = instance

    project_name = args["project_name"]
    params["SageMakerProjectName"] = project_name

    model_registry_suffix = (
        configuration.get("model_registry_name", "registry") or "registry"
    )

    model_registry_name = f"{project_name}-{model_registry_suffix}"

    logger.info(model_registry_name)

    # Get latest approved model
    model_arn = get_approved_package(model_registry_name)
    logger.info(model_arn)
    params["ModelPackageName"] = model_arn

    baselines = get_baselines(model_arn)
    params["DataQualityConstraintsS3Uri"] = baselines.get("ModelDataQuality", {}).get(
        "Constraints", ""
    )
    params["DataQualityStatisticsS3Uri"] = baselines.get("ModelDataQuality", {}).get(
        "Statistics", ""
    )

    # Store params as .txt for use with SAM cli
    store_params(params)
