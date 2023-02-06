import sagemaker
from sagemaker.inputs import TrainingInput


def _get_training_image():
    return sagemaker.image_uris.retrieve(
        framework="xgboost",
        region="eu-west-1",
        version="1.2-2",
        py_version="py3",
    )


def _get_training_inputs(preprocessing_step):
    inputs = {
        "train": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    }
    return inputs


def _get_model_source_for_evaluation(steps, isTuneStep, bucket):

    if isTuneStep:
        model_source = steps["train_tune_step"].get_top_model_s3_uri(
            top_k=0,
            s3_bucket=bucket,  # Bucket where to store artefacts.
        )
        return model_source

    elif not isTuneStep:
        return steps["train_tune_step"].properties.ModelArtifacts.S3ModelArtifacts

    else:
        raise Exception("Fetching the model_source went wrong. Uknown train step type")
