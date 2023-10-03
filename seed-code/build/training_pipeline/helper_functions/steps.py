from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
    CreateModelStep,
)
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.properties import PropertyFile
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics, FileSource
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.inputs import CreateModelInput
from sagemaker.clarify import BiasConfig, DataConfig, ModelConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig,
)


from helper_functions._utils import (
    _get_model_source_for_evaluation,
    _get_training_image,
    _get_training_inputs,
    _get_output_path,
)


def create_processing_step(
    role, bucket, configuration, input_data, PREPROCESSING_DIRECTORY
):
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=configuration["instance_type"],
        instance_count=1,
        base_job_name="Preprocessing-job",
    )

    # Use the sklearn_processor in a Sagemaker pipelines ProcessingStep
    step_preprocess_data = ProcessingStep(
        name=configuration["step_name"],
        processor=sklearn_processor,
        description=configuration["description"],
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=_get_output_path(bucket, "processing/train"),
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=_get_output_path(bucket, "processing/validation"),
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=_get_output_path(bucket, "processing/test"),
            ),
            ProcessingOutput(
                output_name="train_data_with_headers",
                source="/opt/ml/processing/train_data_with_headers",
                destination=_get_output_path(
                    bucket, "processing/train_data_with_headers"
                ),
            ),
            ProcessingOutput(
                output_name="data_baseline_with_headers",
                source="/opt/ml/processing/data_baseline_with_headers",
                destination=_get_output_path(
                    bucket, "processing/data_baseline_with_headers"
                ),
            ),
        ],
        code="{}/{}".format(PREPROCESSING_DIRECTORY, configuration["script"]),
    )

    return step_preprocess_data


def create_training_step(bucket, role, configuration, steps):
    # Fetch container to use for training
    image_uri = _get_training_image()

    xgb_estimator = Estimator(
        image_uri=image_uri,
        output_path=_get_output_path(bucket, "training"),
        instance_type=configuration["instance_type"],
        instance_count=configuration["num_instances"],
        role=role,
        disable_profiler=True,
    )

    xgb_estimator.set_hyperparameters(
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.8,
        objective="binary:logistic",
        num_round=25,
    )

    inputs = _get_training_inputs(steps["preprocessing_step"])

    step_train_model = TrainingStep(
        name=configuration["step_name"],
        description=configuration["description"],
        estimator=xgb_estimator,
        inputs=inputs,
    )

    return step_train_model


def create_tune_step(bucket, role, configuration, steps):
    # Fetch container to use for training
    image_uri = _get_training_image()

    # Create XGBoost estimator object
    # The object contains information about what container to use, what instance type etc.
    estimator = Estimator(
        image_uri=image_uri,
        output_path=_get_output_path(bucket, "training"),
        instance_type=configuration["instance_type"],
        instance_count=configuration["num_instances"],
        role=role,
        disable_profiler=True,
    )

    # Create Hyperparameter tuner object. Ranges from https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:auc",
        hyperparameter_ranges={
            "eta": ContinuousParameter(0, 0.5),
            "alpha": ContinuousParameter(0, 1000),
            "min_child_weight": ContinuousParameter(1, 120),
            "max_depth": IntegerParameter(1, 10),
            "num_round": IntegerParameter(1, 2000),
            "subsample": ContinuousParameter(0.5, 1),
        },
        max_jobs=configuration["max_jobs"],
        max_parallel_jobs=configuration["parallel_jobs"],
    )

    inputs = _get_training_inputs(steps["preprocessing_step"])

    # use the tuner in a SageMaker pipielines tuning step.
    step_tuning = TuningStep(
        name=configuration["step_name"],
        tuner=tuner,
        inputs=inputs,
    )

    return step_tuning


def create_evaluate_step(
    role, bucket, configuration, isTuneStep, steps, EVALUATION_DIRECTORY
):
    image_uri = _get_training_image()

    # Create ScriptProcessor object.
    # The object contains information about what container to use, what instance type etc.
    evaluate_model_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=configuration["instance_type"],
        instance_count=1,
        base_job_name="evaluation-job",
        role=role,
    )

    # Create a PropertyFile
    # A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
    # For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    # Use the evaluate_model_processor in a Sagemaker pipelines ProcessingStep.
    step_evaluate_model = ProcessingStep(
        name=configuration["step_name"],
        processor=evaluate_model_processor,
        inputs=[
            ProcessingInput(
                source=_get_model_source_for_evaluation(steps, isTuneStep, bucket),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=steps["preprocessing_step"]
                .properties.ProcessingOutputConfig.Outputs["test"]
                .S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=_get_output_path(bucket, "evaluation/evaluation-report"),
            ),
            ProcessingOutput(
                output_name="sample_payload",
                source="/opt/ml/processing/sample",
                destination=_get_output_path(bucket, "sample_payload"),
            ),
        ],
        code="{}/{}".format(EVALUATION_DIRECTORY, configuration["script"]),
        property_files=[evaluation_report],
    )

    return step_evaluate_model


def create_data_baseline_step(role, bucket, steps):
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
    )

    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=steps["preprocessing_step"]
        .properties.ProcessingOutputConfig.Outputs["data_baseline_with_headers"]
        .S3Output.S3Uri,
        dataset_format=DatasetFormat.csv(header=True, output_columns_position="START"),
        output_s3_uri=_get_output_path(bucket, "data_baseline"),
    )

    data_quality_check_step = QualityCheckStep(
        name="data-baseline-step",
        skip_check=True,
        register_new_baseline=True,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
    )

    return data_quality_check_step


# def create_explainability_step(step, pipeline_steps, prefix, role, model_step):
def create_explainability_step(role, bucket, steps, step_create_model):
    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
    )

    model_config = ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type="ml.m5.xlarge",
    )

    s3_analysis_config_output_path = "s3://{}/explainability/analysis-cfg".format(
        bucket
    )

    # s3_analysis_config_output_path = _get_output_path(bucket, "explainability")

    model_explainability_data_config = DataConfig(
        s3_data_input_path=steps["preprocessing_step"]
        .properties.ProcessingOutputConfig.Outputs["train_data_with_headers"]
        .S3Output.S3Uri,
        s3_output_path=_get_output_path(bucket, "explainability"),
        s3_analysis_config_output_path=s3_analysis_config_output_path,
        label=0,
        dataset_type="text/csv",
    )

    shap_config = SHAPConfig(seed=123, num_samples=10)

    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )

    model_explainability_check_step = ClarifyCheckStep(
        name="explainability-baseline-step",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=True,
        register_new_baseline=True,
    )

    return model_explainability_check_step, model_explainability_check_config


def create_model_step(
    role, bucket, isTuneStep, steps, model_registry_name, min_accuracy
):
    imageUri = _get_training_image()
    pipeline_session = PipelineSession()

    model_data = _get_model_source_for_evaluation(steps, isTuneStep, bucket)

    model = Model(
        image_uri=imageUri,
        model_data=model_data,
        role=role,
        sagemaker_session=pipeline_session,
    )

    inputs = CreateModelInput(
        instance_type="ml.m5.xlarge",
    )

    # This step is used to create the model for use with the explainability step
    step_create_model = CreateModelStep(
        name="Create-Model-Step",
        model=model,
        inputs=inputs,
    )

    # Create model explainability step
    explainability_step, model_explainability_check_config = create_explainability_step(
        role, bucket, steps, step_create_model
    )

    # Create the data baseline step.
    data_baseline_step = create_data_baseline_step(role, bucket, steps)

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    steps["evaluation_step"]
                    .properties.ProcessingOutputConfig.Outputs["evaluation"]
                    .S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=explainability_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
    )

    baseline = DriftCheckBaselines(
        model_data_statistics=MetricsSource(
            s3_uri=data_baseline_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_baseline_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        explainability_constraints=MetricsSource(
            s3_uri=explainability_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        explainability_config_file=FileSource(
            s3_uri=model_explainability_check_config.monitoring_analysis_config_uri,
            content_type="application/json",
        ),
    )
    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_registry_name,
        model_metrics=model_metrics,
        drift_check_baselines=baseline,
        customer_metadata_properties={
            "sample_payload": "True",
            "my_custom_key": "Some super relevant and important information",
        },
        sample_payload_url=Join(
            on="/",
            values=[
                steps["evaluation_step"]
                .properties.ProcessingOutputConfig.Outputs["sample_payload"]
                .S3Output.S3Uri,
                "payload.csv",
            ],
        ),
    )

    step_register_model = ModelStep(
        name="model-step",
        step_args=register_args,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    # Create accuracy condition to ensure the model meets performance requirements.
    # Models with a test accuracy lower than the condition will not be registered with the model registry.
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=steps["evaluation_step"].name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",
        ),
        right=float(min_accuracy),
    )

    # Create a Sagemaker Pipelines ConditionStep, using the condition above.
    # Enter the steps to perform if the condition returns True / False.
    step_cond = ConditionStep(
        name="Accuracy-Condition",
        conditions=[cond_gte],
        if_steps=[
            data_baseline_step,
            step_register_model,
            step_create_model,
            explainability_step,
        ],
        else_steps=[],
    )

    return step_cond
