from mlflow import MlflowClient 
import mlflow
import os

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
EXPERIMENT_NAME = "demo"


client = MlflowClient()
experiment =  client.get_experiment_by_name(name=EXPERIMENT_NAME)
runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string='attributes.status != "FAILED" AND metrics.rmse < 9')

for run in runs:
    artifacts = client.list_artifacts(run_id=run.info.run_id)
    for artifact in artifacts:
        
        if artifact.path == "sklearn-model":
            # print(artifact)
            model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
            mv = mlflow.register_model(model_uri, "model")
    # client.update_run(run.info.run_id, "PROCESSED")

            
latest_versions = client.get_latest_versions(name="model", stages=['None'])
client.transition_model_version_stage("model", latest_versions[0].version, stage="Staging")
# print("Latest versions : ", latest_versions[0])

