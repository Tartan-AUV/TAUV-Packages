import wandb


def delete_assets(project_name):
    # Initialize wandb
    wandb.init(project=project_name, mode='dryrun')

    # Fetch all runs for the project
    runs = wandb.Api().runs(path=f"tartanauv/{project_name}")

    # Delete assets for each run
    for run in runs:
        print(f"Deleting assets for run: {run.id}")
        wandb_run = wandb.Api().run(f"{project_name}/{run.id}")

        # List all assets for the run
        assets = wandb_run.files()

        # Delete each asset
        for asset in assets:
            print(f"Deleting asset: {asset.name}")
            asset.delete()

    print(f"All assets deleted for project: {project_name}")


# Example usage:
if __name__ == "__main__":
    project_name = "centernet"
    delete_assets(project_name)
