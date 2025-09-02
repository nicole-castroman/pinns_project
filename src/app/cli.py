import typer
from .train_runner import run_training

app = typer.Typer(help="PINN Heat App")

@app.command()
def train(config: str):
    """Train a model based on a YAML config."""
    run_training(config)

if __name__ == "__main__":
    app()
