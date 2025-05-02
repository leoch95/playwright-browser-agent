import typer

app = typer.Typer()

@app.command()
def chat():
    print("Starting chat mode...")

@app.command()
def batch(file: str):
    print(f"Starting batch mode with file: {file}")