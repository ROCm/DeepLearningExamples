import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import subprocess
    return (subprocess,)


@app.cell
def _(subprocess):
    subprocess.run(["bash", "scripts/train.sh", "120", "--amp", "10"])
    return


@app.cell
def _(subprocess):
    subprocess.run(["bash", "scripts/predict.sh"])
    return


if __name__ == "__main__":
    app.run()
