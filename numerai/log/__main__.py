from tensorboard import program


def run_tensor_board():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', "lightning_logs"])
    url = tb.launch()
    print(f"TensorBoard listening on {url}")


if __name__ == "__main__":
    run_tensor_board()
    input()
