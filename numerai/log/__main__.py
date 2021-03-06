from tensorboard import program

from numerai.definitions import LOG_DIR


def run_tensor_board():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', LOG_DIR])
    url = tb.launch()
    print(f"TensorBoard listening on {url}")


if __name__ == "__main__":
    run_tensor_board()
    input()
